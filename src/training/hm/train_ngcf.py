"""
src/training/hm/train_ngcf.py — NGCF Trainer for H&M
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import pickle
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.sparse.check_sparse_tensor_invariants(False)

from config import (
    PROCESSED_DIR, EMBEDDING_DIR, GRAPH_DIR, CHECKPOINT_DIR,
    DEVICE, SEED, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EMBEDDING_DIM, NUM_LAYERS, DROPOUT, THRESHOLD, NUM_WORKERS,
)
from src.models.model_factory import get_model
from src.evaluation.evaluator import Evaluator

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.sparse.check_sparse_tensor_invariants(False)

CSV_CHUNK         = 500_000
_TRAIN_BATCH_SIZE = BATCH_SIZE * 12
_NUM_WORKERS      = 16
_GRAD_ACCUM       = 1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HMTrainDataset(Dataset):
    def __init__(self, user_arr: np.ndarray, item_arr: np.ndarray,
                 n_items: int, adj_csr: sp.csr_matrix):
        self.users   = user_arr
        self.items   = item_arr
        self.n_items = n_items
        self.adj_csr = adj_csr
        self.negs    = np.random.randint(0, n_items, len(user_arr), dtype=np.int32)

    def resample(self):
        self.negs = np.random.randint(0, self.n_items, len(self.users), dtype=np.int32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(int(self.users[idx]), dtype=torch.long),
            torch.tensor(int(self.items[idx]), dtype=torch.long),
            torch.tensor(int(self.negs[idx]),  dtype=torch.long),
        )


class HMEvalDataset(Dataset):
    def __init__(self, user_arr: np.ndarray, item_arr: np.ndarray,
                 n_items: int, adj_csr: sp.csr_matrix, neg_ratio: int = 1):
        self.users    = user_arr
        self.items    = item_arr
        self.n_items  = n_items
        self.adj_csr  = adj_csr
        self._pos_len = len(user_arr)
        self._len     = self._pos_len * (1 + neg_ratio)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < self._pos_len:
            user  = int(self.users[idx])
            item  = int(self.items[idx])
            label = 1.0
        else:
            pos_idx = (idx - self._pos_len) % self._pos_len
            user    = int(self.users[pos_idx])
            bought  = set(self.adj_csr.getrow(user).indices)
            item    = random.randint(0, self.n_items - 1)
            while item in bought:
                item = random.randint(0, self.n_items - 1)
            label = 0.0
        return (
            torch.tensor(user,  dtype=torch.long),
            torch.tensor(item,  dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )

def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_score = (user_emb * pos_emb).sum(dim=-1)
    neg_score = (user_emb * neg_emb).sum(dim=-1)
    return -F.logsigmoid(pos_score - neg_score).mean()

class HMNGCFTrainer:

    MODEL_NAME = "ngcf"

    def __init__(self, feature: str):
        assert feature in ("clip", "fashionclip"), f"Invalid feature: {feature}"
        self.model_name = self.MODEL_NAME
        self.feature    = feature
        self.device     = DEVICE

        self.data_dir  = PROCESSED_DIR / "hm"
        self.emb_dir   = EMBEDDING_DIR / "hm" / feature
        self.graph_dir = GRAPH_DIR     / "hm" / feature
        self.ckpt_dir  = CHECKPOINT_DIR / "hm" / feature / self.model_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler("cuda")
        set_seed(SEED)

    def _load_meta(self) -> dict:
        with open(self.graph_dir / "graph_meta.json") as f:
            meta = json.load(f)
        return {"n_users": meta["n_users"], "n_items": meta["n_items"]}

    def _load_mappings(self) -> tuple[dict, dict]:
        user_map = self.graph_dir / "user2idx.csv"
        item_map = self.graph_dir / "item2idx.csv"

        if user_map.exists() and item_map.exists():
            print("[MAPPING] Loading from CSV...")
            u_df = pd.read_csv(user_map, dtype={"customer_id": str, "user_idx": int})
            i_df = pd.read_csv(item_map, dtype={"article_id":  str, "item_idx": int})
        else:
            print("[MAPPING] CSV not found → parsing graph_meta.json...")
            with open(self.graph_dir / "graph_meta.json") as f:
                meta = json.load(f)
            u_df = pd.DataFrame(list(meta["user2idx"].items()), columns=["customer_id", "user_idx"])
            i_df = pd.DataFrame(list(meta["item2idx"].items()), columns=["article_id",  "item_idx"])
            u_df.to_csv(user_map, index=False)
            i_df.to_csv(item_map, index=False)

        user2idx = dict(zip(u_df["customer_id"], u_df["user_idx"].astype(np.int32)))
        item2idx = dict(zip(i_df["article_id"],  i_df["item_idx"].astype(np.int32)))
        del u_df, i_df
        print(f"[MAPPING] users={len(user2idx):,}  items={len(item2idx):,}")
        return user2idx, item2idx

    def _load_split_as_arrays(self, split: str, user2idx: dict, item2idx: dict,
                               ) -> tuple[np.ndarray, np.ndarray]:
        path = self.data_dir / f"{split}.csv"
        u_bufs, i_bufs = [], []
        dropped = 0
        for chunk in pd.read_csv(path, usecols=["customer_id", "article_id"],
                                  dtype=str, chunksize=CSV_CHUNK):
            chunk["article_id"] = chunk["article_id"].str.zfill(10)
            u    = chunk["customer_id"].map(user2idx)
            i    = chunk["article_id"].map(item2idx)
            mask = u.notna() & i.notna()
            dropped += (~mask).sum()
            u_bufs.append(u[mask].values.astype(np.int32))
            i_bufs.append(i[mask].values.astype(np.int32))
        if dropped:
            print(f"  [MAP] {split}: dropped {dropped:,} rows")
        users = np.concatenate(u_bufs)
        items = np.concatenate(i_bufs)
        print(f"  [MAP] {split}: {len(users):,} interactions")
        return users, items

    def _load_adj_csr(self) -> sp.csr_matrix:
        adj_path = self.graph_dir / "adjacency_matrix.pkl"
        if not adj_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {adj_path}. Chạy build_graph.py trước.")
        with open(adj_path, "rb") as f:
            adj = pickle.load(f)
        mb = (adj.data.nbytes + adj.indices.nbytes + adj.indptr.nbytes) / 1e6
        print(f"[ADJ] CSR shape={adj.shape}  nnz={adj.nnz:,}  size≈{mb:.0f}MB")
        return adj

    def _load_embeddings(self, n_items: int, item2idx: dict) -> torch.Tensor:
        raw = np.load(self.emb_dir / "embeddings.npy")
        ids = (pd.read_csv(self.emb_dir / "article_ids.csv")
               ["article_id"].astype(str).str.zfill(10).tolist())
        dim  = raw.shape[1]
        feat = np.zeros((n_items, dim), dtype=np.float32)
        for i, aid in enumerate(ids):
            idx = item2idx.get(aid)
            if idx is not None:
                feat[idx] = raw[i]
        return torch.tensor(feat, dtype=torch.float32)

    def _build_model(self, n_users: int, n_items: int) -> torch.nn.Module:
        return get_model(
            self.model_name,
            num_users=n_users,
            num_items=n_items,
            embedding_dim=EMBEDDING_DIM,
        ).to(self.device)

    def _init_model_with_clip(self, model, item_feat: torch.Tensor):
        if hasattr(model, "init_item_embeddings_from_clip"):
            feat_dev = item_feat.to(self.device)
            model.init_item_embeddings_from_clip(feat_dev)
            del feat_dev
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"[WARN] Model {self.model_name} không có init_item_embeddings_from_clip.")

    def _train_epoch(self, model, loader, optimizer, edge_index, epoch: int = 0) -> float:
        model.train()
        model.invalidate_cache()
    
        total_loss  = 0.0
        accum_count = 0
        optimizer.zero_grad()
    
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", dynamic_ncols=True)
    
        for step, (user_ids, pos_ids, neg_ids) in enumerate(pbar):
            user_ids = user_ids.to(self.device, non_blocking=True)
            pos_ids  = pos_ids.to(self.device,  non_blocking=True)
            neg_ids  = neg_ids.to(self.device,  non_blocking=True)
    
            with autocast("cuda"):
                use_cache = (step % _GRAD_ACCUM != 0)
                if not use_cache:
                    model.invalidate_cache()
    
                all_user_emb, all_item_emb = model(edge_index, use_cache=use_cache)
    
                loss = bpr_loss(
                    all_user_emb[user_ids],
                    all_item_emb[pos_ids],
                    all_item_emb[neg_ids],
                ) / _GRAD_ACCUM
    
            self.scaler.scale(loss).backward()
    
            is_last = (((step + 1) % _GRAD_ACCUM == 0) or ((step + 1) == len(loader)))
            if is_last:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
    
            total_loss  += loss.item() * _GRAD_ACCUM
            accum_count += 1
            pbar.set_postfix(
                loss=f"{loss.item() * _GRAD_ACCUM:.4f}",
                avg=f"{total_loss / accum_count:.4f}",
            )
    
        return total_loss / accum_count

    @torch.no_grad()
    def _run_eval(self, model, eval_dataset, edge_index):
        model.eval()
        loader = DataLoader(
            eval_dataset,
            batch_size=_TRAIN_BATCH_SIZE * 4,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(_NUM_WORKERS > 0),
        )
    
        with autocast("cuda"):
            all_user_emb, all_item_emb = model(edge_index, use_cache=True)
    
        preds, labels = [], []
        for user_ids, item_ids, label in tqdm(loader, desc="Evaluating"):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            with autocast("cuda"):
                score = torch.sigmoid(
                    (all_user_emb[user_ids] * all_item_emb[item_ids]).sum(dim=-1)
                )
            preds.extend(score.cpu().numpy())
            labels.extend(label.numpy())
    
        return np.array(preds), np.array(labels)

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict:
        y_pred = (preds >= THRESHOLD).astype(int)
        y_true = labels.astype(int)
        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        }

    def train(self, evaluator: "Evaluator | None" = None) -> dict:
        print(f"\n{'='*60}")
        print(f" H&M NGCF | feature={self.feature}")
        print(f"{'='*60}\n")

        meta    = self._load_meta()
        n_users = meta["n_users"]
        n_items = meta["n_items"]
        print(f"[SETUP] n_users={n_users:,}  n_items={n_items:,}")

        user2idx, item2idx = self._load_mappings()

        item_feat  = self._load_embeddings(n_items, item2idx)
        edge_index = torch.load(self.graph_dir / "edge_index.pt", map_location="cpu")
        edge_index = edge_index.to(self.device)
        print(f"[SETUP] edge_index: {edge_index.shape}  ({self.device})")

        adj_csr = self._load_adj_csr()

        print("[SETUP] Loading splits...")
        train_users, train_items = self._load_split_as_arrays("train", user2idx, item2idx)
        val_users,   val_items   = self._load_split_as_arrays("val",   user2idx, item2idx)
        test_users,  test_items  = self._load_split_as_arrays("test",  user2idx, item2idx)
        del user2idx, item2idx

        print(f"[SETUP] train={len(train_users):,}  val={len(val_users):,}  test={len(test_users):,}")

        train_dataset = HMTrainDataset(train_users, train_items, n_items, adj_csr)
        val_dataset   = HMEvalDataset(val_users,   val_items,   n_items, adj_csr)
        test_dataset  = HMEvalDataset(test_users,  test_items,  n_items, adj_csr)

        train_loader = DataLoader(
            train_dataset,
            batch_size=_TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        print(f"[LOADER] batch={_TRAIN_BATCH_SIZE}  workers={_NUM_WORKERS}  "
              f"accum={_GRAD_ACCUM}  effective_batch={_TRAIN_BATCH_SIZE*_GRAD_ACCUM}  "
              f"batches/epoch={len(train_loader):,}")

        model = self._build_model(n_users, n_items)
        self._init_model_with_clip(model, item_feat)
        del item_feat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[SETUP] CLIP embedding injected ✓")

        n_nodes = n_users + n_items
        model.precompute_norm_adj(edge_index, n_nodes)
        print("[SETUP] norm_adj precomputed ✓")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] NGCF | params={n_params:,}\n")

        ckpt_name = f"{self.feature}_{self.model_name}.pth"
        ckpt_path = self.ckpt_dir / ckpt_name

        start_epoch = 1
        best_f1     = 0.0
        best_epoch  = 0

        if ckpt_path.exists():
            print(f"[RESUME] Load từ {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_f1     = ckpt["best_f1"]
            best_epoch  = ckpt["epoch"]
            print(f"[RESUME] Tiếp tục epoch {start_epoch}, best F1={best_f1:.4f}\n")
        else:
            print("[RESUME] Không có checkpoint → train mới.\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            train_dataset.resample()
            train_loss = self._train_epoch(model, train_loader, optimizer, edge_index, epoch)

            val_preds, val_labels = self._run_eval(model, val_dataset, edge_index)
            val_metrics = self._compute_metrics(val_preds, val_labels)
            scheduler.step(val_metrics["f1"])

            print(
                f"Epoch [{epoch:>3}/{NUM_EPOCHS}] loss={train_loss:.4f} | "
                f"acc={val_metrics['accuracy']:.4f} "
                f"prec={val_metrics['precision']:.4f} "
                f"rec={val_metrics['recall']:.4f} "
                f"f1={val_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > best_f1:
                best_f1    = val_metrics["f1"]
                best_epoch = epoch
                torch.save({
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch":     epoch,
                    "best_f1":   best_f1,
                }, ckpt_path)
                print(f"  ✓ Saved {ckpt_name} (f1={best_f1:.4f}, epoch={epoch})\n")

        print(f"\n[TRAIN DONE] Best val F1={best_f1:.4f} at epoch {best_epoch}")

        print("\n[TEST] Loading best checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        test_preds, test_labels = self._run_eval(model, test_dataset, edge_index)
        test_metrics = self._compute_metrics(test_preds, test_labels)

        print(
            f"[TEST] acc={test_metrics['accuracy']:.4f} "
            f"prec={test_metrics['precision']:.4f} "
            f"rec={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f}"
        )

        if evaluator is not None:
            evaluator.evaluate(
                dataset="hm",
                feature=self.feature,
                model_name=self.model_name,
                preds=test_preds,
                targets=test_labels,
            )

        return test_metrics