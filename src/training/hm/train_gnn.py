"""
src/training/hm/train_gnn_hm.py
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
torch.set_float32_matmul_precision("high")

from config import (
    PROCESSED_DIR, EMBEDDING_DIR, GRAPH_DIR, CHECKPOINT_DIR,
    DEVICE, SEED, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EMBEDDING_DIM, NUM_LAYERS, DROPOUT, THRESHOLD, NUM_WORKERS
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
_L2_REG           = 1e-4   # FIX: L2 regularization weight cho BPR loss


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class HMTrainDataset(Dataset):
    def __init__(self, user_arr: np.ndarray, item_arr: np.ndarray,
                 n_items: int, adj_csr: sp.csr_matrix):
        self.users   = user_arr
        self.items   = item_arr
        self.n_items = n_items
        self.adj_csr = adj_csr
        self.negs    = np.random.randint(0, n_items, len(user_arr), dtype=np.int32)

    # FIX: hỗ trợ popularity-based hard negative sampling
    def resample(self, item_popularity: np.ndarray = None):
        """
        Resample negatives mỗi epoch.
        - item_popularity=None  → uniform random (dễ)
        - item_popularity=array → sample theo popularity^0.75 (harder negatives)
        """
        if item_popularity is not None:
            probs = item_popularity / item_popularity.sum()
            self.negs = np.random.choice(
                self.n_items, len(self.users), p=probs
            ).astype(np.int32)
        else:
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


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

# FIX: thêm L2 regularization vào BPR loss
# Không có L2 → embedding norm tăng vô hạn → score collapse → model predict all-same
def bpr_loss(user_emb, pos_emb, neg_emb, l2_reg: float = _L2_REG):
    pos_score = (user_emb * pos_emb).sum(dim=-1)
    neg_score = (user_emb * neg_emb).sum(dim=-1)
    bpr = -F.logsigmoid(pos_score - neg_score).mean()
    l2  = (
        user_emb.norm(dim=-1).pow(2).mean()
        + pos_emb.norm(dim=-1).pow(2).mean()
        + neg_emb.norm(dim=-1).pow(2).mean()
    )
    return bpr + l2_reg * l2


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class HMGNNTrainer:

    def __init__(self, model_name: str, feature: str):
        assert model_name in ("lightgcn", "graphsage", "ngcf")
        assert feature in ("clip", "fashionclip")

        self.model_name = model_name
        self.feature    = feature
        self.device     = DEVICE

        self.data_dir  = PROCESSED_DIR / "hm"
        self.emb_dir   = EMBEDDING_DIR / "hm" / feature
        self.graph_dir = GRAPH_DIR     / "hm" / feature
        self.ckpt_dir  = CHECKPOINT_DIR / "hm" / feature / model_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler("cuda")

        set_seed(SEED)

    # ── helpers ───────────────────────────────────────────────────────────

    def _load_meta(self):
        with open(self.graph_dir / "graph_meta.json") as f:
            meta = json.load(f)
        return {"n_users": meta["n_users"], "n_items": meta["n_items"]}

    def _load_mappings(self):
        user_map_path = self.graph_dir / "user2idx.csv"
        item_map_path = self.graph_dir / "item2idx.csv"

        if user_map_path.exists() and item_map_path.exists():
            print("[MAPPING] Loading from CSV...")
            user_df = pd.read_csv(user_map_path, dtype={"customer_id": str, "user_idx": int})
            item_df = pd.read_csv(item_map_path, dtype={"article_id": str,  "item_idx": int})
        else:
            print("[MAPPING] CSV not found → parsing graph_meta.json...")
            with open(self.graph_dir / "graph_meta.json") as f:
                meta = json.load(f)
            user_df = pd.DataFrame(list(meta["user2idx"].items()), columns=["customer_id", "user_idx"])
            item_df = pd.DataFrame(list(meta["item2idx"].items()), columns=["article_id",  "item_idx"])
            user_df.to_csv(user_map_path, index=False)
            item_df.to_csv(item_map_path, index=False)

        user2idx_dict = dict(zip(user_df["customer_id"], user_df["user_idx"].astype(np.int32)))
        item2idx_dict = dict(zip(item_df["article_id"],  item_df["item_idx"].astype(np.int32)))
        del user_df, item_df

        print(f"[MAPPING] users={len(user2idx_dict):,}  items={len(item2idx_dict):,}")
        return user2idx_dict, item2idx_dict

    def _load_split_as_arrays(self, split: str, user2idx: dict, item2idx: dict
                               ) -> tuple[np.ndarray, np.ndarray]:
        path = self.data_dir / f"{split}.csv"
        user_bufs, item_bufs = [], []
        dropped = 0

        for chunk in pd.read_csv(path, usecols=["customer_id", "article_id"],
                                  dtype=str, chunksize=CSV_CHUNK):
            chunk["article_id"] = chunk["article_id"].str.zfill(10)
            u    = chunk["customer_id"].map(user2idx)
            i    = chunk["article_id"].map(item2idx)
            mask = u.notna() & i.notna()
            dropped += (~mask).sum()
            user_bufs.append(u[mask].values.astype(np.int32))
            item_bufs.append(i[mask].values.astype(np.int32))

        if dropped:
            print(f"  [MAP] {split}: dropped {dropped:,} rows")

        users = np.concatenate(user_bufs)
        items = np.concatenate(item_bufs)
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
        feat = np.zeros((len(item2idx), dim), dtype=np.float32)

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
            item_feat_dev = item_feat.to(self.device)
            model.init_item_embeddings_from_clip(item_feat_dev)
            del item_feat_dev
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"[WARN] Model {self.model_name} chưa có init_item_embeddings_from_clip, bỏ qua.")

    # ── train one epoch ───────────────────────────────────────────────────

    def _train_epoch(self, model, loader, optimizer, edge_index, epoch=0):
        model.train()
        model.invalidate_cache()

        total_loss  = 0.0
        accum_count = 0
        optimizer.zero_grad()

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]",
            dynamic_ncols=True,
        )

        for step, (user_ids, pos_ids, neg_ids) in enumerate(pbar):

            user_ids = user_ids.to(self.device, non_blocking=True)
            pos_ids  = pos_ids.to(self.device, non_blocking=True)
            neg_ids  = neg_ids.to(self.device, non_blocking=True)

            use_cache = (step % _GRAD_ACCUM != 0)

            if not use_cache:
                model.invalidate_cache()

            all_user_emb, all_item_emb = model(edge_index, use_cache=use_cache)

            with autocast("cuda"):
                # FIX: bpr_loss giờ có L2 regularization
                loss = bpr_loss(
                    all_user_emb[user_ids],
                    all_item_emb[pos_ids],
                    all_item_emb[neg_ids],
                ) / _GRAD_ACCUM

            is_last_accum = (
                ((step + 1) % _GRAD_ACCUM == 0)
                or ((step + 1) == len(loader))
            )

            self.scaler.scale(loss).backward()

            if is_last_accum:
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

    # ── evaluate ──────────────────────────────────────────────────────────

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

        # FIX: use_cache=False để tránh dùng cache stale từ train
        all_user_emb, all_item_emb = model(edge_index, use_cache=False)

        preds, labels = [], []

        for user_ids, item_ids, label in tqdm(loader, desc="Evaluating"):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)

            with autocast("cuda"):
                # FIX: raw dot-product, không sigmoid
                # BPR score không calibrate cho sigmoid threshold tuyệt đối
                score = (all_user_emb[user_ids] * all_item_emb[item_ids]).sum(dim=-1)

            preds.extend(score.cpu().float().numpy())
            labels.extend(label.numpy())

        preds  = np.array(preds)
        labels = np.array(labels)

        # FIX: min-max normalize để threshold có ý nghĩa
        score_min, score_max = preds.min(), preds.max()
        if score_max - score_min > 1e-8:
            preds = (preds - score_min) / (score_max - score_min)
        else:
            print("[WARN] All eval scores are identical — model may not be learning.")

        return preds, labels

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict:
        y_true = labels.astype(int)

        try:
            auc = roc_auc_score(y_true, preds)
        except ValueError:
            auc = 0.5

        # FIX: dùng Youden's J (tpr - fpr) để tìm threshold tối ưu từ ROC curve
        # Median threshold sai vì với neg_ratio=1 (50/50 dataset), median luôn
        # chia đôi → acc=prec=rec=f1 bất kể model học tốt hay không
        try:
            fpr, tpr, thresholds = roc_curve(y_true, preds)
            best_idx      = int(np.argmax(tpr - fpr))
            best_threshold = float(thresholds[best_idx])
        except ValueError:
            best_threshold = 0.5

        y_pred = (preds >= best_threshold).astype(int)

        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "auc":       auc,
            "threshold": best_threshold,
        }

    # ── main entry ────────────────────────────────────────────────────────

    def train(self, evaluator: "Evaluator | None" = None):
        print(f"\n{'='*60}")
        print(f" H&M GNN | model={self.model_name} | feature={self.feature}")
        print(f"{'='*60}\n")

        meta    = self._load_meta()
        n_users = meta["n_users"]
        n_items = meta["n_items"]
        print(f"[SETUP] n_users={n_users:,}  n_items={n_items:,}")

        user2idx, item2idx = self._load_mappings()

        item_feat = self._load_embeddings(n_items, item2idx)
        print(f"[SETUP] item_feat: {item_feat.shape}")

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

        # FIX: tính item popularity cho hard negative sampling
        item_counts     = np.bincount(train_items, minlength=n_items).astype(np.float32)
        item_popularity = item_counts ** 0.75   # smoothed (word2vec style)
        print(f"[SETUP] item popularity computed (non-zero items: {(item_counts > 0).sum():,})")

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

        # FIX: chỉ gọi 1 lần (bản cũ bị duplicate)
        self._init_model_with_clip(model, item_feat)
        del item_feat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[SETUP] CLIP embedding injected ✓")

        if self.model_name == "lightgcn":
            n = n_users + n_items
            model.precompute_norm_adj(edge_index, n)
            print("[SETUP] norm_adj precomputed")

        # FIX: differential LR — bảo vệ CLIP-initialized embeddings
        # Embedding đã có semantic meaning từ CLIP → học chậm hơn (×0.1)
        # Linear layer chưa có prior → học nhanh bình thường
        emb_params   = [p for n, p in model.named_parameters() if "embedding" in n]
        other_params = [p for n, p in model.named_parameters() if "embedding" not in n]
        optimizer = torch.optim.Adam([
            {"params": emb_params,   "lr": LEARNING_RATE * 0.1},
            {"params": other_params, "lr": LEARNING_RATE},
        ], lr=LEARNING_RATE)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )

        print(f"[MODEL] {self.model_name.upper()} | "
              f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

        ckpt_name = f"{self.feature}_{self.model_name}.pth"
        ckpt_path = self.ckpt_dir / ckpt_name

        start_epoch      = 1
        best_auc         = 0.0
        best_epoch       = 0
        epochs_no_improve = 0      # FIX: early stopping counter
        _EARLY_STOP_PATIENCE = 5

        if ckpt_path.exists():
            print(f"[RESUME] Load từ {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_auc    = ckpt.get("best_auc", ckpt.get("best_f1", 0.0))
            best_epoch  = ckpt["epoch"]
            print(f"[RESUME] Tiếp tục epoch {start_epoch}, best AUC={best_auc:.4f}\n")
        else:
            print("[RESUME] Không có checkpoint → train mới.\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            # FIX: delay hard negatives đến epoch 6
            # Epoch 1-5: uniform sampling để model ổn định với CLIP init
            # Epoch 6+: popularity-based hard negatives
            if epoch <= 5:
                train_dataset.resample(item_popularity=None)
            else:
                train_dataset.resample(item_popularity=item_popularity)

            train_loss = self._train_epoch(model, train_loader, optimizer, edge_index, epoch)

            val_preds, val_labels = self._run_eval(model, val_dataset, edge_index)
            val_metrics = self._compute_metrics(val_preds, val_labels)

            # FIX: scheduler theo AUC
            scheduler.step(val_metrics["auc"])

            print(
                f"Epoch [{epoch:>3}/{NUM_EPOCHS}] loss={train_loss:.4f} | "
                f"acc={val_metrics['accuracy']:.4f} "
                f"prec={val_metrics['precision']:.4f} "
                f"rec={val_metrics['recall']:.4f} "
                f"f1={val_metrics['f1']:.4f} "
                f"auc={val_metrics['auc']:.4f} "
                f"thr={val_metrics['threshold']:.4f}"
            )

            # FIX: save checkpoint theo AUC thay vì F1
            if val_metrics["auc"] > best_auc:
                best_auc          = val_metrics["auc"]
                best_epoch        = epoch
                epochs_no_improve = 0
                torch.save({
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch":     epoch,
                    "best_auc":  best_auc,
                    "best_f1":   val_metrics["f1"],
                }, ckpt_path)
                print(f"  ✓ Saved {ckpt_name} (auc={best_auc:.4f}, f1={val_metrics['f1']:.4f}, epoch={epoch})\n")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= _EARLY_STOP_PATIENCE:
                    print(f"\n[EARLY STOP] No improvement for {_EARLY_STOP_PATIENCE} epochs. Stopping at epoch {epoch}.")
                    break

        print(f"\n[TRAIN DONE] Best val AUC={best_auc:.4f} at epoch {best_epoch}")

        print("\n[TEST] Loading best checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        test_preds, test_labels = self._run_eval(model, test_dataset, edge_index)
        test_metrics = self._compute_metrics(test_preds, test_labels)

        print(
            f"[TEST] acc={test_metrics['accuracy']:.4f} "
            f"prec={test_metrics['precision']:.4f} "
            f"rec={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f} "
            f"auc={test_metrics['auc']:.4f} "
            f"thr={test_metrics['threshold']:.4f}"
        )

        if evaluator is not None:
            evaluator.evaluate(
                dataset="hm",
                feature=self.feature,
                model_name=self.model_name,
                all_targets=test_labels,
            )

        return test_metrics