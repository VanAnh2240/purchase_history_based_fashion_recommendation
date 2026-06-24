"""
src/evaluation/runner.py
"""

import gc
import json
import pickle
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

from config import (
    PROCESSED_DIR, EMBEDDING_DIR, GRAPH_DIR, CHECKPOINT_DIR,
    DEVICE, SEED, BATCH_SIZE, EMBEDDING_DIM, NUM_WORKERS,
    LOGIT_SCALE,
)
from src.models.model_factory import get_model
from src.models.siamese import SiameseProjector
from src.evaluation.evaluator import Evaluator

CSV_CHUNK    = 500_000
_EVAL_BATCH  = BATCH_SIZE * 4
_NUM_WORKERS = NUM_WORKERS
_LOGIT_SCALE = LOGIT_SCALE

ALL_GNN_MODELS = ["lightgcn", "graphsage", "ngcf"]

_cache: dict = {}

def _ck(*args):
    return args

def clear_cache():
    _cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[WARN] clear_cache: cuda.empty_cache() failed ({e})")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EvalDataset(Dataset):
    def __init__(self, user_arr, item_arr, n_items, adj_csr, neg_ratio=1):
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
            user, item, label = int(self.users[idx]), int(self.items[idx]), 1.0
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


class BPREvalDatasetSimple(Dataset):
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, n_items: int):
        n = len(user_ids)
        neg_items = np.random.randint(0, n_items, n, dtype=np.int32)

        all_u = np.concatenate([user_ids, user_ids])
        all_i = np.concatenate([item_ids, neg_items])
        all_l = np.concatenate([np.ones(n, np.float32), np.zeros(n, np.float32)])

        self.users  = torch.from_numpy(all_u.astype(np.int32))
        self.items  = torch.from_numpy(all_i.astype(np.int32))
        self.labels = torch.from_numpy(all_l)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

class SiameseEvalDataset(Dataset):
    def __init__(
        self,
        test_users,
        test_items,
        user_hist_emb,
        item_feat,
        adj_csr,
        neg_ratio=1
    ):
        self.users = test_users
        self.items = test_items
        self.user_hist_emb = user_hist_emb
        self.item_feat = item_feat
        self.adj_csr = adj_csr
        self.n_items = item_feat.shape[0]

        self._pos_len = len(test_users)
        self._len = self._pos_len * (1 + neg_ratio)

        print(f"[SiameseEvalDataset] pairs: {self._pos_len:,}")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        if idx < self._pos_len:
            u = int(self.users[idx])
            i = int(self.items[idx])
            label = 1.0

        else:
            pos_idx = (idx - self._pos_len) % self._pos_len
            u = int(self.users[pos_idx])

            bought = set(self.adj_csr.getrow(u).indices)

            i = random.randint(0, self.n_items - 1)
            while i in bought:
                i = random.randint(0, self.n_items - 1)

            label = 0.0

        user_vec = self.user_hist_emb[u]
        item_vec = self.item_feat[i]

        return user_vec, item_vec, np.float32(label)

class ModelEvaluator:

    def __init__(self, dataset, feature, model_name):
        self.dataset    = dataset
        self.feature    = feature
        self.model_name = model_name
        self.device     = DEVICE
        self.data_dir   = PROCESSED_DIR / dataset

        if feature:
            self.graph_dir = GRAPH_DIR      / dataset / feature
            self.emb_dir   = EMBEDDING_DIR  / dataset / feature
            self.ckpt_dir  = CHECKPOINT_DIR / dataset / feature / model_name
        else:
            self.graph_dir = GRAPH_DIR / dataset
            self.emb_dir   = None
            self.ckpt_dir  = CHECKPOINT_DIR / dataset / model_name

        set_seed(SEED)

    # ── loaders với cache ─────────────────────────────────────────────────

    def _load_meta(self):
        k = _ck("meta", self.dataset, self.feature)
        if k not in _cache:
            with open(self.graph_dir / "graph_meta.json") as f:
                _cache[k] = json.load(f)
            n = _cache[k]
            print(f"[META] n_users={n['n_users']:,}  n_items={n['n_items']:,}")
        return _cache[k]

    def _load_mappings(self, meta):
        k = _ck("mappings", self.dataset, self.feature)
        if k not in _cache:
            user_map = self.graph_dir / "user2idx.csv"
            item_map = self.graph_dir / "item2idx.csv"
            if user_map.exists() and item_map.exists():
                u_df = pd.read_csv(user_map, dtype={"customer_id": str, "user_idx": int})
                i_df = pd.read_csv(item_map, dtype={"article_id":  str, "item_idx": int})
            else:
                u_df = pd.DataFrame(list(meta["user2idx"].items()), columns=["customer_id", "user_idx"])
                i_df = pd.DataFrame(list(meta["item2idx"].items()), columns=["article_id",  "item_idx"])
            user2idx = dict(zip(u_df["customer_id"], u_df["user_idx"].astype(np.int32)))
            item2idx = dict(zip(i_df["article_id"],  i_df["item_idx"].astype(np.int32)))
            _cache[k] = (user2idx, item2idx)
        return _cache[k]

    def _load_split_arrays(self, split, user2idx, item2idx, with_time=False):
        k = _ck("split", self.dataset, self.feature, split, with_time)
        if k not in _cache:
            path = self.data_dir / f"{split}.csv"
            cols = ["t_dat", "customer_id", "article_id"] if with_time else ["customer_id", "article_id"]
            u_bufs, i_bufs, t_bufs = [], [], []
            dropped = 0
            for chunk in pd.read_csv(path, usecols=cols, dtype=str, chunksize=CSV_CHUNK):
                chunk["article_id"] = chunk["article_id"].str.zfill(10)
                u    = chunk["customer_id"].map(user2idx)
                i    = chunk["article_id"].map(item2idx)
                mask = u.notna() & i.notna()
                dropped += (~mask).sum()
                u_bufs.append(u[mask].values.astype(np.int32))
                i_bufs.append(i[mask].values.astype(np.int32))
                if with_time:
                    t = pd.to_datetime(chunk[mask]["t_dat"]).astype(np.int64).values // 10**9
                    t_bufs.append(t.astype(np.int64))
            if dropped:
                print(f"  [MAP] {split}: dropped {dropped:,} rows")
            users = np.concatenate(u_bufs)
            items = np.concatenate(i_bufs)
            _cache[k] = (users, items, np.concatenate(t_bufs)) if with_time else (users, items)
        result = _cache[k]
        return result if with_time else result[:2]

    def _load_adj(self):
        k = _ck("adj", self.dataset, self.feature)
        if k not in _cache:
            with open(self.graph_dir / "adjacency_matrix.pkl", "rb") as f:
                _cache[k] = pickle.load(f)
        return _cache[k]

    def _load_edge_index(self):
        k = _ck("edge_index", self.dataset, self.feature)
        if k not in _cache:
            _cache[k] = torch.load(self.graph_dir / "edge_index.pt",
                                   map_location=self.device, weights_only=False)
        return _cache[k]

    def _load_item_feat_numpy(self, n_items, item2idx):
        k = _ck("item_feat", self.dataset, self.feature)
        if k not in _cache:
            raw  = np.load(self.emb_dir / "embeddings.npy")
            ids  = (pd.read_csv(self.emb_dir / "article_ids.csv")
                    ["article_id"].astype(str).str.zfill(10).tolist())
            feat = np.zeros((n_items, raw.shape[1]), dtype=np.float32)
            for i, aid in enumerate(ids):
                idx = item2idx.get(aid)
                if idx is not None:
                    feat[idx] = raw[i]
            norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-10
            _cache[k] = feat / norms
        return _cache[k]

    def _load_item_feat_tensor(self, n_items, item2idx):
        return torch.tensor(self._load_item_feat_numpy(n_items, item2idx), dtype=torch.float32)

    # ── checkpoint ────────────────────────────────────────────────────────

    def _load_checkpoint(self, model):
        ckpt_name = (
            "bpr.pth"
            if self.model_name == "bpr"
            else f"{self.feature}_{self.model_name}.pth"
        )
    
        ckpt_path = self.ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {ckpt_path}\n"
                "Hãy chạy train.py trước."
            )
    
        ckpt = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=False
        )
    
        state_dict = ckpt["model"]
    
        if self.model_name == "siamese":
            state_dict = {
                k.replace("projector.", "", 1): v
                for k, v in state_dict.items()
            }
    
        model.load_state_dict(state_dict)
    
        print(
            f"[CKPT] {ckpt_path.name}  "
            f"epoch={ckpt.get('epoch', '?')}  "
            f"best_val_f1={ckpt.get('best_f1', 0):.4f}"
        )
        return model

    # ── inference ─────────────────────────────────────────────────────────

    def _make_loader(self, dataset, num_workers=None):
        nw = _NUM_WORKERS if num_workers is None else num_workers
        return DataLoader(
            dataset, batch_size=_EVAL_BATCH, shuffle=False,
            num_workers=nw,
            pin_memory=torch.cuda.is_available() and nw == 0,
            persistent_workers=(nw > 0),
        )

    def _normalize_scores(self, preds: np.ndarray, label: str = "") -> np.ndarray:
        """Min-max normalize scores về [0, 1] để threshold có ý nghĩa."""
        score_min, score_max = preds.min(), preds.max()
        if score_max - score_min > 1e-8:
            return (preds - score_min) / (score_max - score_min)
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}[WARN] All scores identical — model may not be learning.")
        return preds

    @torch.no_grad()
    def _run_gnn_inference(self, model, dataset, edge_index):
        model.eval()
        all_user_emb, all_item_emb = model(edge_index, use_cache=False)
        preds, labels = [], []

        for user_ids, item_ids, label in self._make_loader(dataset):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            with autocast("cuda"):
                # FIX: raw dot-product, không sigmoid
                # BPR score không calibrate cho sigmoid threshold tuyệt đối
                score = (all_user_emb[user_ids] * all_item_emb[item_ids]).sum(dim=-1)
            preds.extend(score.cpu().float().numpy())
            labels.extend(label.numpy())

        preds  = np.array(preds, dtype=np.float64)
        labels = np.array(labels)
        # FIX: normalize để Youden threshold trong metrics.py hoạt động đúng
        preds  = self._normalize_scores(preds, label=self.model_name)
        return preds, labels

    @torch.no_grad()
    def _run_bpr_inference(self, model, dataset):
        model.eval()
        preds, labels = [], []

        for user_ids, item_ids, label in self._make_loader(dataset):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            # FIX: raw dot-product, không sigmoid — nhất quán với BPR loss
            score = model.predict(user_ids, item_ids)
            preds.extend(score.cpu().float().numpy())
            labels.extend(label.numpy())

        preds  = np.array(preds, dtype=np.float64)
        labels = np.array(labels)
        # FIX: normalize để Youden threshold trong metrics.py hoạt động đúng
        preds  = self._normalize_scores(preds, label="bpr")
        return preds, labels

    @torch.no_grad()
    def _run_siamese_inference(self, model, dataset):
        model.eval()
        loader = DataLoader(
            dataset, 
            batch_size=4096,                
            shuffle=False, 
            num_workers=0, 
            pin_memory=False)
        preds, labels = [], []
        for ea, eb, label in loader:
            ea = ea.to(self.device, non_blocking=True)
            eb = eb.to(self.device, non_blocking=True)
            with autocast("cuda"):
                logit = (model(ea) * model(eb)).sum(-1) * _LOGIT_SCALE
                prob  = torch.sigmoid(logit)
            preds.extend(prob.cpu().numpy())
            labels.extend(label.numpy())
        return np.array(preds), np.array(labels)

    # ── main ──────────────────────────────────────────────────────────────

    def run(self):
        meta    = self._load_meta()
        n_users = meta["n_users"]
        n_items = meta["n_items"]

        user2idx, item2idx = self._load_mappings(meta)

        if self.model_name in ALL_GNN_MODELS:
            adj_csr = self._load_adj()
        
            test_users, test_items = self._load_split_arrays(
                "test",
                user2idx,
                item2idx,
                with_time=False
            )
            print(f"[TEST] interactions={len(test_users):,}")
        
            test_dataset = EvalDataset(
                test_users,
                test_items,
                n_items,
                adj_csr
            )
        
            model = get_model(
                self.model_name,
                num_users=n_users,
                num_items=n_items,
                embedding_dim=EMBEDDING_DIM,
            ).to(self.device)
        
            model = self._load_checkpoint(model)
        
            edge_index = self._load_edge_index()
        
            if self.model_name in ("lightgcn", "ngcf"):
                model.precompute_norm_adj(
                    edge_index,
                    n_users + n_items
                )
        
            return self._run_gnn_inference(
                model,
                test_dataset,
                edge_index
            )

        elif self.model_name == "bpr":
            _cache.clear()
            gc.collect()
        
            ckpt_path = self.ckpt_dir / "bpr.pth"
            ckpt_tmp  = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False
            )
            ckpt_n_users = ckpt_tmp.get("n_users")
            ckpt_n_items = ckpt_tmp.get("n_items")
            del ckpt_tmp
        
            if ckpt_n_users is None or ckpt_n_items is None:
                raise ValueError(
                    "[BPR] Checkpoint không có n_users/n_items. "
                    "Hãy retrain với train_bpr.py."
                )
        
            print(
                f"[BPR] ckpt size: "
                f"n_users={ckpt_n_users:,}  "
                f"n_items={ckpt_n_items:,}"
            )
        
            # giữ log cũ
            print("[BPR] Building index map from train.csv...")
            print(
                f"[BPR] index map: "
                f"users={ckpt_n_users:,}  "
                f"items={ckpt_n_items:,}"
            )
        
            # dùng graph mapping chung như GNN
            test_users, test_items = self._load_split_arrays(
                "test",
                user2idx,
                item2idx,
                with_time=False
            )
        
            # chỉ giữ sample nằm trong phạm vi BPR checkpoint
            mask = (
                (test_users < ckpt_n_users) &
                (test_items < ckpt_n_items)
            )
        
            dropped = (~mask).sum()
        
            if dropped:
                print(f"  [MAP] test (bpr): dropped {dropped:,} rows")
        
            bpr_test_users = test_users[mask]
            bpr_test_items = test_items[mask]
        
            print(f"[TEST] interactions={len(bpr_test_users):,}")
        
            test_dataset = BPREvalDatasetSimple(
                bpr_test_users,
                bpr_test_items,
                ckpt_n_items
            )
        
            del bpr_test_users, bpr_test_items
            gc.collect()
        
            model = get_model(
                "bpr",
                num_users=ckpt_n_users,
                num_items=ckpt_n_items,
                embedding_dim=EMBEDDING_DIM,
            ).to(self.device)
        
            model = self._load_checkpoint(model)
        
            return self._run_bpr_inference(
                model,
                test_dataset
            )

        elif self.model_name == "siamese":
            item_feat = self._load_item_feat_numpy(n_items, item2idx)
            dim_in    = item_feat.shape[1]

            adj_csr = self._load_adj()

            test_users, test_items = self._load_split_arrays(
                "test",
                user2idx,
                item2idx,
                with_time=False
            )
            
            print(f"[TEST] interactions={len(test_users):,}")
            
            user_hist_emb = np.load(
                self.emb_dir / "user_hist_emb.npy"
            )
            
            test_dataset = SiameseEvalDataset(
                test_users=test_users,
                test_items=test_items,
                user_hist_emb=user_hist_emb,
                item_feat=item_feat,
                adj_csr=adj_csr
            )

            model = SiameseProjector(dim_in=dim_in).to(self.device)
            model = self._load_checkpoint(model)
            return self._run_siamese_inference(model, test_dataset)

        raise ValueError(f"Model không hỗ trợ: {self.model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_ALL_MODELS = [
    ("lightgcn", "clip"), ("lightgcn", "fashionclip"),
    ("graphsage", "clip"), ("graphsage", "fashionclip"),
    ("ngcf",      "clip"), ("ngcf",      "fashionclip"),
    ("bpr",       None),
    ("siamese",   "clip"), ("siamese",   "fashionclip"),
]


def run_evaluation(dataset, feature, model_name):
    if dataset == "polyvore":
        raise NotImplementedError("Polyvore evaluation chưa implement.")

    # ── Evaluate ALL ──────────────────────────────────────────────────────
    if model_name is None:
        print(f"\n{'='*60}")
        print(f" EVALUATE ALL | dataset={dataset}")
        print(f"{'='*60}\n")
        set_seed(SEED)
        evaluator = Evaluator()

        candidates = []
        for m_name, m_feat in _ALL_MODELS:
            if feature is not None and m_feat != feature:
                continue
            if m_feat:
                ckpt_dir  = CHECKPOINT_DIR / dataset / m_feat / m_name
                ckpt_name = f"{m_feat}_{m_name}.pth"
            else:
                ckpt_dir  = CHECKPOINT_DIR / dataset / m_name
                ckpt_name = f"{m_name}.pth"
            if (ckpt_dir / ckpt_name).exists():
                candidates.append((m_name, m_feat))

        if not candidates:
            print("[WARN] Không tìm thấy checkpoint nào.")
            return {}

        print(f"[SCAN] Tìm thấy {len(candidates)} checkpoint(s):")
        for m_name, m_feat in candidates:
            ckpt_name = f"{m_feat}_{m_name}.pth" if m_feat else f"{m_name}.pth"
            print(f"  • {ckpt_name}")
        print()

        results = {}
        for m_name, m_feat in candidates:
            try:
                runner = ModelEvaluator(dataset=dataset, feature=m_feat, model_name=m_name)
                preds, labels = runner.run()
                metrics = evaluator.evaluate(
                    dataset=dataset, feature=m_feat, model_name=m_name,
                    preds=preds, targets=labels,
                )
                results[f"{m_name}/{m_feat}"] = metrics
            except Exception as e:
                print(f"[ERROR] {m_name}/{m_feat}: {e}")

        evaluator.save_report(dataset)
        clear_cache()
        return results

    # ── Evaluate 1 model cụ thể ───────────────────────────────────────────
    if model_name in ("lightgcn", "graphsage", "ngcf", "siamese") and feature is None:
        raise ValueError(f"--feature là bắt buộc với model={model_name}")

    print(f"\n{'='*60}")
    print(f" EVALUATE | dataset={dataset} | feature={feature} | model={model_name}")
    print(f"{'='*60}\n")
    set_seed(SEED)

    runner = ModelEvaluator(dataset=dataset, feature=feature, model_name=model_name)
    preds, labels = runner.run()

    evaluator = Evaluator()
    metrics   = evaluator.evaluate(
        dataset=dataset, feature=feature, model_name=model_name,
        preds=preds, targets=labels,
    )
    evaluator.save_report(dataset)
    clear_cache()
    return metrics