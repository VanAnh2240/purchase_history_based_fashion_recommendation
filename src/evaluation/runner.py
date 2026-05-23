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

# ─── Module-level cache ────────────────────────────────────────────────────
_cache: dict = {}

def _ck(*args):
    return args

def clear_cache():
    _cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Datasets (không đổi)
# ─────────────────────────────────────────────────────────────────────────────

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


class SiameseEvalDataset(Dataset):
    def __init__(self, test_users, test_items, test_times,
                 train_users, train_items, train_times,
                 item_feat, item_counts, hard_neg_k=5_000, neg_ratio=1):

        self.item_feat = item_feat
        self.neg_ratio = neg_ratio
        n_items = item_feat.shape[0]

        top_k = min(hard_neg_k, n_items)
        self.hard_neg_pool = np.argsort(item_counts)[::-1][:top_k].astype(np.int32)

        tr_df = pd.DataFrame({"u": train_users, "i": train_items, "t": train_times})
        last_train = (tr_df.sort_values("t")
                           .groupby("u")["i"].last()
                           .reset_index().rename(columns={"i": "anchor"}))
        del tr_df; gc.collect()

        te_df = pd.DataFrame({"u": test_users, "i": test_items, "t": test_times})
        first_test = (te_df.sort_values("t")
                           .groupby("u")["i"].first()
                           .reset_index().rename(columns={"i": "positive"}))
        del te_df; gc.collect()

        merged = last_train.merge(first_test, on="u", how="inner")
        del last_train, first_test; gc.collect()

        n = len(merged)
        neg_idx        = np.random.randint(0, len(self.hard_neg_pool), n)
        self.anchors   = merged["anchor"].values.astype(np.int32)
        self.positives = merged["positive"].values.astype(np.int32)
        self.negatives = self.hard_neg_pool[neg_idx].astype(np.int32)
        self._pos_len  = n
        print(f"[SiameseEvalDataset] pairs: {n:,}")

    def __len__(self):
        return self._pos_len * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        if idx < self._pos_len:
            a, b, label = int(self.anchors[idx]), int(self.positives[idx]), 1.0
        else:
            p = (idx - self._pos_len) % self._pos_len
            a, b, label = int(self.anchors[p]), int(self.negatives[p]), 0.0
        return self.item_feat[a], self.item_feat[b], np.float32(label)


# ─────────────────────────────────────────────────────────────────────────────
# ModelEvaluator — dùng cache cho mọi I/O nặng
# ─────────────────────────────────────────────────────────────────────────────

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
            self.graph_dir = GRAPH_DIR / dataset / "clip"
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
        ckpt_name = f"{self.feature}_{self.model_name}.pth"
        ckpt_path = self.ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {ckpt_path}\nHãy chạy train.py trước."
            )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[CKPT] {ckpt_path.name}  epoch={ckpt.get('epoch','?')}  "
              f"best_val_f1={ckpt.get('best_f1', 0):.4f}")
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

    @torch.no_grad()
    def _run_gnn_inference(self, model, dataset, edge_index):
        model.eval()
        all_user_emb, all_item_emb = model(edge_index, use_cache=False)
        preds, labels = [], []
        for user_ids, item_ids, label in self._make_loader(dataset):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            with autocast("cuda"):
                score = torch.sigmoid(
                    (all_user_emb[user_ids] * all_item_emb[item_ids]).sum(dim=-1)
                )
            preds.extend(score.cpu().numpy())
            labels.extend(label.numpy())
        return np.array(preds), np.array(labels)

    @torch.no_grad()
    def _run_bpr_inference(self, model, dataset):
        model.eval()
        preds, labels = [], []
        for user_ids, item_ids, label in self._make_loader(dataset):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            score = torch.sigmoid(model.predict(user_ids, item_ids))
            preds.extend(score.cpu().numpy())
            labels.extend(label.numpy())
        return np.array(preds), np.array(labels)

    @torch.no_grad()
    def _run_siamese_inference(self, model, dataset):
        model.eval()
        loader = DataLoader(dataset, batch_size=_EVAL_BATCH,
                            shuffle=False, num_workers=0, pin_memory=False)
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

        if self.model_name in ALL_GNN_MODELS or self.model_name == "bpr":
            adj_csr = self._load_adj()
            test_users, test_items = self._load_split_arrays(
                "test", user2idx, item2idx, with_time=False)
            print(f"[TEST] interactions={len(test_users):,}")
            test_dataset = EvalDataset(test_users, test_items, n_items, adj_csr)

            model = get_model(self.model_name, num_users=n_users,
                              num_items=n_items, embedding_dim=EMBEDDING_DIM).to(self.device)
            model = self._load_checkpoint(model)

            if self.model_name in ALL_GNN_MODELS:
                edge_index = self._load_edge_index()
                if self.model_name == "lightgcn":
                    model.precompute_norm_adj(edge_index, n_users + n_items)
                return self._run_gnn_inference(model, test_dataset, edge_index)
            else:
                return self._run_bpr_inference(model, test_dataset)

        elif self.model_name == "siamese":
            item_feat = self._load_item_feat_numpy(n_items, item2idx)
            dim_in    = item_feat.shape[1]

            train_users, train_items, train_times = self._load_split_arrays(
                "train", user2idx, item2idx, with_time=True)
            test_users, test_items, test_times = self._load_split_arrays(
                "test",  user2idx, item2idx, with_time=True)
            print(f"[TEST] interactions={len(test_users):,}")

            item_counts  = np.bincount(train_items, minlength=n_items).astype(np.float32)
            test_dataset = SiameseEvalDataset(
                test_users, test_items, test_times,
                train_users, train_items, train_times,
                item_feat, item_counts,
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

        # Bước 1: Scan checkpoints
        candidates = []
        for m_name, m_feat in _ALL_MODELS:
            if feature is not None and m_feat != feature:
                continue
            ckpt_dir  = (CHECKPOINT_DIR / dataset / m_feat / m_name
                         if m_feat else CHECKPOINT_DIR / dataset / m_name)
            ckpt_name = f"{m_feat}_{m_name}.pth" if m_feat else f"{m_name}.pth"
            if (ckpt_dir / ckpt_name).exists():
                candidates.append((m_name, m_feat))

        if not candidates:
            print("[WARN] Không tìm thấy checkpoint nào.")
            return {}

        print(f"[SCAN] Tìm thấy {len(candidates)} checkpoint(s):")
        for m_name, m_feat in candidates:
            print(f"  • {m_feat}_{m_name}.pth")
        print()

        # Bước 2: Evaluate — data nặng tự động được cache
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

        # Bước 3: Ghi report một lần
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