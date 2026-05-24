# File src/training/hm/train_bpr_hm.py

import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    PROCESSED_DIR, CHECKPOINT_DIR,
    DEVICE, SEED,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EMBEDDING_DIM, LOGIT_SCALE, NUM_WORKERS, THRESHOLD,
)
from src.models.bpr import BPRModel

_CSV_CHUNK   = 500_000
_NUM_WORKERS = 4

# FIX: override các hyperparams không phù hợp cho BPR large-scale
# LEARNING_RATE từ config (2e-4) quá nhỏ → dùng 1e-3
# WEIGHT_DECAY từ config (1e-4) quá lớn → dùng 1e-6
# LOGIT_SCALE từ config (2.0) scale down score → dùng 1.0
_LR           = 1e-3
_WEIGHT_DECAY = 1e-6
_LOGIT_SCALE  = 1.0


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class BPRTrainDataset(Dataset):
    """
    Triplet (user, pos, neg). Negatives pre-generated — resample() mỗi epoch.
    Không gọi adj_csr trong __getitem__ → DataLoader multi-worker được.
    """
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, n_items: int):
        self.users   = user_ids
        self.items   = item_ids
        self.n_items = n_items
        self.negs = np.random.randint(0, n_items, len(user_ids), dtype=np.int32)

    def resample(self):
        """Gọi đầu mỗi epoch để thay negative mới."""
        self.negs = np.random.randint(0, self.n_items, len(self.users), dtype=np.int32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(int(self.users[idx]), dtype=torch.long),
            torch.tensor(int(self.items[idx]), dtype=torch.long),
            torch.tensor(int(self.negs[idx]),  dtype=torch.long),
        )


class BPREvalDataset(Dataset):
    """
    Balanced 1:1 pos:neg. Negatives pre-generated lúc __init__
    → không adj_csr trong __getitem__ → multi-worker OK.
    """
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray,
                 n_items: int, adj_csr: sp.csr_matrix):
        pos_u = user_ids.tolist()
        pos_i = item_ids.tolist()
        n     = len(pos_u)

        print(f"    pre-generating {n:,} negatives...")
        neg_i = np.random.randint(0, n_items, n, dtype=np.int32)

        pos_arr = np.array(pos_i, dtype=np.int32)
        bad     = np.where(neg_i == pos_arr)[0]
        while len(bad) > 0:
            neg_i[bad] = np.random.randint(0, n_items, len(bad), dtype=np.int32)
            bad = bad[neg_i[bad] == pos_arr[bad]]

        all_u = np.concatenate([np.array(pos_u, dtype=np.int32),
                                 np.array(pos_u, dtype=np.int32)])
        all_i = np.concatenate([pos_arr, neg_i])
        all_l = np.concatenate([np.ones(n, dtype=np.float32),
                                 np.zeros(n, dtype=np.float32)])

        self.users  = torch.from_numpy(all_u)
        self.items  = torch.from_numpy(all_i.astype(np.int32))
        self.labels = torch.from_numpy(all_l)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class HMBPRTrainer:

    def __init__(self):
        self.data_dir = PROCESSED_DIR / "hm"
        self.ckpt_dir = CHECKPOINT_DIR / "hm" / "bpr"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device   = DEVICE
        set_seed(SEED)

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_idx_maps(self) -> tuple[dict, dict]:
        print("[BPR] Pass 1: collecting unique IDs from train.csv...")
        unique_users: set[str] = set()
        unique_items: set[str] = set()

        for chunk in tqdm(
            pd.read_csv(
                self.data_dir / "train.csv",
                usecols=["customer_id", "article_id"],
                dtype=str, chunksize=_CSV_CHUNK,
            ), desc="Pass 1",
        ):
            chunk["article_id"] = chunk["article_id"].str.zfill(10)
            unique_users.update(chunk["customer_id"].tolist())
            unique_items.update(chunk["article_id"].tolist())

        user2idx = {u: i for i, u in enumerate(sorted(unique_users))}
        item2idx = {it: i for i, it in enumerate(sorted(unique_items))}
        del unique_users, unique_items
        gc.collect()
        print(f"[BPR] users={len(user2idx):,}  items={len(item2idx):,}")
        return user2idx, item2idx

    def _load_split(self, split: str, user2idx: dict,
                    item2idx: dict) -> tuple[np.ndarray, np.ndarray]:
        u_bufs, i_bufs = [], []
        dropped = 0
        for chunk in pd.read_csv(
            self.data_dir / f"{split}.csv",
            usecols=["customer_id", "article_id"],
            dtype=str, chunksize=_CSV_CHUNK,
        ):
            chunk["article_id"] = chunk["article_id"].str.zfill(10)
            u_m = chunk["customer_id"].map(user2idx)
            i_m = chunk["article_id"].map(item2idx)
            mask = u_m.notna() & i_m.notna()
            dropped += (~mask).sum()
            u_bufs.append(u_m[mask].values.astype(np.int32))
            i_bufs.append(i_m[mask].values.astype(np.int32))

        if dropped:
            print(f"  [MAP] {split}: dropped {dropped:,} unknown rows")
        users = np.concatenate(u_bufs)
        items = np.concatenate(i_bufs)
        print(f"  [MAP] {split}: {len(users):,} interactions")
        return users, items

    def _build_adj_csr(self, user_ids: np.ndarray, item_ids: np.ndarray,
                       n_users: int, n_items: int) -> sp.csr_matrix:
        data = np.ones(len(user_ids), dtype=np.float32)
        adj  = sp.csr_matrix(
            (data, (user_ids.astype(np.int64), item_ids.astype(np.int64))),
            shape=(n_users, n_items),
        )
        mb = (adj.data.nbytes + adj.indices.nbytes + adj.indptr.nbytes) / 1e6
        print(f"  [ADJ] shape={adj.shape}  nnz={adj.nnz:,}  size≈{mb:.0f}MB")
        return adj

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict:
        y_pred = (preds >= THRESHOLD).astype(int)
        y_true = labels.astype(int)
        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        }

    # ── train one epoch ───────────────────────────────────────────────────

    def _train_epoch(self, model, loader, optimizer, epoch: int) -> float:
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]",
                    dynamic_ncols=True)
        for user_ids, pos_ids, neg_ids in pbar:
            user_ids = user_ids.to(self.device, non_blocking=True)
            pos_ids  = pos_ids.to(self.device,  non_blocking=True)
            neg_ids  = neg_ids.to(self.device,  non_blocking=True)

            optimizer.zero_grad()
            loss = model.bpr_loss(user_ids, pos_ids, neg_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             avg=f"{total_loss / (pbar.n + 1):.4f}")

        return total_loss / len(loader)

    # ── evaluate ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_eval(self, model, dataset) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE * 4,
            shuffle=False, num_workers=_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        preds, labels = [], []
        for user_ids, item_ids, label in tqdm(loader, desc="Evaluating", leave=False):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            score    = torch.sigmoid(model.predict(user_ids, item_ids))
            preds.extend(score.cpu().numpy())
            labels.extend(label.numpy())
        return np.array(preds), np.array(labels)

    # ── main entry ────────────────────────────────────────────────────────

    def train(self, evaluator=None):
        print(f"\n{'='*60}")
        print(f" H&M BPR | Collaborative Filtering Baseline")
        print(f"{'='*60}\n")

        # 1. Build idx maps
        user2idx, item2idx = self._build_idx_maps()
        n_users, n_items   = len(user2idx), len(item2idx)

        # 2. Load splits
        print("[BPR] Loading splits...")
        train_users, train_items = self._load_split("train", user2idx, item2idx)
        val_users,   val_items   = self._load_split("val",   user2idx, item2idx)
        test_users,  test_items  = self._load_split("test",  user2idx, item2idx)
        del user2idx, item2idx
        gc.collect()

        # 3. Build CSR adj (train only)
        print("[BPR] Building CSR adjacency matrix...")
        adj_csr = self._build_adj_csr(train_users, train_items, n_users, n_items)

        # 4. Datasets — negatives pre-generated, không lazy trong __getitem__
        print("[BPR] Building datasets (pre-generating negatives)...")
        train_dataset = BPRTrainDataset(train_users, train_items, n_items)
        print("  [DS] val:")
        val_dataset   = BPREvalDataset(val_users,   val_items,   n_items, adj_csr)
        print("  [DS] test:")
        test_dataset  = BPREvalDataset(test_users,  test_items,  n_items, adj_csr)
        del adj_csr
        gc.collect()

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=_NUM_WORKERS, pin_memory=torch.cuda.is_available(),
            drop_last=True, persistent_workers=(_NUM_WORKERS > 0),
            generator=torch.Generator().manual_seed(SEED),
        )
        print(f"[LOADER] batch={BATCH_SIZE}  workers={_NUM_WORKERS}  "
              f"batches/epoch={len(train_loader):,}")

        # 5. Model & optimizer
        # FIX: dùng _LOGIT_SCALE=1.0 thay vì LOGIT_SCALE=2.0 từ config
        model = BPRModel(
            n_users=n_users, n_items=n_items,
            embedding_dim=EMBEDDING_DIM, logit_scale=_LOGIT_SCALE,
        ).to(self.device)

        # FIX: dùng _LR=1e-3 và _WEIGHT_DECAY=1e-6 thay vì giá trị từ config
        optimizer = optim.Adam(
            model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )

        print(f"[MODEL] BPR-MF | lr={_LR} wd={_WEIGHT_DECAY} logit_scale={_LOGIT_SCALE} | "
              f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

        # 6. Resume checkpoint
        ckpt_path   = self.ckpt_dir / "bpr.pth"
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

        # 7. Training loop
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            train_dataset.resample()
            train_loss = self._train_epoch(model, train_loader, optimizer, epoch)

            val_preds, val_labels = self._run_eval(model, val_dataset)
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
                    "n_users":   n_users,
                    "n_items":   n_items,
                }, ckpt_path)
                print(f"  ✓ Saved bpr.pth (f1={best_f1:.4f}, epoch={epoch})\n")

        print(f"\n[TRAIN DONE] Best val F1={best_f1:.4f} at epoch {best_epoch}")

        # 8. Test evaluation
        print("\n[TEST] Loading best checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        test_preds, test_labels = self._run_eval(model, test_dataset)
        test_metrics = self._compute_metrics(test_preds, test_labels)

        print(
            f"[TEST] acc={test_metrics['accuracy']:.4f} "
            f"prec={test_metrics['precision']:.4f} "
            f"rec={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f}"
        )
        return test_metrics