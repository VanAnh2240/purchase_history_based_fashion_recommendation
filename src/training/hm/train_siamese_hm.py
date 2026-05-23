"""
src/training/hm/train_siamese_hm.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

from config import (
    PROCESSED_DIR, EMBEDDING_DIR, GRAPH_DIR, CHECKPOINT_DIR,
    DEVICE, SEED, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    THRESHOLD, NUM_WORKERS, LOGIT_SCALE
)
from src.models.siamese import SiameseProjector
from src.evaluation.metrics import calculate_metrics, print_metrics_table
from src.evaluation.evaluator import Evaluator

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

CSV_CHUNK         = 1_000_000
_TRAIN_BATCH_SIZE = BATCH_SIZE
_NUM_WORKERS      = 0
_GRAD_ACCUM       = 4
_LOGIT_SCALE      = LOGIT_SCALE
_LR_SIAMESE       = min(LEARNING_RATE, 1e-4)
_WARMUP_EPOCHS    = 3
_HARD_NEG_POOL_K  = 5_000


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_positive_pairs(user_arr: np.ndarray, item_arr: np.ndarray,
                         time_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized positive pair builder dùng pandas shift.
    Tránh hoàn toàn Python dict loop → không bị OOM với 30M rows.

    Ý tưởng:
      - Sort theo (user, time)
      - Với mỗi user, anchor[k] = item[k], positive[k] = item[k+1]
      - Dùng groupby shift(-1) để lấy item kế tiếp trong cùng user
      - Drop rows cuối mỗi group (NaN sau shift)
    """
    print("[PAIRS] Building positive pairs (vectorized)...")
    df = pd.DataFrame({"u": user_arr, "i": item_arr, "t": time_arr})
    del user_arr, item_arr, time_arr
    gc.collect()

    df.sort_values(["u", "t"], inplace=True, kind="mergesort")
    df["i_next"] = df.groupby("u")["i"].shift(-1)
    df.dropna(subset=["i_next"], inplace=True)

    anchors   = df["i"].values.astype(np.int32)
    positives = df["i_next"].values.astype(np.int32)
    del df
    gc.collect()

    print(f"[PAIRS] positive pairs: {len(anchors):,}")
    return anchors, positives


def build_eval_pairs(val_user: np.ndarray, val_item: np.ndarray,
                     val_time: np.ndarray,
                     train_user: np.ndarray, train_item: np.ndarray,
                     train_time: np.ndarray,
                     hard_neg_pool: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized eval pair builder.

    anchor  = item mới nhất của user trong train (last train item)
    positive= item sớm nhất của user trong val/test (first val item)
    negative= random từ hard_neg_pool (không check known để tránh loop)
    """
    print("[PAIRS] Building eval pairs (vectorized)...")

    train_df = pd.DataFrame({"u": train_user, "i": train_item, "t": train_time})
    last_train = (train_df.sort_values("t")
                          .groupby("u")["i"]
                          .last()
                          .reset_index()
                          .rename(columns={"i": "anchor"}))
    del train_df
    gc.collect()

    val_df = pd.DataFrame({"u": val_user, "i": val_item, "t": val_time})
    first_val = (val_df.sort_values("t")
                       .groupby("u")["i"]
                       .first()
                       .reset_index()
                       .rename(columns={"i": "positive"}))
    del val_df
    gc.collect()

    merged = last_train.merge(first_val, on="u", how="inner")
    del last_train, first_val
    gc.collect()

    n         = len(merged)
    neg_idx   = np.random.randint(0, len(hard_neg_pool), n)
    negatives = hard_neg_pool[neg_idx].astype(np.int32)

    anchors   = merged["anchor"].values.astype(np.int32)
    positives = merged["positive"].values.astype(np.int32)
    del merged
    gc.collect()

    print(f"[PAIRS] eval pairs: {n:,}")
    return anchors, positives, negatives


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class SiameseTrainDataset(Dataset):
    def __init__(self, anchors: np.ndarray, positives: np.ndarray,
                 item_feat: np.ndarray, item_counts: np.ndarray,
                 neg_ratio: int = 1):
        self.item_feat = item_feat
        self.neg_ratio = neg_ratio

        top_k = min(_HARD_NEG_POOL_K, item_feat.shape[0])
        self.hard_neg_pool = np.argsort(item_counts)[::-1][:top_k].astype(np.int32)

        self.anchors   = anchors
        self.positives = positives
        self._pos_len  = len(anchors)
        self.negatives = self._sample_neg()

    def _sample_neg(self):
        idx = np.random.randint(0, len(self.hard_neg_pool), self._pos_len)
        return self.hard_neg_pool[idx]

    def resample(self):
        self.negatives = self._sample_neg()

    def __len__(self):
        return self._pos_len * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        if idx < self._pos_len:
            a, b, label = int(self.anchors[idx]), int(self.positives[idx]), 1.0
        else:
            p = (idx - self._pos_len) % self._pos_len
            a, b, label = int(self.anchors[p]), int(self.negatives[p]), 0.0
        return self.item_feat[a], self.item_feat[b], np.float32(label)


class SiameseEvalDataset(Dataset):
    def __init__(self, anchors: np.ndarray, positives: np.ndarray,
                 negatives: np.ndarray, item_feat: np.ndarray,
                 neg_ratio: int = 1):
        self.item_feat = item_feat
        self.neg_ratio = neg_ratio
        self.anchors   = anchors
        self.positives = positives
        self.negatives = negatives
        self._pos_len  = len(anchors)

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
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class HMSiameseTrainer:

    def __init__(self, feature: str):
        assert feature in ("clip", "fashionclip")
        self.feature    = feature
        self.device     = DEVICE
        self.model_name = "siamese"
        self.data_dir   = PROCESSED_DIR / "hm"
        self.emb_dir    = EMBEDDING_DIR / "hm" / feature
        self.graph_dir  = GRAPH_DIR / "hm" / feature
        self.ckpt_dir   = CHECKPOINT_DIR / "hm" / feature / "siamese"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.scaler     = GradScaler("cuda")
        self.criterion  = nn.BCEWithLogitsLoss()
        set_seed(SEED)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_embeddings(self) -> tuple[np.ndarray, dict]:
        raw      = np.load(self.emb_dir / "embeddings.npy")
        ids      = (pd.read_csv(self.emb_dir / "article_ids.csv")
                    ["article_id"].astype(str).str.zfill(10).tolist())
        feat     = F.normalize(torch.tensor(raw, dtype=torch.float32), dim=-1).numpy()
        item2idx = {aid: idx for idx, aid in enumerate(ids)}
        print(f"[EMB] {feat.shape}  items={len(item2idx):,}")
        return feat, item2idx

    def _load_mappings(self) -> tuple[dict, dict]:
        """Load global mapping từ graph — nhất quán với GNN trainer."""
        user_df  = pd.read_csv(self.graph_dir / "user2idx.csv",
                               dtype={"customer_id": str, "user_idx": int})
        item_df  = pd.read_csv(self.graph_dir / "item2idx.csv",
                               dtype={"article_id": str, "item_idx": int})
        cust2idx = dict(zip(user_df["customer_id"], user_df["user_idx"].astype(np.int32)))
        item2idx = dict(zip(item_df["article_id"],  item_df["item_idx"].astype(np.int32)))
        print(f"[MAPPING] users={len(cust2idx):,}  items={len(item2idx):,}")
        return cust2idx, item2idx

    def _load_split(self, split: str, cust2idx: dict,
                    item2idx: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load split, map sang global index, trả (user, item, time) arrays."""
        user_bufs, item_bufs, time_bufs = [], [], []
        dropped = 0

        for chunk in pd.read_csv(
            self.data_dir / f"{split}.csv",
            usecols=["t_dat", "customer_id", "article_id"],
            dtype={"customer_id": str, "article_id": str},
            chunksize=CSV_CHUNK,
        ):
            chunk["article_id"] = chunk["article_id"].str.zfill(10)
            u = chunk["customer_id"].map(cust2idx)
            i = chunk["article_id"].map(item2idx)
            mask = u.notna() & i.notna()
            dropped += (~mask).sum()

            chunk_ok = chunk[mask]
            t = pd.to_datetime(chunk_ok["t_dat"]).astype(np.int64).values // 10**9

            user_bufs.append(u[mask].values.astype(np.int32))
            item_bufs.append(i[mask].values.astype(np.int32))
            time_bufs.append(t.astype(np.int64))

        if dropped:
            print(f"  [MAP] {split}: dropped {dropped:,}")

        users = np.concatenate(user_bufs)
        items = np.concatenate(item_bufs)
        times = np.concatenate(time_bufs)
        print(f"  [MAP] {split}: {len(users):,}")
        return users, items, times

    @staticmethod
    def _score(model, ea, eb):
        return (model(ea) * model(eb)).sum(-1) * _LOGIT_SCALE

    @staticmethod
    def _make_scheduler(optimizer, num_epochs, warmup_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── train / eval ──────────────────────────────────────────────────────────

    def _train_epoch(self, model, loader, optimizer, epoch):
        model.train()
        total, count = 0.0, 0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]",
                    dynamic_ncols=True, leave=False)
        for step, (ea, eb, label) in enumerate(pbar):
            ea    = ea.to(self.device, non_blocking=True)
            eb    = eb.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            is_last = (step + 1) % _GRAD_ACCUM == 0 or (step + 1) == len(loader)
            with autocast("cuda"):
                loss = self.criterion(self._score(model, ea, eb), label) / _GRAD_ACCUM
            self.scaler.scale(loss).backward()
            if is_last:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            total += loss.item() * _GRAD_ACCUM
            count += 1
            pbar.set_postfix(loss=f"{loss.item()*_GRAD_ACCUM:.4f}",
                             avg=f"{total/count:.4f}")
        return total / count

    @torch.no_grad()
    def _run_eval(self, model, dataset):
        model.eval()
        loader = DataLoader(dataset, batch_size=_TRAIN_BATCH_SIZE * 4,
                            shuffle=False, num_workers=0, pin_memory=False)
        preds, labels = [], []
        for ea, eb, label in tqdm(loader, desc="Evaluating", leave=False):
            with autocast("cuda"):
                score = self._score(model,
                                    ea.to(self.device, non_blocking=True),
                                    eb.to(self.device, non_blocking=True))
            preds.extend(score.cpu().numpy())
            labels.extend(label.numpy())
        return np.array(preds), np.array(labels)

    # ── main ──────────────────────────────────────────────────────────────────

    def train(self, evaluator: "Evaluator | None" = None):
        print(f"\n{'='*60}\n H&M Siamese | feature={self.feature}\n{'='*60}\n")

        item_feat, _ = self._load_embeddings()
        n_items      = item_feat.shape[0]

        cust2idx, item2idx = self._load_mappings()

        print("[SETUP] Loading splits...")
        train_u, train_i, train_t = self._load_split("train", cust2idx, item2idx)
        val_u,   val_i,   val_t   = self._load_split("val",   cust2idx, item2idx)
        test_u,  test_i,  test_t  = self._load_split("test",  cust2idx, item2idx)
        del cust2idx, item2idx
        gc.collect()

        item_counts = np.bincount(train_i, minlength=n_items).astype(np.float32)
        top_k       = min(_HARD_NEG_POOL_K, n_items)
        hard_neg_pool = np.argsort(item_counts)[::-1][:top_k].astype(np.int32)
        print(f"[NEG] hard_neg_pool: {len(hard_neg_pool):,}")

        tr_anc, tr_pos = build_positive_pairs(train_u, train_i, train_t)
        val_anc, val_pos, val_neg = build_eval_pairs(
            val_u, val_i, val_t, train_u, train_i, train_t, hard_neg_pool)
        test_anc, test_pos, test_neg = build_eval_pairs(
            test_u, test_i, test_t, train_u, train_i, train_t, hard_neg_pool)
        del train_u, train_i, train_t, val_u, val_i, val_t, test_u, test_i, test_t
        gc.collect()

        train_ds = SiameseTrainDataset(tr_anc, tr_pos, item_feat, item_counts)
        val_ds   = SiameseEvalDataset(val_anc,  val_pos,  val_neg,  item_feat)
        test_ds  = SiameseEvalDataset(test_anc, test_pos, test_neg, item_feat)

        train_loader = DataLoader(
            train_ds,
            batch_size=_TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
        )

        print(f"[LOADER] batch={_TRAIN_BATCH_SIZE}  workers=0  "
              f"accum={_GRAD_ACCUM}  "
              f"effective={_TRAIN_BATCH_SIZE*_GRAD_ACCUM}  "
              f"batches={len(train_loader):,}")

        model     = SiameseProjector(dim_in=item_feat.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=_LR_SIAMESE, weight_decay=WEIGHT_DECAY)
        scheduler = self._make_scheduler(optimizer, NUM_EPOCHS, _WARMUP_EPOCHS)

        print(f"[MODEL] params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}  "
              f"lr={_LR_SIAMESE}  scale={_LOGIT_SCALE}  warmup={_WARMUP_EPOCHS}\n")

        ckpt_name = f"{self.feature}_{self.model_name}.pth"
        ckpt_path = self.ckpt_dir / ckpt_name

        start_epoch = 1
        best_f1     = 0.0
        best_epoch  = 0

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_f1     = ckpt["best_f1"]
            best_epoch  = ckpt["epoch"]
            print(f"[RESUME] epoch {start_epoch}, best F1={best_f1:.4f}\n")
        else:
            print("[RESUME] train mới.\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            train_ds.resample()
            train_loss = self._train_epoch(model, train_loader, optimizer, epoch)
            scheduler.step()

            val_preds, val_labels = self._run_eval(model, val_ds)
            val_m = calculate_metrics(val_preds, val_labels, threshold=THRESHOLD)
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch [{epoch:>3}/{NUM_EPOCHS}] loss={train_loss:.4f} | "
                  f"acc={val_m['Accuracy']:.4f}  "
                  f"prec={val_m['Precision']:.4f}  "
                  f"rec={val_m['Recall']:.4f}  "
                  f"f1={val_m['F1-score']:.4f}  "
                  f"lr={current_lr:.2e}")

            if val_m["F1-score"] > best_f1:
                best_f1, best_epoch = val_m["F1-score"], epoch
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "best_f1": best_f1}, ckpt_path)
                print(f"  ✓ Saved {ckpt_name} (f1={best_f1:.4f}, epoch={epoch})\n")

        print(f"\n[TRAIN DONE] Best val F1={best_f1:.4f} at epoch {best_epoch}")

        print("\n[TEST] Loading best checkpoint...")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=False)["model"]
        )
        test_preds, test_labels = self._run_eval(model, test_ds)
        test_m = calculate_metrics(test_preds, test_labels, threshold=THRESHOLD)
        print_metrics_table(test_m, title=f"HM | {self.feature.upper()} | SIAMESE")

        if evaluator is not None:
            # FIX 4: dùng đúng tên tham số preds/targets (không phải all_preds/all_targets)
            evaluator.evaluate(
                dataset=dataset, feature=self.feature,
                model_name="siamese",
                preds=test_preds, targets=test_labels,
            )
        return test_m