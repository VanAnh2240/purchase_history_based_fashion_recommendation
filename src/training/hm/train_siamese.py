"""
src/training/hm/train_siamese.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

from config import (
    PROCESSED_DIR, EMBEDDING_DIR, GRAPH_DIR, CHECKPOINT_DIR,
    DEVICE, SEED, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    THRESHOLD, NUM_WORKERS,
)
from src.models.siamese import SiameseProjector
from src.evaluation.metrics import calculate_metrics, print_metrics_table
from src.evaluation.evaluator import Evaluator
from src.training.hm.precompute_user_emb import get_user_hist_emb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

CSV_CHUNK         = 1_000_000
_TRAIN_BATCH_SIZE = BATCH_SIZE 
_NUM_WORKERS      = 8
_GRAD_ACCUM       = 2
_LR_SIAMESE       = min(LEARNING_RATE, 1e-4)
_WARMUP_EPOCHS    = 3


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_shared_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t

class SiameseUserItemModel(nn.Module):
    def __init__(self, dim_in: int, dim_out: int = 256, dropout: float = 0.2):
        super().__init__()
        self.projector = SiameseProjector(dim_in, dim_out, dropout)

    def encode_user(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)

    def forward(self, user_hist: torch.Tensor,
                item_feat: torch.Tensor) -> torch.Tensor:
        return (self.encode_user(user_hist) * self.encode_item(item_feat)).sum(-1)


class SiameseTrainDataset(Dataset):
    def __init__(self, user_arr, item_arr, user_hist_emb, item_feat, n_items):
        assert isinstance(user_hist_emb, torch.Tensor), \
        assert isinstance(item_feat, torch.Tensor), \
        self.user_hist_emb = user_hist_emb
        self.item_feat     = item_feat
        self.user_arr      = user_arr
        self.item_arr      = item_arr
        self.n_items       = n_items
        self.negs          = np.random.randint(0, n_items, len(user_arr), dtype=np.int32)

    def resample(self):
        self.negs = np.random.randint(0, self.n_items, len(self.user_arr), dtype=np.int32)

    def __len__(self):
        return len(self.user_arr)

    def __getitem__(self, idx):
        u   = int(self.user_arr[idx])
        pos = int(self.item_arr[idx])
        neg = int(self.negs[idx])
        return (
            self.user_hist_emb[u],
            self.item_feat[pos],
            self.item_feat[neg],
        )


class SiameseEvalDataset(Dataset):
   def __init__(self, user_arr, item_arr, neg_arr, user_hist_emb, item_feat):
        assert isinstance(user_hist_emb, torch.Tensor), \
        assert isinstance(item_feat, torch.Tensor), \
        self.user_hist_emb = user_hist_emb
        self.item_feat     = item_feat
        self.user_arr      = user_arr
        self.item_arr      = item_arr
        self.neg_arr       = neg_arr
        self._pos_len      = len(user_arr)

    def __len__(self):
        return self._pos_len * 2

    def __getitem__(self, idx):
        if idx < self._pos_len:
            u    = int(self.user_arr[idx])
            item = int(self.item_arr[idx])
            lbl  = 1.0
        else:
            p    = idx - self._pos_len
            u    = int(self.user_arr[p])
            item = int(self.neg_arr[p])
            lbl  = 0.0
        return (
            self.user_hist_emb[u],
            self.item_feat[item],
            torch.tensor(lbl, dtype=torch.float32),
        )

def bpr_loss(u_emb, pos_emb, neg_emb):
    return -F.logsigmoid(
        (u_emb * pos_emb).sum(-1) - (u_emb * neg_emb).sum(-1)
    ).mean()

class HMSiameseTrainer:

    def __init__(self, feature: str):
        assert feature in ("clip", "fashionclip")
        self.feature    = feature
        self.device     = DEVICE
        self.model_name = "siamese"
        self.data_dir   = PROCESSED_DIR / "hm"
        self.emb_dir    = EMBEDDING_DIR / "hm" / feature
        self.graph_dir  = GRAPH_DIR     / "hm" / feature
        self.ckpt_dir   = CHECKPOINT_DIR / "hm" / feature / "siamese"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.scaler     = GradScaler("cuda")
        set_seed(SEED)

    def _load_embeddings(self):
        raw  = np.load(self.emb_dir / "embeddings.npy")
        ids  = (pd.read_csv(self.emb_dir / "article_ids.csv")
                ["article_id"].astype(str).str.zfill(10).tolist())
        feat = F.normalize(torch.tensor(raw, dtype=torch.float32), dim=-1).numpy()
        item2idx = {aid: idx for idx, aid in enumerate(ids)}
        print(f"[EMB] {feat.shape}  items={len(item2idx):,}")
        return feat, item2idx

    def _load_mappings(self):
        user_df = pd.read_csv(self.graph_dir / "user2idx.csv",
                              dtype={"customer_id": str, "user_idx": int})
        item_df = pd.read_csv(self.graph_dir / "item2idx.csv",
                              dtype={"article_id": str, "item_idx": int})
        cust2idx = dict(zip(user_df["customer_id"],
                            user_df["user_idx"].astype(np.int32)))
        item2idx = dict(zip(item_df["article_id"],
                            item_df["item_idx"].astype(np.int32)))
        print(f"[MAPPING] users={len(cust2idx):,}  items={len(item2idx):,}")
        return cust2idx, item2idx

    def _load_split(self, split, cust2idx, item2idx):
        u_bufs, i_bufs = [], []
        dropped = 0
        for chunk in pd.read_csv(
            self.data_dir / f"{split}.csv",
            usecols=["customer_id", "article_id"],
            dtype=str, chunksize=CSV_CHUNK,
        ):
            chunk["article_id"] = chunk["article_id"].str.zfill(10)
            u    = chunk["customer_id"].map(cust2idx)
            i    = chunk["article_id"].map(item2idx)
            mask = u.notna() & i.notna()
            dropped += (~mask).sum()
            u_bufs.append(u[mask].values.astype(np.int32))
            i_bufs.append(i[mask].values.astype(np.int32))
        if dropped:
            print(f"  [MAP] {split}: dropped {dropped:,}")
        users = np.concatenate(u_bufs)
        items = np.concatenate(i_bufs)
        print(f"  [MAP] {split}: {len(users):,}")
        return users, items

    def _align_item_feat(self, raw_feat, raw_item2idx, global_item2idx, n_items):
        dim  = raw_feat.shape[1]
        feat = np.zeros((n_items, dim), dtype=np.float32)
        for aid, raw_idx in raw_item2idx.items():
            gidx = global_item2idx.get(aid)
            if gidx is not None:
                feat[gidx] = raw_feat[raw_idx]
        return feat

    @staticmethod
    def _make_scheduler(optimizer, num_epochs, warmup_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            p = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + np.cos(np.pi * p))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── train / eval ──────────────────────────────────────────────────────────

    def _train_epoch(self, model, loader, optimizer, epoch):
        model.train()
        total, count = 0.0, 0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]",
                    dynamic_ncols=True, leave=False)
        for step, (u_hist, pos_feat, neg_feat) in enumerate(pbar):
            u_hist   = u_hist.to(self.device,   non_blocking=True)
            pos_feat = pos_feat.to(self.device,  non_blocking=True)
            neg_feat = neg_feat.to(self.device,  non_blocking=True)
            is_last  = ((step + 1) % _GRAD_ACCUM == 0
                        or (step + 1) == len(loader))
            with autocast("cuda"):
                u_emb   = model.encode_user(u_hist)
                pos_emb = model.encode_item(pos_feat)
                neg_emb = model.encode_item(neg_feat)
                loss    = bpr_loss(u_emb, pos_emb, neg_emb) / _GRAD_ACCUM
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
        loader = DataLoader(
            dataset,
            batch_size=_TRAIN_BATCH_SIZE * 4,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        preds, labels = [], []
        for u_hist, item_f, label in tqdm(loader, desc="Evaluating", leave=False):
            u_hist = u_hist.to(self.device, non_blocking=True)
            item_f = item_f.to(self.device, non_blocking=True)
            with autocast("cuda"):
                score = torch.sigmoid(model(u_hist, item_f))
            preds.extend(score.cpu().numpy())
            labels.extend(label.numpy())
        return np.array(preds), np.array(labels)

    # ── main ──────────────────────────────────────────────────────────────────

    def train(self, evaluator=None, dataset: str = "hm"):
        print(f"\n{'='*60}")
        print(f" H&M Siamese Baseline | feature={self.feature}")
        print(f" Bài toán: score(user, item) — đồng nhất với GNN")
        print(f"{'='*60}\n")

        # 1. CLIP embeddings
        raw_feat, raw_item2idx = self._load_embeddings()

        # 2. Global mappings
        cust2idx, global_item2idx = self._load_mappings()
        n_users = len(cust2idx)
        n_items = len(global_item2idx)
        print(f"[SETUP] n_users={n_users:,}  n_items={n_items:,}")

        # 3. Align item_feat → global index
        item_feat = self._align_item_feat(
            raw_feat, raw_item2idx, global_item2idx, n_items)
        print(f"[SETUP] item_feat (aligned): {item_feat.shape}")
        del raw_feat, raw_item2idx
        gc.collect()

        # 4. Load splits
        print("[SETUP] Loading splits...")
        train_u, train_i = self._load_split("train", cust2idx, global_item2idx)
        val_u,   val_i   = self._load_split("val",   cust2idx, global_item2idx)
        test_u,  test_i  = self._load_split("test",  cust2idx, global_item2idx)
        del cust2idx, global_item2idx
        gc.collect()

        # 5. User history embeddings
        # Chạy trước: python -m src.training.hm.precompute_user_emb --feature <feature>
        # Cache tại: EMBEDDING_DIR/hm/<feature>/user_hist_emb.npy
        user_hist_emb_np = get_user_hist_emb(
            feature   = self.feature,
            emb_dir   = self.emb_dir,
            graph_dir = self.graph_dir,
            data_dir  = self.data_dir,
        )

        # 6. Chuyển sang shared memory tensor TRƯỚC KHI tạo Dataset
        print("[SETUP] Moving arrays to shared memory...")
        user_hist_emb = to_shared_tensor(user_hist_emb_np)
        item_feat_t   = to_shared_tensor(item_feat)
        del user_hist_emb_np, item_feat
        gc.collect()
        print(f"[SETUP] user_hist_emb: {tuple(user_hist_emb.shape)}  "
              f"item_feat: {tuple(item_feat_t.shape)}")

        # 7. Train dataset
        train_ds = SiameseTrainDataset(
            train_u, train_i, user_hist_emb, item_feat_t, n_items)
        del train_u, train_i
        gc.collect()

        # 8. Eval datasets
        val_neg  = np.random.randint(0, n_items, len(val_u),  dtype=np.int32)
        test_neg = np.random.randint(0, n_items, len(test_u), dtype=np.int32)

        val_ds  = SiameseEvalDataset(val_u,  val_i,  val_neg,  user_hist_emb, item_feat_t)
        test_ds = SiameseEvalDataset(test_u, test_i, test_neg, user_hist_emb, item_feat_t)
        del val_neg, test_neg, val_u, val_i, test_u, test_i
        gc.collect()

        # 9. DataLoader
        train_loader = DataLoader(
            train_ds,
            batch_size=_TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,  
            prefetch_factor=4,       
        )
        print(f"[LOADER] batch={_TRAIN_BATCH_SIZE}  accum={_GRAD_ACCUM}  "
              f"effective={_TRAIN_BATCH_SIZE*_GRAD_ACCUM}  "
              f"batches={len(train_loader):,}")

        # 10. Model
        dim_in = user_hist_emb.shape[1]
        model  = SiameseUserItemModel(dim_in=dim_in).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=_LR_SIAMESE, weight_decay=WEIGHT_DECAY)
        scheduler = self._make_scheduler(optimizer, NUM_EPOCHS, _WARMUP_EPOCHS)
        print(f"[MODEL] params="
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}  "
              f"lr={_LR_SIAMESE}  warmup={_WARMUP_EPOCHS}\n")

        # 11. Checkpoint
        ckpt_name = f"{self.feature}_{self.model_name}_user.pth"
        ckpt_path = self.ckpt_dir / ckpt_name
        start_epoch, best_f1, best_epoch = 1, 0.0, 0

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_f1     = ckpt["best_f1"]
            best_epoch  = ckpt["epoch"]
            print(f"[RESUME] epoch {start_epoch}, best F1={best_f1:.4f}\n")
        else:
            print("[RESUME] Train mới.\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 12. Training loop
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            train_ds.resample()
            train_loss = self._train_epoch(model, train_loader, optimizer, epoch)
            scheduler.step()

            val_preds, val_labels = self._run_eval(model, val_ds)
            val_m = calculate_metrics(val_preds, val_labels, threshold=THRESHOLD)
            lr    = optimizer.param_groups[0]["lr"]

            print(f"Epoch [{epoch:>3}/{NUM_EPOCHS}] loss={train_loss:.4f} | "
                  f"acc={val_m['Accuracy']:.4f}  "
                  f"prec={val_m['Precision']:.4f}  "
                  f"rec={val_m['Recall']:.4f}  "
                  f"f1={val_m['F1-score']:.4f}  "
                  f"lr={lr:.2e}")

            if val_m["F1-score"] > best_f1:
                best_f1, best_epoch = val_m["F1-score"], epoch
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "best_f1": best_f1}, ckpt_path)
                print(f"  ✓ Saved {ckpt_name} (f1={best_f1:.4f}, epoch={epoch})\n")

        print(f"\n[TRAIN DONE] Best val F1={best_f1:.4f} at epoch {best_epoch}")

        # 13. Test
        print("\n[TEST] Loading best checkpoint...")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device,
                       weights_only=False)["model"])
        test_preds, test_labels = self._run_eval(model, test_ds)
        test_m = calculate_metrics(test_preds, test_labels, threshold=THRESHOLD)
        print_metrics_table(test_m, title=f"HM | {self.feature.upper()} | SIAMESE")

        if evaluator is not None:
            evaluator.evaluate(
                dataset=dataset, feature=self.feature,
                model_name="siamese",
                preds=test_preds, targets=test_labels,
            )
        return test_m