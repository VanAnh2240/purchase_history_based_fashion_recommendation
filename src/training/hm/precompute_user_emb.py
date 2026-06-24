"""
src/training/hm/precompute_user_emb.py

python -m src.training.hm.precompute_user_emb --feature clip
python -m src.training.hm.precompute_user_emb --feature fashionclip
python -m src.training.hm.precompute_user_emb --feature clip --force
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from config import (
    PROCESSED_DIR,
    EMBEDDING_DIR,
    GRAPH_DIR,
)

CSV_CHUNK = 1_000_000
_USER_EMB_CHUNK = 500_000


def _load_embeddings(emb_dir: Path):
    raw = np.load(emb_dir / "embeddings.npy")

    ids = (
        pd.read_csv(emb_dir / "article_ids.csv")["article_id"]
        .astype(str)
        .str.zfill(10)
        .tolist()
    )

    feat = F.normalize(
        torch.tensor(raw, dtype=torch.float32),
        dim=-1
    ).numpy()

    item2idx = {aid: idx for idx, aid in enumerate(ids)}

    print(f"[EMB] {feat.shape}  items={len(item2idx):,}")
    return feat, item2idx


def _load_mappings(graph_dir: Path):
    user_df = pd.read_csv(
        graph_dir / "user2idx.csv",
        dtype={"customer_id": str, "user_idx": int}
    )

    item_df = pd.read_csv(
        graph_dir / "item2idx.csv",
        dtype={"article_id": str, "item_idx": int}
    )

    cust2idx = dict(
        zip(user_df["customer_id"],
            user_df["user_idx"].astype(np.int32))
    )

    item2idx = dict(
        zip(item_df["article_id"],
            item_df["item_idx"].astype(np.int32))
    )

    print(f"[MAPPING] users={len(cust2idx):,}  items={len(item2idx):,}")
    return cust2idx, item2idx


def _align_item_feat(raw_feat, raw_item2idx, global_item2idx, n_items):
    dim = raw_feat.shape[1]
    feat = np.zeros((n_items, dim), dtype=np.float32)

    for aid, raw_idx in raw_item2idx.items():
        gidx = global_item2idx.get(aid)
        if gidx is not None:
            feat[gidx] = raw_feat[raw_idx]

    return feat


def _load_train_split(data_dir: Path, cust2idx, item2idx):
    u_bufs, i_bufs = [], []
    dropped = 0

    for chunk in pd.read_csv(
        data_dir / "train.csv",
        usecols=["customer_id", "article_id"],
        dtype=str,
        chunksize=CSV_CHUNK
    ):
        chunk["article_id"] = chunk["article_id"].str.zfill(10)

        u = chunk["customer_id"].map(cust2idx)
        i = chunk["article_id"].map(item2idx)

        mask = u.notna() & i.notna()

        dropped += (~mask).sum()

        u_bufs.append(u[mask].values.astype(np.int32))
        i_bufs.append(i[mask].values.astype(np.int32))

    if dropped:
        print(f"  [MAP] train: dropped {dropped:,}")

    users = np.concatenate(u_bufs)
    items = np.concatenate(i_bufs)

    print(f"  [MAP] train: {len(users):,} interactions")
    return users, items


def _close_memmap(mm):
    if mm is not None:
        mm.flush()
        if hasattr(mm, "_mmap") and mm._mmap is not None:
            mm._mmap.close()


def compute_user_hist_emb(
    train_user: np.ndarray,
    train_item: np.ndarray,
    item_feat: np.ndarray,
    n_users: int,
    out_path: Path,
):
    dim = item_feat.shape[1]
    n = len(train_user)

    tmp_accum = out_path.with_suffix(".accum.tmp")
    tmp_count = out_path.with_suffix(".count.tmp")

    print(f"[BUILD] Accumulating {n:,} interactions into memmap...")

    accum = np.memmap(
        str(tmp_accum),
        dtype=np.float32,
        mode="w+",
        shape=(n_users, dim)
    )

    count = np.memmap(
        str(tmp_count),
        dtype=np.int32,
        mode="w+",
        shape=(n_users,)
    )

    accum[:] = 0
    count[:] = 0

    for start in tqdm(range(0, n, _USER_EMB_CHUNK), desc="  mean-pool"):
        end = min(start + _USER_EMB_CHUNK, n)

        u_chunk = train_user[start:end]
        feat_chunk = item_feat[train_item[start:end]]

        np.add.at(accum, u_chunk, feat_chunk)
        np.add.at(count, u_chunk, 1)

        del feat_chunk
        gc.collect()

    print("[BUILD] Normalizing...")

    users_with_hist = 0
    norm_chunk = 100_000

    for start in range(0, n_users, norm_chunk):
        end = min(start + norm_chunk, n_users)

        c = count[start:end]
        mask = c > 0

        users_with_hist += int(mask.sum())

        rows = accum[start:end].copy()

        rows[mask] /= c[mask, None]

        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)

        rows /= norms
        accum[start:end] = rows

    print(f"[BUILD] users_with_history={users_with_hist:,} / {n_users:,}")

    # save final
    print(f"[BUILD] Saving to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fp = np.lib.format.open_memmap(
        str(out_path),
        mode="w+",
        dtype=np.float32,
        shape=(n_users, dim)
    )

    for start in range(0, n_users, 200_000):
        end = min(start + 200_000, n_users)
        fp[start:end] = accum[start:end]

    # cleanup safely
    _close_memmap(fp)
    _close_memmap(accum)
    _close_memmap(count)

    del fp, accum, count
    gc.collect()

    time.sleep(1)  

    tmp_accum.unlink(missing_ok=True)
    tmp_count.unlink(missing_ok=True)

    print("[BUILD] Loading into RAM...")
    arr = np.load(str(out_path))

    print(f"[BUILD] Done. shape={arr.shape}  RAM≈{arr.nbytes/1e9:.1f} GB")
    return arr, users_with_hist



def get_user_hist_emb(
    feature: str,
    emb_dir: Path,
    graph_dir: Path,
    data_dir: Path,
    force: bool = False,
):
    cache_npy = emb_dir / "user_hist_emb.npy"
    cache_meta = emb_dir / "user_hist_meta.json"

    if not force and cache_npy.exists() and cache_meta.exists():
        print(f"[USER_EMB] Cache found: {cache_npy}")

        arr = np.load(str(cache_npy))

        with open(cache_meta) as f:
            meta = json.load(f)

        print(
            f"[USER_EMB] Loaded from cache. "
            f"shape={arr.shape} "
            f"n_train={meta.get('n_train', '?'):,} "
            f"feature={meta.get('feature')}"
        )
        return np.ascontiguousarray(arr)

    print("[USER_EMB] Cache not found (or --force). Computing from scratch...")

    raw_feat, raw_item2idx = _load_embeddings(emb_dir)
    cust2idx, global_item2idx = _load_mappings(graph_dir)

    n_users = len(cust2idx)
    n_items = len(global_item2idx)

    item_feat = _align_item_feat(
        raw_feat,
        raw_item2idx,
        global_item2idx,
        n_items
    )

    del raw_feat, raw_item2idx
    gc.collect()

    train_u, train_i = _load_train_split(
        data_dir,
        cust2idx,
        global_item2idx
    )

    n_train = len(train_u)

    del cust2idx, global_item2idx
    gc.collect()

    arr, users_with_hist = compute_user_hist_emb(
        train_u,
        train_i,
        item_feat,
        n_users,
        cache_npy
    )

    del train_u, train_i, item_feat
    gc.collect()

    meta = {
        "feature": feature,
        "n_users": n_users,
        "dim": int(arr.shape[1]),
        "n_train": n_train,
        "users_with_hist": users_with_hist,
        "cache_path": str(cache_npy),
    }

    with open(cache_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[USER_EMB] Meta saved to {cache_meta}")

    return np.ascontiguousarray(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature",
        choices=["clip", "fashionclip"],
        required=True
    )
    parser.add_argument(
        "--force",
        action="store_true"
    )

    args = parser.parse_args()

    emb_dir = EMBEDDING_DIR / "hm" / args.feature
    graph_dir = GRAPH_DIR / "hm" / args.feature
    data_dir = PROCESSED_DIR / "hm"

    arr = get_user_hist_emb(
        feature=args.feature,
        emb_dir=emb_dir,
        graph_dir=graph_dir,
        data_dir=data_dir,
        force=args.force
    )

    print(f"\n✓ user_hist_emb ready: {arr.shape}  dtype={arr.dtype}")
    print(f"  Saved to: {emb_dir / 'user_hist_emb.npy'}")


if __name__ == "__main__":
    main()