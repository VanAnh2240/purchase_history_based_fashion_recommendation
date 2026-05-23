"""
src/graph/build_hm_graph.py
"""

import gc
import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

from config import PROCESSED_DIR, EMBEDDING_DIR, GRAPH_DIR

_CHUNK = 500_000


class HMGraphBuilder:
    def __init__(self, feature: str = "clip"):
        if feature not in ("clip", "fashionclip"):
            raise ValueError(f"Invalid feature: {feature}")
        self.feature       = feature
        self.data_dir      = PROCESSED_DIR / "hm"
        self.embedding_dir = EMBEDDING_DIR / "hm" / feature
        self.output_dir    = GRAPH_DIR / "hm" / feature
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self):
        print("\n[HMGraphBuilder] BUILD START v5")

        # Step 1: item2idx từ embeddings
        item2idx = self._load_item2idx()
        n_items  = len(item2idx)

        # Step 2: Pass 1 — collect unique customer_id strings (RAM ~300MB)
        cust2idx = self._build_cust2idx()
        n_users  = len(cust2idx)
        print(f"[CUST] unique customers: {n_users:,}")

        # Step 3: Pass 2 — map sang int, gom arrays
        u_arr, i_arr = self._scan_transactions(cust2idx, item2idx)

        # Step 4: dedup bằng pack int64
        u_arr, i_arr = self._dedup(u_arr, i_arr)
        print(f"[DEDUP] unique pairs: {len(u_arr):,}")

        # Step 5: build graph
        edge_index, adj = self._build_graph(u_arr, i_arr, n_users, n_items)

        # Step 6: save
        self._save(edge_index, adj, cust2idx, item2idx, n_users, n_items)

        del u_arr, i_arr
        gc.collect()

        print("\n[HMGraphBuilder] DONE ✓")
        print(f"  Users : {n_users:,}")
        print(f"  Items : {n_items:,}")
        print(f"  Edges : {edge_index.shape[1] // 2:,}")

    # ── step 1 ────────────────────────────────────────────────────────────────

    def _load_item2idx(self) -> dict:
        ids = (pd.read_csv(self.embedding_dir / "article_ids.csv")
               ["article_id"].astype(str).str.zfill(10).tolist())
        item2idx = {aid: idx for idx, aid in enumerate(ids)}
        print(f"[ALIGN] embedding items: {len(item2idx):,}")
        return item2idx

    # ── step 2: Pass 1 — build global customer map ────────────────────────────

    def _build_cust2idx(self) -> dict:
        """
        Đọc toàn bộ transactions.csv, collect unique customer_id strings.
        Chỉ đọc 1 cột → RAM ~300MB cho 1.36M unique customers.
        """
        print("[CUST] Pass 1: collecting unique customer_ids...")
        unique_custs = set()
        for chunk in tqdm(
            pd.read_csv(
                self.data_dir / "transactions.csv",
                usecols=["customer_id"],
                dtype=str,
                chunksize=_CHUNK,
            ),
            desc="Pass 1",
        ):
            unique_custs.update(chunk["customer_id"].tolist())

        # Sort để mapping deterministic (reproducible)
        sorted_custs = sorted(unique_custs)
        cust2idx = {c: i for i, c in enumerate(sorted_custs)}
        del unique_custs, sorted_custs
        gc.collect()
        return cust2idx

    # ── step 3: Pass 2 — map sang int arrays ──────────────────────────────────

    def _scan_transactions(self, cust2idx: dict,
                           item2idx: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Pass 2: dùng global cust2idx để map customer_id → int nhất quán.
        Không có per-chunk category codes → không bị loạn index.
        """
        print("[LOAD] Pass 2: mapping transactions to int arrays...")
        u_bufs, i_bufs = [], []
        dropped = 0
        total   = 0

        for chunk in tqdm(
            pd.read_csv(
                self.data_dir / "transactions.csv",
                usecols=["customer_id", "article_id"],
                dtype=str,
                chunksize=_CHUNK,
            ),
            desc="Pass 2",
        ):
            total += len(chunk)
            chunk["article_id"] = chunk["article_id"].str.zfill(10)

            i_mapped = chunk["article_id"].map(item2idx)
            mask     = i_mapped.notna()
            dropped += (~mask).sum()

            chunk_ok = chunk[mask]
            i_int    = i_mapped[mask].values.astype(np.int32)
            u_int    = chunk_ok["customer_id"].map(cust2idx).values.astype(np.int32)

            u_bufs.append(u_int)
            i_bufs.append(i_int)

        print(f"[LOAD] total rows: {total:,}  dropped: {dropped:,}")

        u_all = np.concatenate(u_bufs)
        i_all = np.concatenate(i_bufs)
        del u_bufs, i_bufs
        gc.collect()
        return u_all, i_all

    # ── step 4 ────────────────────────────────────────────────────────────────

    def _dedup(self, u_arr: np.ndarray,
               i_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        print("[DEDUP] Packing and deduplicating...")
        packed = (u_arr.astype(np.int64) << 32) | i_arr.astype(np.int64)
        del u_arr, i_arr
        gc.collect()

        unique = np.unique(packed)
        del packed
        gc.collect()

        u = (unique >> 32).astype(np.int32)
        i = (unique & 0xFFFFFFFF).astype(np.int32)
        del unique
        gc.collect()
        return u, i

    # ── step 5 ────────────────────────────────────────────────────────────────

    def _build_graph(self, u_arr: np.ndarray, i_arr: np.ndarray,
                     n_users: int, n_items: int):
        i_offset = i_arr.astype(np.int64) + n_users
        u_long   = u_arr.astype(np.int64)

        src = np.concatenate([u_long, i_offset])
        dst = np.concatenate([i_offset, u_long])
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

        del src, dst, i_offset, u_long
        gc.collect()

        print(f"[GRAPH] edge_index  : {edge_index.shape}")
        print(f"[GRAPH] total nodes : {n_users + n_items:,}")

        data = np.ones(len(u_arr), dtype=np.float32)
        adj  = sp.csr_matrix(
            (data, (u_arr.astype(np.int64), i_arr.astype(np.int64))),
            shape=(n_users, n_items),
        )
        print(f"[GRAPH] adj         : {adj.shape}  nnz={adj.nnz:,}")
        return edge_index, adj

    # ── step 6 ────────────────────────────────────────────────────────────────

    def _save(self, edge_index, adj, cust2idx: dict, item2idx: dict,
              n_users: int, n_items: int):
        out = self.output_dir

        torch.save(edge_index, out / "edge_index.pt")
        print("[SAVE] edge_index.pt")

        with open(out / "adjacency_matrix.pkl", "wb") as f:
            pickle.dump(adj, f)
        print("[SAVE] adjacency_matrix.pkl")

        # user2idx.csv: customer_id string thật → user_idx
        # Siamese và GNN trainer đều dùng file này để map nhất quán
        pd.DataFrame(
            list(cust2idx.items()), columns=["customer_id", "user_idx"]
        ).to_csv(out / "user2idx.csv", index=False)
        print(f"[SAVE] user2idx.csv  ({n_users:,} users)")

        pd.DataFrame(
            list(item2idx.items()), columns=["article_id", "item_idx"]
        ).to_csv(out / "item2idx.csv", index=False)
        print(f"[SAVE] item2idx.csv  ({n_items:,} items)")

        meta = {
            "n_users": n_users,
            "n_items": n_items,
            "feature": self.feature,
        }
        with open(out / "graph_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[SAVE] graph_meta.json → {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default="clip", choices=["clip", "fashionclip"])
    args = parser.parse_args()
    HMGraphBuilder(args.feature).build()