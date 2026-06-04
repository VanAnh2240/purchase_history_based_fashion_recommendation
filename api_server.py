"""
api_server.py
"""

import os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.models.ngcf import NGCF

# ── Cấu hình ──────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/hm/fashionclip/ngcf/fashionclip_ngcf.pth")
ARTICLE_IDS_CSV = os.getenv("ARTICLE_IDS_CSV", "embeddings/hm/fashionclip/article_ids.csv")
GRAPH_EDGE_PATH = os.getenv("GRAPH_EDGE_PATH", "graphs/hm/fashionclip/edge_index.pt")
EMB_CACHE_PATH  = os.getenv("EMB_CACHE_PATH",  "data/demo/emb_cache.npz")
TOP_K_DEFAULT   = 100

# ── Singleton state ────────────────────────────────────────────────────────────
class _State:
    users_emb   : np.ndarray = None  
    items_emb   : np.ndarray = None
    article_ids : list       = []
    article_idx : dict       = {}    # article_id → row index (lookup O(1))

S = _State()
app = FastAPI(title="AuraFit Recommendation API", version="1.0.0")


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_model():
    import pandas as pd

    df = pd.read_csv(ARTICLE_IDS_CSV, dtype={"article_id": str})
    df["article_id"] = df["article_id"].str.zfill(10)
    S.article_ids = df["article_id"].tolist()
    S.article_idx = {aid: i for i, aid in enumerate(S.article_ids)}  # ← thêm dòng này

    if os.path.exists(EMB_CACHE_PATH):
        print(f"[startup] Loading embeddings cache từ {EMB_CACHE_PATH} ...")
        cache = np.load(EMB_CACHE_PATH)
        S.users_emb = cache["users_emb"]
        S.items_emb = cache["items_emb"]
        print(f"[startup] ✓ users={S.users_emb.shape[0]:,}  items={S.items_emb.shape[0]:,}  (from cache)")
        return

    print(f"[startup] Cache chưa có → chạy NGCF propagation ...")

    ckpt       = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    num_users     = state_dict["user_embedding.weight"].shape[0]
    num_items     = state_dict["item_embedding.weight"].shape[1]
    embedding_dim = state_dict["user_embedding.weight"].shape[1]

    torch.set_num_threads(os.cpu_count())

    model = NGCF(
        num_users     = num_users,
        num_items     = num_items,
        embedding_dim = embedding_dim,
        num_layers    = 3,
        dropout       = 0.0,
    )
    model.load_state_dict(state_dict)
    model.eval()

    edge_index = torch.load(GRAPH_EDGE_PATH, map_location="cpu", weights_only=False)
    model.precompute_norm_adj(edge_index=edge_index, n_nodes=num_users + num_items)

    print(f"[startup] Đang tính embeddings ({num_users:,} users × {num_items:,} items) ...")
    with torch.no_grad():
        users_emb, items_emb = model(use_cache=True)

    S.users_emb = users_emb.numpy().astype(np.float32)
    S.items_emb = items_emb.numpy().astype(np.float32)

    os.makedirs(os.path.dirname(EMB_CACHE_PATH) or ".", exist_ok=True)
    np.savez(EMB_CACHE_PATH, users_emb=S.users_emb, items_emb=S.items_emb)
    print(f"[startup] ✓ Đã lưu cache → {EMB_CACHE_PATH}")
    print(f"[startup] ✓ users={num_users:,}  items={num_items:,}  dim={embedding_dim}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/recommend/{customer_id}")
def recommend(customer_id: int, top_k: int = TOP_K_DEFAULT):
    if S.users_emb is None:
        raise HTTPException(503, "Model chưa sẵn sàng")

    num_users = S.users_emb.shape[0]
    if customer_id < 0 or customer_id >= num_users:
        raise HTTPException(404, f"customer_id={customer_id} không hợp lệ (model có {num_users} users, index 0–{num_users-1})")

    user_vec = S.users_emb[customer_id]
    scores   = S.items_emb @ user_vec

    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return {
        "customer_id": customer_id,
        "top_k"      : top_k,
        "items"      : [
            {"product_id": S.article_ids[i], "score": round(float(scores[i]), 6)}
            for i in top_indices
        ],
    }


class FilterRequest(BaseModel):
    article_ids: List[str]
    top_k: int = 8

@app.post("/recommend/{customer_id}/filter")
def recommend_filter(customer_id: int, req: FilterRequest):
    """Chỉ score các article_id được truyền vào — tránh trả payload 105K items."""
    if S.users_emb is None:
        raise HTTPException(503, "Model chưa sẵn sàng")

    num_users = S.users_emb.shape[0]
    if customer_id < 0 or customer_id >= num_users:
        raise HTTPException(404, f"customer_id={customer_id} không hợp lệ")

    # Chỉ lấy indices của các article_id có trong model
    indices = [S.article_idx[aid] for aid in req.article_ids if aid in S.article_idx]
    if not indices:
        raise HTTPException(404, "Không có article_id nào khớp với model")

    user_vec     = S.users_emb[customer_id]
    idx_arr      = np.array(indices, dtype=np.int32)
    subset_emb   = S.items_emb[idx_arr]          # (N_db, D)
    subset_scores = subset_emb @ user_vec         # (N_db,)

    top_k = min(req.top_k, len(indices))
    top_pos = np.argsort(subset_scores)[::-1][:top_k]

    return {
        "customer_id": customer_id,
        "top_k"      : top_k,
        "items"      : [
            {
                "product_id": S.article_ids[idx_arr[p]],
                "score"     : round(float(subset_scores[p]), 6),
            }
            for p in top_pos
        ],
    }


@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model_loaded": S.users_emb is not None,
        "items_count" : len(S.article_ids),
        "users_count" : int(S.users_emb.shape[0]) if S.users_emb is not None else 0,
        "cache_exists": os.path.exists(EMB_CACHE_PATH),
    }


@app.delete("/cache")
def clear_cache():
    if os.path.exists(EMB_CACHE_PATH):
        os.remove(EMB_CACHE_PATH)
        return {"deleted": True, "path": EMB_CACHE_PATH}
    return {"deleted": False, "path": EMB_CACHE_PATH}