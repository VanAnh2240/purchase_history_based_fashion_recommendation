import os
import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.models.ngcf import NGCF

CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    "checkpoints/hm/fashionclip/ngcf/fashionclip_ngcf.pth",
)
ARTICLE_IDS_CSV = os.getenv(
    "ARTICLE_IDS_CSV",
    "embeddings/hm/fashionclip/article_ids.csv",
)
GRAPH_EDGE_PATH = os.getenv(
    "GRAPH_EDGE_PATH",
    "graphs/hm/fashionclip/edge_index.pt",
)
EMB_CACHE_PATH = os.getenv(
    "EMB_CACHE_PATH",
    "data/demo/emb_cache.npz",
)

app = FastAPI(
    title="Purchase History-Based Fashion Recommendation API",
    version="1.0.0",
)

users_emb = None
items_emb = None
article_ids = []
article_idx = {}


@app.on_event("startup")
def load_model():
    global users_emb, items_emb, article_ids, article_idx

    df = pd.read_csv(ARTICLE_IDS_CSV, dtype={"article_id": str})
    df["article_id"] = df["article_id"].str.zfill(10)

    article_ids = df["article_id"].tolist()
    article_idx = {aid: i for i, aid in enumerate(article_ids)}

    if os.path.exists(EMB_CACHE_PATH):
        cache = np.load(EMB_CACHE_PATH)
        users_emb = cache["users_emb"]
        items_emb = cache["items_emb"]
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]

    num_users = state_dict["user_embedding.weight"].shape[0]
    num_items = state_dict["item_embedding.weight"].shape[0]
    embedding_dim = state_dict["user_embedding.weight"].shape[1]

    model = NGCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=3,
        dropout=0.0,
    )

    model.load_state_dict(state_dict)
    model.eval()

    edge_index = torch.load(GRAPH_EDGE_PATH, map_location="cpu", weights_only=False)
    model.precompute_norm_adj(edge_index=edge_index, n_nodes=num_users + num_items)

    with torch.no_grad():
        users, items = model(use_cache=True)

    users_emb = users.numpy().astype(np.float32)
    items_emb = items.numpy().astype(np.float32)

    os.makedirs(os.path.dirname(EMB_CACHE_PATH) or ".", exist_ok=True)
    np.savez(
        EMB_CACHE_PATH,
        users_emb=users_emb,
        items_emb=items_emb,
    )


def validate_customer(customer_id: int):
    if users_emb is None:
        raise HTTPException(503, "Model not loaded")

    if customer_id < 0 or customer_id >= len(users_emb):
        raise HTTPException(404, "Invalid customer_id")


@app.get("/recommend/{customer_id}")
def recommend(customer_id: int, top_k: int = 100):
    validate_customer(customer_id)

    scores = items_emb @ users_emb[customer_id]

    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    return {
        "customer_id": customer_id,
        "top_k": top_k,
        "items": [
            {
                "product_id": article_ids[i],
                "score": round(float(scores[i]), 6),
            }
            for i in top_idx
        ],
    }


class FilterRequest(BaseModel):
    article_ids: List[str]
    top_k: int = 8


@app.post("/recommend/{customer_id}/filter")
def recommend_filter(customer_id: int, req: FilterRequest):
    validate_customer(customer_id)

    indices = [article_idx[i] for i in req.article_ids if i in article_idx]

    if not indices:
        raise HTTPException(404, "No matching article_id found")

    idx = np.array(indices)
    scores = items_emb[idx] @ users_emb[customer_id]

    top_k = min(req.top_k, len(idx))
    top = np.argsort(scores)[::-1][:top_k]

    return {
        "customer_id": customer_id,
        "top_k": top_k,
        "items": [
            {
                "product_id": article_ids[idx[i]],
                "score": round(float(scores[i]), 6),
            }
            for i in top
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": users_emb is not None,
    }


@app.delete("/cache")
def clear_cache():
    if os.path.exists(EMB_CACHE_PATH):
        os.remove(EMB_CACHE_PATH)
        return {"deleted": True}

    return {"deleted": False}