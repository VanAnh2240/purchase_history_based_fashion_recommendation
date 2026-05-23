# File src/models/model_factory.py

from src.models.lightgcn import LightGCN
from src.models.graphsage import GraphSAGE
from src.models.ngcf import NGCF

from src.models.bpr import BPRMatrixFactorization
from src.models.siamese import SiameseProjector


def get_model(model_name, num_users=None, num_items=None, embedding_dim=64):
    model_name = model_name.lower()
    
    # GNN MODELS
    if model_name == "lightgcn":
        return LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim
        )

    elif model_name == "graphsage":
        return GraphSAGE(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim
        )

    elif model_name == "ngcf":
        return NGCF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim
        )

    # BASELINE MODELS
    elif model_name == "bpr":
        return BPRMatrixFactorization(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim
        )

    elif model_name == "siamese":
        return SiameseProjector(dim_in=512)
        
    else:
        raise ValueError(
            f"Model '{model_name}' không tồn tại. "
            "Chọn: lightgcn | graphsage | ngcf | bpr | siamese"
        )