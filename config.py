# File config.py

import torch
from pathlib import Path


# PATHS
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

EMBEDDING_DIR = ROOT_DIR / "embeddings"
GRAPH_DIR     = ROOT_DIR / "graphs"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULT_DIR     = ROOT_DIR / "results"


# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TRAIN CONFIG
SEED          = 42
BATCH_SIZE    = 16384
NUM_EPOCHS    = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 1e-4

# MODEL CONFIG
EMBEDDING_DIM = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
NUM_WORKERS   = 4


LOGIT_SCALE   = 2.0

# EVALUATION
THRESHOLD = 0.5
METRICS   = ["accuracy", "precision", "recall", "f1"]