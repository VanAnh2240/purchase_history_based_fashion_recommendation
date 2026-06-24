"""
train.py 
"""

import argparse
import sys
import torch

from src.evaluation.evaluator import Evaluator

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


def parse_args():
    parser = argparse.ArgumentParser(description="Fashion Recommendation — Trainer")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hm", "polyvore"]
    )

    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        choices=["clip", "fashionclip"]
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lightgcn", "graphsage", "ngcf", "bpr", "siamese"]
    )

    parser.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["item", "user"]
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "hm":

        if args.model in ("lightgcn", "graphsage", "ngcf", "siamese") and args.feature is None:
            print(f"[ERROR] --feature là bắt buộc với model={args.model}")
            sys.exit(1)

        evaluator = Evaluator()

        if args.model == "lightgcn":
            from src.training.hm.train_lightgcn import HMGNNTrainer
            trainer = HMGNNTrainer(
                model_name=args.model,
                feature=args.feature
            )
        elif args.model == "graphsage":
            from src.training.hm.train_graphsage import HMGNNTrainer
            trainer = HMGNNTrainer(
                model_name=args.model,
                feature=args.feature
            )
            
        elif args.model == "ngcf":
            from src.training.hm.train_ngcf import HMNGCFTrainer
            trainer = HMNGCFTrainer(
                feature=args.feature
            )

        elif args.model == "bpr":
            from src.training.hm.train_bpr import HMBPRTrainer
            trainer = HMBPRTrainer()

        elif args.model == "siamese":
            from src.training.hm.train_siamese import HMSiameseTrainer

            trainer = HMSiameseTrainer(feature=args.feature)

        trainer.train(evaluator=evaluator)

    elif args.dataset == "polyvore":
        print("[INFO] Polyvore training chưa implement.")
        sys.exit(0)


if __name__ == "__main__":
    main()