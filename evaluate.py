"""
evaluate.py  (root)

Cách dùng:
  # Evaluate 1 model cụ thể:
  python evaluate.py --dataset hm --feature clip         --model lightgcn
  python evaluate.py --dataset hm --feature clip         --model graphsage
  python evaluate.py --dataset hm --feature clip         --model ngcf
  python evaluate.py --dataset hm --feature fashionclip  --model lightgcn
  python evaluate.py --dataset hm                        --model bpr
  python evaluate.py --dataset hm --feature clip         --model siamese

  # Evaluate TẤT CẢ model có checkpoint + tạo summary so sánh:
  python evaluate.py --dataset hm
"""

import argparse
from src.evaluation.runner import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name, e.g. hm")
    parser.add_argument("--feature", default=None,
                        help="Feature type: clip | fashionclip  "
                             "(bắt buộc với GNN / siamese)")
    parser.add_argument("--model",   default=None,
                        help="Model: lightgcn | graphsage | ngcf | bpr | siamese  "
                             "(bỏ qua để evaluate ALL)")
    args = parser.parse_args()

    run_evaluation(
        dataset    = args.dataset,
        feature    = args.feature,
        model_name = args.model,     # None → evaluate ALL
    )


if __name__ == "__main__":
    main()