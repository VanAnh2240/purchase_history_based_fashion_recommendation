import argparse

from config import RAW_DIR, PROCESSED_DIR
from src.data.hm.hm_preprocess import HMPreprocess

def run_hm():
    input_dir = RAW_DIR / "hm"
    output_dir = PROCESSED_DIR / "hm"

    print("[HM] input :", input_dir)
    print("[HM] output:", output_dir)

    pre = HMPreprocess(str(input_dir), str(output_dir))
    pre.run()


def run_polyvore():
    input_dir = RAW_DIR / "polyvore"
    output_dir = PROCESSED_DIR / "polyvore"

    print("[POLYVORE] input :", input_dir)
    print("[POLYVORE] output:", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hm", "polyvore"]
    )

    args = parser.parse_args()

    if args.dataset == "hm":
        run_hm()
    else:
        run_polyvore()


if __name__ == "__main__":
    main()