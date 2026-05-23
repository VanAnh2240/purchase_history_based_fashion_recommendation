import argparse

from src.graph.build_hm_graph import HMGraphBuilder


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="hm",
        choices=["hm"],
        help="Dataset name"
    )

    parser.add_argument(
        "--feature",
        type=str,
        default="clip",
        choices=["clip", "fashionclip"],
        help="Feature extractor type"
    )

    args = parser.parse_args()

    print("\n==============================")
    print("GRAPH BUILDING...")
    print(f"Dataset : {args.dataset}")
    print(f"Feature : {args.feature}")
    print("==============================")

    if args.dataset != "hm":
        raise ValueError("Currently only HM dataset is supported")

    builder = HMGraphBuilder(feature=args.feature)
    builder.build()

    print("\nGRAPH BUILD DONE")


if __name__ == "__main__":
    main()