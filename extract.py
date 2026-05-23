# File: extract.py

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hm", "polyvore"],
    )

    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        choices=["clip", "fashionclip"],
    )

    args = parser.parse_args()

    # H&M
    if args.dataset == "hm":

        if args.feature == "clip":
            from src.feature_extraction.hm.hm_clip_extract import HMClipExtractor
            extractor = HMClipExtractor()

        elif args.feature == "fashionclip":
            from src.feature_extraction.hm.hm_fashionclip_extract import HMFashionCLIPExtractor
            extractor = HMFashionCLIPExtractor()

    # # POLYVORE
    # elif args.dataset == "polyvore":

    #     if args.feature == "clip":
    #         from src.feature_extraction.polyvore.extract_clip import PolyvoreCLIPExtractor
    #         extractor = PolyvoreClipExtractor()

    #     elif args.feature == "fashionclip":
    #         from src.feature_extraction.polyvore.extract_fashionclip import PolyvoreFashionClipExtractor
    #         extractor = PolyvoreFashionCLIPExtractor()

    # print(f"\nStart extracting: {args.dataset} + {args.feature}\n")

    extractor.extract()

    print("\nDone!")


if __name__ == "__main__":
    main()