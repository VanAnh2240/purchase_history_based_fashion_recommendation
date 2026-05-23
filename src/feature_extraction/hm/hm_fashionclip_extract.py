# file src/feature_extraction/hm/extract_clip.py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from config import DEVICE, PROCESSED_DIR, EMBEDDING_DIR


class HMFashionCLIPExtractor:
    def __init__(self):

        self.device = DEVICE
        print(f"Using device: {self.device}")

        self.model = CLIPModel.from_pretrained(
             "patrickjohncyh/fashion-clip"
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
             "patrickjohncyh/fashion-clip"
        )

        self.model.eval()

        self.data_dir = PROCESSED_DIR / "hm"
        self.image_dir = self.data_dir / "images"

        self.output_dir = EMBEDDING_DIR / "hm" / "fashionclip"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_text(self, row):
        parts = [
            str(row.get("prod_name", "")),
            str(row.get("detail_desc", "")),
            str(row.get("product_type_name", "")),
            str(row.get("graphical_appearance_name", "")),
            str(row.get("colour_group_name", "")),
        ]
        return " ".join(parts)[:200]

    def load_valid_images(self, df):
        valid_map = {}
    
        for _, row in df.iterrows():   # FIX HERE
            article_id = str(row["article_id"]).zfill(10)
    
            img_path = (
                self.image_dir /
                article_id[:3] /
                f"{article_id}.jpg"
            )
    
            if img_path.exists():
                valid_map[article_id] = (img_path, row)
    
        print(f"[INFO] Valid images found: {len(valid_map):,}")
        return valid_map

    def extract(self):

        csv_path = self.data_dir / "articles.csv"
        df = pd.read_csv(csv_path)

        print(f"Total articles: {len(df):,}")

        valid_map = self.load_valid_images(df)

        embeddings = []
        article_ids = []

        # =========================
        # LOOP ONLY VALID DATA
        # =========================
        for article_id, (img_path, row) in tqdm(valid_map.items()):

            try:
                image = Image.open(img_path).convert("RGB")
                text = self.build_text(row)

                inputs = self.processor(
                    text=[text],
                    images=image,
                    return_tensors="pt",
                    padding=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                    emb = (outputs.image_embeds + outputs.text_embeds) / 2
                    emb = emb.cpu().numpy().squeeze()

                embeddings.append(emb)
                article_ids.append(article_id)

            except Exception as e:
                print(f"Skip {article_id}: {e}")

        embeddings = np.array(embeddings)

        np.save(self.output_dir / "embeddings.npy", embeddings)

        pd.DataFrame({
            "article_id": article_ids
        }).to_csv(self.output_dir / "article_ids.csv", index=False)

        print("\nDONE")
        print("embeddings shape:", embeddings.shape)
        print("saved:", self.output_dir)


if __name__ == "__main__":
    extractor = HMClipExtractor()
    extractor.extract()
