"""
generate_sql_product.py
"""

import re
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

ARTICLES_CSV    = "data/demo/articles_demo.csv"
ARTICLE_IDS_CSV = "data/demo/article_ids_demo.csv"
OUTPUT_SQL      = "hm_product_import.sql"

IMAGE_BASE_URL  = "https://14.225.207.37/hm-images"

HM_CATEGORY_PREFIX = "HM"
HM_CATSUB_PREFIX   = "HM"
HM_COLOR_OFFSET    = 90000
EMBEDDING_MODEL    = "fashionclip_ngcf"

BATCH_SIZE = 200


def esc(val) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, float) and pd.isna(val):
        return "NULL"
    s = str(val)
    # Escape single quote và backslash
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


def write_batches(f, table: str, columns: list, rows: list):
    cols = ", ".join(columns)
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        values_str = ",\n  ".join(batch)
        f.write(f"INSERT IGNORE INTO `{table}` ({cols}) VALUES\n  {values_str};\n\n")


def main():
    print("=" * 60)
    print("AuraFit — Generate SQL from H&M CSV (offline)")
    print("=" * 60)

    # ── Load CSV ──────────────────────────────────────────────
    print("\nĐọc CSV ...")
    if not Path(ARTICLES_CSV).exists():
        print(f"Không tìm thấy: {ARTICLES_CSV}")
        sys.exit(1)
    if not Path(ARTICLE_IDS_CSV).exists():
        print(f"Không tìm thấy: {ARTICLE_IDS_CSV}")
        sys.exit(1)

    articles = pd.read_csv(ARTICLES_CSV, dtype={"article_id": str})
    articles["article_id"] = articles["article_id"].str.zfill(10)

    article_ids_df = pd.read_csv(ARTICLE_IDS_CSV, dtype={"article_id": str})
    article_ids_df["article_id"] = article_ids_df["article_id"].str.zfill(10)

    print(f"  articles: {len(articles):,} | article_ids: {len(article_ids_df):,}")

    cat_map    = {}   
    catsub_map = {}  
    color_map  = {}  
    
    groups = sorted(articles["index_group_name"].dropna().unique())
    for i, group in enumerate(groups, start=1):
        cat_map[group] = f"{HM_CATEGORY_PREFIX}{i:02d}"

    sections = (
        articles[["index_group_name", "section_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["index_group_name", "section_name"])
    )
    sub_counter = 1
    for _, row in sections.iterrows():
        key = (row["index_group_name"], row["section_name"])
        catsub_map[key] = f"{HM_CATSUB_PREFIX}{sub_counter:03d}"
        sub_counter += 1

    colors = sorted(articles["colour_group_name"].dropna().unique())
    for i, cname in enumerate(colors):
        color_map[cname] = HM_COLOR_OFFSET + i

    print(f"\nXuất ra {OUTPUT_SQL} ...")
    with open(OUTPUT_SQL, "w", encoding="utf-8") as f:

        # Header
        f.write(f"-- AuraFit H&M Import\n")
        f.write(f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"-- articles: {len(articles):,} | colors: {len(color_map)} | categories: {len(cat_map)}\n\n")
        f.write("SET NAMES utf8mb4;\n")
        f.write("SET foreign_key_checks = 0;\n\n")

        # ── 1. category ───────────────────────────────────────
        print("  [1/6] category ...")
        f.write("-- =============================================\n")
        f.write("-- 1. category\n")
        f.write("-- =============================================\n")
        cat_rows = []
        for group, cat_id in cat_map.items():
            cat_rows.append(f"({esc(cat_id)}, {esc(group[:100])})")
        write_batches(f, "category", ["`category_id`", "`category_name`"], cat_rows)

        # ── 2. category_sub ───────────────────────────────────
        print("  [2/6] category_sub ...")
        f.write("-- =============================================\n")
        f.write("-- 2. category_sub\n")
        f.write("-- =============================================\n")
        sub_rows = []
        for (group, section), sub_id in catsub_map.items():
            cat_id = cat_map.get(group, "")
            sub_rows.append(
                f"({esc(sub_id)}, {esc(cat_id)}, {esc(section[:100])})"
            )
        write_batches(
            f, "category_sub",
            ["`category_sub_id`", "`category_id`", "`category_sub_name`"],
            sub_rows,
        )

        # ── 3. color ──────────────────────────────────────────
        print("  [3/6] color ...")
        f.write("-- =============================================\n")
        f.write("-- 3. color\n")
        f.write("-- =============================================\n")
        color_rows = []
        for cname, code in color_map.items():
            type_en = "Solid" if "solid" in cname.lower() else "Pattern"
            color_rows.append(
                f"({code}, {esc(type_en)}, {esc(cname[:100])}, {esc(type_en)}, {esc(cname[:100])})"
            )
        write_batches(
            f, "color",
            ["`color_code`", "`type_en`", "`color_name_en`", "`type_vi`", "`color_name_vi`"],
            color_rows,
        )

        # ── 4. product ────────────────────────────────────────
        print("  [4/6] product ...")
        f.write("-- =============================================\n")
        f.write("-- 4. product\n")
        f.write("-- =============================================\n")
        product_rows = []
        imported_ids = set()
        skipped = 0

        for _, art in tqdm(articles.iterrows(), total=len(articles), desc="  product"):
            article_id = str(art["article_id"]).zfill(10)
            group      = art.get("index_group_name", "")
            section    = art.get("section_name", "")
            cat_sub_id = catsub_map.get((group, section))

            if not cat_sub_id:
                skipped += 1
                continue

            name        = str(art.get("prod_name", ""))[:255]
            description = str(art.get("detail_desc", ""))
            composition = str(art.get("garment_group_name", ""))

            product_rows.append(
                f"({esc(article_id)}, {esc(cat_sub_id)}, {esc(name)}, 0, "
                f"{esc(description)}, {esc(composition)}, 1, NULL, NULL)"
            )
            imported_ids.add(article_id)

        write_batches(
            f, "product",
            ["`product_id`", "`category_sub_id`", "`name`", "`price`",
             "`description`", "`composition`", "`is_selling`",
             "`personal_color`", "`body_shape`"],
            product_rows,
        )
        print(f"    ✓ {len(product_rows):,} sản phẩm | bỏ qua: {skipped}")

        # ── 5. product_color ──────────────────────────────────
        print("  [5/6] product_color ...")
        f.write("-- =============================================\n")
        f.write("-- 5. product_color\n")
        f.write("-- =============================================\n")
        pc_rows = []
        for _, art in tqdm(articles.iterrows(), total=len(articles), desc="  product_color"):
            article_id = str(art["article_id"]).zfill(10)
            if article_id not in imported_ids:
                continue
            color_name = art.get("colour_group_name", "")
            color_code = color_map.get(color_name)
            if color_code is None:
                continue
            image_id = article_id[:8]
            pc_rows.append(f"({esc(article_id)}, {color_code}, {esc(image_id)})")

        write_batches(
            f, "product_color",
            ["`product_id`", "`color_code`", "`image_id`"],
            pc_rows,
        )

        # ── 6. product_image ──────────────────────────────────
        print("  [6/6] product_image ...")
        f.write("-- =============================================\n")
        f.write("-- 6. product_image\n")
        f.write("-- =============================================\n")
        pi_rows = []
        for _, art in tqdm(articles.iterrows(), total=len(articles), desc="  product_image"):
            article_id = str(art["article_id"]).zfill(10)
            if article_id not in imported_ids:
                continue
            color_name = art.get("colour_group_name", "")
            color_code = color_map.get(color_name)
            if color_code is None:
                continue
            prefix     = article_id[:3]
            image_link = f"{IMAGE_BASE_URL}/{prefix}/{article_id}.jpg"
            pi_rows.append(
                f"({esc(article_id)}, {esc(image_link)}, {color_code}, {esc(article_id)})"
            )

        write_batches(
            f, "product_image",
            ["`product_id`", "`image_link`", "`color_code`", "`image_code`"],
            pi_rows,
        )

        # ── 7. hm_article_map ────────────────────────────────
        f.write("-- =============================================\n")
        f.write("-- 7. hm_article_map\n")
        f.write("-- =============================================\n")
        map_rows = []
        for idx, row in article_ids_df.iterrows():
            article_id = str(row["article_id"]).zfill(10)
            if article_id not in imported_ids:
                continue
            map_rows.append(
                f"({esc(article_id)}, {esc(article_id)}, {int(idx)}, {esc(EMBEDDING_MODEL)})"
            )

        write_batches(
            f, "hm_article_map",
            ["`article_id`", "`product_id`", "`hm_internal_idx`", "`embedding_model`"],
            map_rows,
        )

        f.write("SET foreign_key_checks = 1;\n")
        f.write("-- =============================================\n")
        f.write(f"-- Import hoàn tất: {len(imported_ids):,} sản phẩm\n")
        f.write("-- =============================================\n")

    size_mb = Path(OUTPUT_SQL).stat().st_size / 1024 / 1024
    print(f"\n✓ Xong! File: {OUTPUT_SQL} ({size_mb:.1f} MB)")
    print(f"  {len(cat_map)} categories | {len(catsub_map)} sub-categories")
    print(f"  {len(color_map)} colors | {len(imported_ids):,} products")
    print()

if __name__ == "__main__":
    main()