"""
filter_demo_data.py
"""
import shutil, os
import numpy as np
import pandas as pd

# ── Cấu hình ──────────────────────────────────────────────────────────────────
TEST_CSV        = "data/processed/hm/test.csv"
ARTICLES_CSV    = "data/processed/hm/articles.csv"
CUSTOMERS_CSV   = "data/processed/hm/customers.csv"
EMBEDDINGS_NPY  = "embeddings/hm/fashionclip/embeddings.npy"
ARTICLE_IDS_CSV = "embeddings/hm/fashionclip/article_ids.csv"
IMAGE_SOURCE_DIR = "data/processed/hm/images_test"
OUT_DIR         = "data/demo"
IMAGE_OUT_DIR   = f"{OUT_DIR}/images"

N_USERS       = 20
AGE_MIN       = 18
AGE_MAX       = 35
MIN_PURCHASES = 5
TOP_K_ITEMS   = 50
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Lọc sản phẩm─────────────────────────────────────────────────────────
print("[1/6] Lọc sản phẩm...")

articles = pd.read_csv(ARTICLES_CSV, dtype={"article_id": str})
articles["article_id"] = articles["article_id"].str.zfill(10)

ladieswear = articles[
    (articles["index_name"] == "Ladieswear")
    &
    (
        articles["product_group_name"].isin([
            "Garment Upper body",
            "Garment Lower body",
            "Garment Full body"
        ])
    )
].copy()
ladieswear_ids = set(ladieswear["article_id"].unique())
print("\nProduct groups sau lọc:")
print(
    ladieswear["product_group_name"]
    .value_counts()
    .sort_index()
)
print(f"  Tổng: {len(articles):,} → Ladieswear: {len(ladieswear):,}")

# ── 2. Lọc khách hàng theo tuổi ───────────────────────────────────────────────
print(f"\n[2/6] Lọc khách hàng tuổi {AGE_MIN}-{AGE_MAX} ...")

customers = pd.read_csv(CUSTOMERS_CSV, dtype={"customer_id": str})
qualified_customers_set = set(
    customers[(customers["age"] >= AGE_MIN) & (customers["age"] <= AGE_MAX)]["customer_id"]
)
print(f"  Tổng: {len(customers):,} → Đủ tuổi: {len(qualified_customers_set):,}")

# ── 3. Lọc test.csv ───────────────────────────────────────────────────────────
print(f"\n[3/6] Lọc test.csv ...")

test = pd.read_csv(TEST_CSV, dtype={"article_id": str, "customer_id": str})
test["article_id"] = test["article_id"].str.zfill(10)

test_filtered = test[
    test["customer_id"].isin(qualified_customers_set) &
    test["article_id"].isin(ladieswear_ids)
].copy()

print(f"  Giao dịch: {len(test):,} → {len(test_filtered):,}")
print(f"  User còn lại: {test_filtered['customer_id'].nunique():,}")

# ── 4. Chọn 20 user đại diện ───────────────────────────────────────────────
print(f"\n[4/6] Chọn {N_USERS} user đại diện ...")

user_counts = (
    test_filtered
    .groupby("customer_id")["article_id"]
    .count()
    .rename("purchases")
)

eligible_users = user_counts[
    user_counts >= MIN_PURCHASES
].sort_values(ascending=False)

# Lấy spread đều để đa dạng
step = max(1, len(eligible_users) // N_USERS)
selected_users = eligible_users.iloc[::step].head(N_USERS).index.tolist()

print(
    f"  User đủ điều kiện: {len(eligible_users):,} "
    f"→ Chọn {len(selected_users)}"
)

for uid in selected_users[:5]:
    age_val = customers[
        customers["customer_id"] == uid
    ]["age"].values

    age_str = (
        f"{int(age_val[0])}"
        if len(age_val) > 0
        else "?"
    )

    print(
        f"    {uid[:12]}... | tuổi {age_str} "
        f"| {int(eligible_users.loc[uid])} giao dịch"
    )

print(f"    ... (và {len(selected_users)-5} user khác)")

# ── 5. Tạo demo dataset ────────────────────────────────────────────────────────
print(f"\n[5/6] Tạo demo dataset ...")

# Giao dịch
demo_test = (
    test_filtered[test_filtered["customer_id"].isin(selected_users)]
    .groupby("customer_id").head(TOP_K_ITEMS)
    .reset_index(drop=True)
)

# Dùng chính xác set article_id có trong demo_test
demo_item_ids = set(demo_test["article_id"].astype(str).str.zfill(10).unique())

# Articles — chỉ lấy những ID có trong demo_test (đảm bảo khớp số)
cols_keep = [c for c in [
    "article_id", "prod_name", "product_type_name", "product_group_name",
    "graphical_appearance_name", "colour_group_name", "index_name",
    "index_group_name", "section_name", "garment_group_name", "detail_desc",
] if c in articles.columns]
demo_articles = articles[articles["article_id"].isin(demo_item_ids)][cols_keep].copy()

# Embeddings
all_embeddings  = np.load(EMBEDDINGS_NPY)
all_article_ids = pd.read_csv(ARTICLE_IDS_CSV, dtype={"article_id": str})
all_article_ids["article_id"] = all_article_ids["article_id"].str.zfill(10)

mask = all_article_ids["article_id"].isin(demo_item_ids)
demo_indices     = all_article_ids[mask].index.tolist()
demo_embeddings  = all_embeddings[demo_indices]
demo_article_ids = all_article_ids[mask].reset_index(drop=True)

# Lưu files
demo_test.to_csv(f"{OUT_DIR}/test_20users.csv", index=False)
demo_articles.to_csv(f"{OUT_DIR}/articles_demo.csv", index=False)
demo_article_ids.to_csv(f"{OUT_DIR}/article_ids_demo.csv", index=False)
np.save(f"{OUT_DIR}/embeddings_demo.npy", demo_embeddings)
pd.DataFrame({"customer_id": selected_users}).to_csv(f"{OUT_DIR}/user_list.csv", index=False)

print(f"  Giao dịch: {len(demo_test):,} | Sản phẩm unique: {len(demo_item_ids):,}")
print(f"  Articles saved: {len(demo_articles):,} | Embeddings: {demo_embeddings.shape}")

# Copy ảnh
os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
copied_images, missing_images = 0, []

for article_id in demo_item_ids:
    article_id = str(article_id).zfill(10)
    prefix = article_id[:3]
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        src = os.path.join(IMAGE_SOURCE_DIR, prefix, f"{article_id}{ext}")
        if os.path.exists(src):
            dst_dir = os.path.join(IMAGE_OUT_DIR, prefix)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, f"{article_id}{ext}"))
            copied_images += 1
            break
    else:
        missing_images.append(article_id)

print(f"  Ảnh đã copy: {copied_images:,} | Không tìm thấy: {len(missing_images):,}")

if missing_images:
    pd.DataFrame({"article_id": missing_images}).to_csv(f"{OUT_DIR}/missing_images.csv", index=False)

# ── 6. Tóm tắt ────────────────────────────────────────────────────────────────
print(f"\n[6/6] Ghi demo_summary.txt ...")

ages = [
    float(customers[customers["customer_id"] == uid]["age"].values[0])
    for uid in selected_users
    if len(customers[customers["customer_id"] == uid]["age"].values) > 0
]

summary = f"""DEMO DATA SUMMARY — AuraFit Recommendation
===========================================
BỘ LỌC
  Độ tuổi   : {AGE_MIN} – {AGE_MAX}
  Sản phẩm  : Ladieswear

KẾT QUẢ
  Số user demo       : {len(selected_users)}
  Tuổi trung bình    : {sum(ages)/len(ages):.1f}
  Tuổi min/max       : {min(ages):.0f} / {max(ages):.0f}
  Số giao dịch       : {len(demo_test):,}
  Số sản phẩm unique : {len(demo_articles):,}
  Số ảnh demo        : {copied_images:,}
  Embeddings shape   : {demo_embeddings.shape}

FILES (data/demo/)
  test_20users.csv      — giao dịch lịch sử 20 user
  articles_demo.csv     — thông tin sản phẩm
  article_ids_demo.csv  — mapping index ↔ article_id
  embeddings_demo.npy   — FashionCLIP embeddings
  user_list.csv         — danh sách customer_id
  images/               — ảnh sản phẩm
"""

with open(f"{OUT_DIR}/demo_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
print(f"✓ Tất cả files đã lưu vào {OUT_DIR}/")