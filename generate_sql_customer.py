"""
generate_customer_sql.py
========================
Đọc user_list.csv (20 demo customers) từ máy local,
xuất ra file customer_import.sql để import vào phpMyAdmin.

KHÔNG cần kết nối database.

Cấu trúc bảng liên quan:
  - customer       : thông tin tài khoản (username, password, email, ...)
  - profile        : thông tin cá nhân (để trống, tạo sau)
  - cart           : giỏ hàng (tạo tự động khi insert customer)
  - hm_user_map    : mapping customer_id (int) ↔ hm_customer_id (string H&M) ↔ internal_idx

Chạy:
    pip install pandas tqdm
    python generate_sql_customer.py

Output:
    hm_customer_import.sql
"""

import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CONFIG — chỉnh các dòng này nếu cần
# ═══════════════════════════════════════════════════════════════
USER_LIST_CSV   = "data/demo/user_list.csv"       
CUSTOMERS_CSV   = "data/processed/hm/customers.csv"
OUTPUT_SQL      = "hm_customer_import.sql"
CUSTOMER_ID_START = 100

# Mật khẩu mặc định cho tất cả demo accounts (đã hash bcrypt)
# Giá trị này tương đương plaintext: "Demo@12345"
# DEFAULT_PASSWORD = "$2y$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uSc/kDTKK"
DEFAULT_PASSWORD = "DemoHM@12345"

EMBEDDING_MODEL = "fashionclip_ngcf"

BATCH_SIZE = 200
# ═══════════════════════════════════════════════════════════════


def esc(val) -> str:
    """Escape giá trị để dùng trong SQL string, trả về 'NULL' nếu None/NaN."""
    if val is None:
        return "NULL"
    if isinstance(val, float) and pd.isna(val):
        return "NULL"
    s = str(val)
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


def write_batches(f, table: str, columns: list, rows: list):
    """Ghi INSERT IGNORE theo batch."""
    if not rows:
        print(f"    ⚠ Không có row nào cho bảng {table}, bỏ qua.")
        return
    cols = ", ".join(columns)
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        values_str = ",\n  ".join(batch)
        f.write(f"INSERT IGNORE INTO `{table}` ({cols}) VALUES\n  {values_str};\n\n")


def main():
    print("=" * 60)
    print("AuraFit — Generate Customer SQL from H&M user_list (offline)")
    print("=" * 60)

    # ── Load CSV ──────────────────────────────────────────────
    print("\nĐọc CSV ...")

    if not Path(USER_LIST_CSV).exists():
        print(f"❌ Không tìm thấy: {USER_LIST_CSV}")
        sys.exit(1)

    user_list = pd.read_csv(USER_LIST_CSV, dtype={"customer_id": str})
    hm_ids = user_list["customer_id"].tolist()
    print(f"  user_list: {len(hm_ids)} users")

    # Load customers.csv gốc của H&M nếu có (để lấy age)
    customers_hm = None
    if Path(CUSTOMERS_CSV).exists():
        customers_hm = pd.read_csv(CUSTOMERS_CSV, dtype={"customer_id": str})
        print(f"  customers.csv: {len(customers_hm):,} rows")
    else:
        print(f"  ⚠ Không tìm thấy {CUSTOMERS_CSV} — sẽ dùng thông tin mặc định")

    # ── Ghi SQL ───────────────────────────────────────────────
    print(f"\nXuất ra {OUTPUT_SQL} ...")

    with open(OUTPUT_SQL, "w", encoding="utf-8") as f:

        # Header
        f.write(f"-- AuraFit Demo Customer Import\n")
        f.write(f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"-- {len(hm_ids)} demo customers\n")
        f.write(f"-- Password mặc định (plaintext): Demo@12345\n\n")
        f.write("SET NAMES utf8mb4;\n")
        f.write("SET foreign_key_checks = 0;\n\n")

        # ── 1. cart (phải tạo trước vì customer FK tới cart) ──
        print("  [1/4] cart ...")
        f.write("-- =============================================\n")
        f.write("-- 1. cart (tạo giỏ hàng trống cho mỗi demo user)\n")
        f.write("-- =============================================\n")
        cart_rows = []
        for i, _ in enumerate(hm_ids):
            cart_id = CUSTOMER_ID_START + i
            cart_rows.append(f"({cart_id}, {cart_id}, 0)")

        write_batches(
            f, "cart",
            ["`cart_id`", "`customer_id`", "`quantity`"],
            cart_rows,
        )

        # ── 2. customer ───────────────────────────────────────
        print("  [2/4] customer ...")
        f.write("-- =============================================\n")
        f.write("-- 2. customer\n")
        f.write("-- =============================================\n")

        customer_rows = []
        for i, hm_cid in enumerate(tqdm(hm_ids, desc="  customer")):
            db_id   = CUSTOMER_ID_START + i
            cart_id = db_id  # 1-1 với cart

            # Lấy tuổi từ file H&M nếu có
            birthday_val = "NULL"
            if customers_hm is not None:
                row = customers_hm[customers_hm["customer_id"] == hm_cid]
                if not row.empty and not pd.isna(row.iloc[0].get("age", float("nan"))):
                    age = int(row.iloc[0]["age"])
                    # Ước lượng năm sinh (demo), tháng/ngày mặc định 01-01
                    birth_year = datetime.now().year - age
                    birthday_val = f"'{birth_year}-01-01'"

            # username: hm_demo_u{i+1:02d}  (ngắn, dễ nhớ khi test)
            username = f"hm_demo_u{i+1:02d}"
            email    = f"hm_demo{i+1:02d}@aurafit.demo"

            customer_rows.append(
                f"({db_id}, {cart_id}, "
                f"{esc(username)}, {esc(DEFAULT_PASSWORD)}, {esc(email)}, "
                f"NULL, {esc(f'HMDemo')}, {esc(f'User{i+1:02d}')}, "
                f"{birthday_val}, NULL, "
                f"NULL, NULL, NULL, "
                f"CURRENT_TIMESTAMP, 0)"
            )

        write_batches(
            f, "customer",
            ["`customer_id`", "`cart_id`",
             "`username`", "`password`", "`email`",
             "`phone_number`", "`first_name`", "`last_name`",
             "`birthday`", "`avatar`",
             "`province_code`", "`ward_code`", "`address_detail`",
             "`joining_date`", "`is_admin`"],
            customer_rows,
        )

        # ── 3. profile (tạo row trống để app không bị lỗi khi query) ─
        print("  [3/4] profile ...")
        f.write("-- =============================================\n")
        f.write("-- 3. profile (row trống — user có thể điền sau)\n")
        f.write("-- =============================================\n")
        profile_rows = []
        for i, _ in enumerate(hm_ids):
            db_id = CUSTOMER_ID_START + i
            profile_rows.append(
                f"({db_id}, NULL, NULL, NULL, NULL, NULL, NULL, NULL)"
            )

        write_batches(
            f, "profile",
            ["`customer_id`", "`weight`", "`height`",
             "`body_shape`", "`personal_color`", "`favourite_styles`",
             "`face_image`", "`portrait_image`"],
            profile_rows,
        )

        # ── 4. hm_user_map ────────────────────────────────────
        print("  [4/4] hm_user_map ...")
        f.write("-- =============================================\n")
        f.write("-- 4. hm_user_map (mapping DB customer_id ↔ H&M customer_id ↔ internal_idx)\n")
        f.write("-- =============================================\n")
        map_rows = []
        for i, hm_cid in enumerate(tqdm(hm_ids, desc="  hm_user_map")):
            db_id        = CUSTOMER_ID_START + i
            internal_idx = i   # index trong model NGCF (0-based, tương ứng vị trí trong user_list.csv)
            map_rows.append(
                f"({db_id}, {esc(hm_cid)}, {internal_idx}, {esc(EMBEDDING_MODEL)})"
            )

        write_batches(
            f, "hm_user_map",
            ["`customer_id`", "`hm_customer_id`", "`internal_idx`", "`embedding_model`"],
            map_rows,
        )

        # Footer
        f.write("SET foreign_key_checks = 1;\n")
        f.write("-- =============================================\n")
        f.write(f"-- Import hoàn tất: {len(hm_ids)} demo customers\n")
        f.write(f"-- customer_id range: {CUSTOMER_ID_START} → {CUSTOMER_ID_START + len(hm_ids) - 1}\n")
        f.write("-- =============================================\n")

    # ── Summary ───────────────────────────────────────────────
    size_kb = Path(OUTPUT_SQL).stat().st_size / 1024
    print(f"\n✓ Xong! File: {OUTPUT_SQL} ({size_kb:.1f} KB)")
    print(f"\n{'─'*50}")
    print(f"  {'Customer ID trong DB':30s}: {CUSTOMER_ID_START} → {CUSTOMER_ID_START + len(hm_ids) - 1}")
    print(f"  {'API endpoint để test':30s}: /recommend/{{customer_id}}")
    print(f"  {'Ví dụ':30s}: /recommend/{CUSTOMER_ID_START}")
    print(f"  {'Username / Password':30s}: hm_demo_u01 / HMDemo@12345")
    print(f"{'─'*50}")
    print()
    print("⚠  Lưu ý quan trọng:")
    print("   1. CUSTOMER_ID_START phải lớn hơn customer_id lớn nhất đang có trong DB")
    print("      → Kiểm tra: SELECT MAX(customer_id) FROM customer;")
    print("   2. internal_idx trong hm_user_map là index 0-based theo thứ tự user_list.csv")
    print("      → API gọi: /recommend/{customer_id}")
    print("      → Backend cần lookup: SELECT internal_idx FROM hm_user_map WHERE customer_id=?")
    print()


if __name__ == "__main__":
    main()