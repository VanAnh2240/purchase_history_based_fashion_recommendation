"""
PATCH cho train_siamese_hm_user.py
Thay thế toàn bộ hàm build_user_history_emb và phần gọi nó trong train()
bằng get_user_hist_emb từ precompute_user_emb.py
"""

# ── 1. Thêm import này vào đầu train_siamese_hm.py ───────────────────────────

from src.training.hm.precompute_user_emb import get_user_hist_emb

# ── 2. XOÁ toàn bộ hàm build_user_history_emb() (không cần nữa) ──────────────

# ── 3. Trong HMSiameseTrainer.train(), tìm đoạn: ─────────────────────────────
#
#   # 5. User history embeddings → mmap trên disk, KHÔNG giữ 2.6 GB trong RAM
#   user_hist_emb, _user_emb_tmp = build_user_history_emb(
#       train_u, train_i, item_feat, n_users)
#
#   # Đăng ký xoá thư mục temp khi thoát
#   def _cleanup_user_emb():
#       ...
#   atexit.register(_cleanup_user_emb)
#
# ── Thay bằng: ────────────────────────────────────────────────────────────────

"""
        # 5. User history embeddings — load từ cache nếu có, tính 1 lần nếu chưa
        #    Cache tại: EMBEDDING_DIR/hm/<feature>/user_hist_emb.npy
        #    Chạy trước: python -m src.training.hm.precompute_user_emb --feature <feature>
        user_hist_emb = get_user_hist_emb(
            feature   = self.feature,
            emb_dir   = self.emb_dir,
            graph_dir = self.graph_dir,
            data_dir  = self.data_dir,
        )
        # Sau khi có user_hist_emb, không cần train arrays nữa
"""

# ── 4. Đoạn "6. Train dataset" giữ nguyên: ───────────────────────────────────
#
#   train_ds = SiameseTrainDataset(
#       train_u, train_i, user_hist_emb, item_feat, n_items)
#   del train_u, train_i
#   gc.collect()
#
# Không thay đổi gì.

# ── Kết quả: ─────────────────────────────────────────────────────────────────
# Lần đầu train:  tính ~7 phút, cache ra disk, load vào RAM
# Các lần sau:    chỉ np.load() ~vài giây