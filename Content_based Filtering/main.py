import argparse
import os
import random
import numpy as np
from data_loader import load_data, leave_one_out_split
from model import ContentBasedRecommender
from evaluator import evaluate_rmse, evaluate_top_n

def main():

    np.random.seed(42)
    random.seed(42)
    # Thiết lập bộ nhận diện tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Content-Based Recommender System với ML-1M")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help="Chế độ chạy: 'train' (huấn luyện & lưu) hoặc 'predict' (load trọng số & đánh giá)")
    args = parser.parse_args()

    # Đường dẫn lưu trọng số
    weights_dir = './weights'
    weights_path = os.path.join(weights_dir, 'cb_weights.npz')

    print("1. Đang tải dataset ML-1M và xử lý đặc trưng...")
    ratings, item_features, user2idx, movie2idx = load_data(path='./data/ml-1m')
    
    n_users = len(user2idx)
    n_items = len(movie2idx)
    d_features = item_features.shape[1]
    print(f"   Khởi tạo cấu trúc: Users: {n_users}, Items: {n_items}, Features: {d_features}")
    
    print("2. Chia dữ liệu theo phương pháp Leave-One-Out...")
    train_df, test_df = leave_one_out_split(ratings)
    
    # Khởi tạo mô hình trống
    model = ContentBasedRecommender(n_users, d_features, alpha=1.0)

    if args.mode == 'train':
        print("3. Bắt đầu huấn luyện mô hình (Ridge Regression)...")
        model.fit(train_df, item_features)
        
        # Tạo thư mục weights nếu chưa có và lưu lại
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        model.save_weights(weights_path)
        
    elif args.mode == 'predict':
        print("3. Đang đọc trọng số có sẵn...")
        if not os.path.exists(weights_path):
            print(f"Lỗi: Không tìm thấy file trọng số tại '{weights_path}'. Vui lòng chạy `--mode train` trước!")
            return
        model.load_weights(weights_path)

    # === PHẦN ĐÁNH GIÁ METRICS (Chạy ở cả 2 chế độ hoặc ưu tiên Predict) ===
    print("\n4. Đang tính toán các Metrics đánh giá hệ thống...")
    rmse = evaluate_rmse(model, test_df, item_features)
    print(f"   -> RMSE: {rmse:.4f}")
    
    K = 10
    print(f"5. Đang đánh giá thứ hạng Top-{K} (Leave-one-out với 99 negative samples)...")
    metrics = evaluate_top_n(model, train_df, test_df, item_features, n_items, k=K, n_negatives=99)
    
    for metric_name, value in metrics.items():
        print(f"   -> {metric_name}: {value:.4f}")

    # === PHẦN PREDICT CHO USER & ITEM NGẪU NHIÊN ===
    print("\n6. Tiến hành dự đoán thử nghiệm cho User và Item ngẫu nhiên:")
    random_u_idx = random.randint(0, n_users - 1)
    random_m_idx = random.randint(0, n_items - 1)
    
    # Dự đoán điểm số
    predicted_rating = model.predict(random_u_idx, [random_m_idx], item_features)[0]
    
    print(f"   - User Index ngẫu nhiên: {random_u_idx}")
    print(f"   - Movie Index ngẫu nhiên: {random_m_idx}")
    print(f"   => Dự đoán số sao (Predicted Rating): {predicted_rating:.2f} / 5.0")

if __name__ == "__main__":
    main()