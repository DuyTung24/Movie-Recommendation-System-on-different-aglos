import argparse
import os
import random
import numpy as np
import pandas as pd
from data_loader import load_data, random_80_20_split 
from model import ContentBasedRecommender
from evaluator import evaluate_rmse, evaluate_top_n

def main():

    parser = argparse.ArgumentParser(description="Content-Based Recommender System với ML-1M")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help="Chế độ chạy: 'train' (huấn luyện & lưu) hoặc 'predict' (load trọng số & đánh giá)")
    args = parser.parse_args()

    weights_dir = './weights'
    weights_path = os.path.join(weights_dir, 'cb_weights.npz')
    train_df_path = os.path.join(weights_dir, 'train_split.csv')
    test_df_path = os.path.join(weights_dir, 'test_split.csv')

    print("1. Đang tải dữ liệu đặc trưng hệ thống...")
    # Lưu ý: Ở chế độ predict, ta chỉ cần load item_features và ma trận ánh xạ index ban đầu
    ratings, item_features, user2idx, movie2idx = load_data(path='./data/ml-1m')
    
    n_users = len(user2idx)
    n_items = len(movie2idx)
    d_features = item_features.shape[1]
    
    model = ContentBasedRecommender(n_users, d_features, alpha=1.0)

    # ==================== CHẾ ĐỘ TRAIN ====================
    if args.mode == 'train':
        print("2. [TRAIN] Chia dữ liệu ngẫu nhiên 80/20...")
        train_df, test_df = random_80_20_split(ratings, ratio=0.8)
        
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            
        # Lưu lại kiến trúc chia dữ liệu để lần sau dùng lại đúng tập Test này
        train_df.to_csv(train_df_path, index=False)
        test_df.to_csv(test_df_path, index=False)
        print("   -> Đã lưu cấu trúc tập Train/Test vào thư mục weights/")

        print("3. [TRAIN] Bắt đầu huấn luyện mô hình (Ridge Regression)...")
        model.fit(train_df, item_features)
        model.save_weights(weights_path)
        
    # ==================== CHẾ ĐỘ PREDICT ====================
    elif args.mode == 'predict':
        print("2. [PREDICT] Kiểm tra và nạp lại cấu trúc dữ liệu cũ...")
        if not (os.path.exists(weights_path) and os.path.exists(train_df_path) and os.path.exists(test_df_path)):
            print(f"Lỗi: Thiếu file trọng số hoặc dữ liệu split tại '{weights_dir}'. Vui lòng chạy `--mode train` trước!")
            return
            
        train_df = pd.read_csv(train_df_path)
        test_df = pd.read_csv(test_df_path)
        print(f"   -> Tải thành công tập Train ({len(train_df)} dòng) và Test ({len(test_df)} dòng).")
        
        print("3. [PREDICT] Đang nạp trọng số mô hình đã lưu...")
        model.load_weights(weights_path)

    # ==================== ĐÁNH GIÁ METRICS ====================
    print("\n4. Đang tính toán các Metrics đánh giá hệ thống...")
    rmse = evaluate_rmse(model, test_df, item_features)
    print(f"   -> RMSE: {rmse:.4f}")
    
    K = 20
    print(f"5. Đang đánh giá thứ hạng Top-{K}...")
    metrics = evaluate_top_n(model, train_df, test_df, item_features, n_items, k=K)
    
    for metric_name, value in metrics.items():
        print(f"   -> {metric_name}: {value:.4f}")

    # ==================== DỰ ĐOÁN NGẪU NHIÊN CHO NGƯỜI DÙNG BẤT KỲ ====================
    print("\n6. Tiến hành dự đoán thử nghiệm cho User và Item ngẫu nhiên:")
    random_u_idx = random.randint(0, n_users - 1)
    random_m_idx = random.randint(0, n_items - 1)
    
    predicted_rating = model.predict(random_u_idx, [random_m_idx], item_features)[0]
    
    print(f"   - User Index ngẫu nhiên hiện tại: {random_u_idx}")
    print(f"   - Movie Index ngẫu nhiên hiện tại: {random_m_idx}")
    print(f"   => Dự đoán số sao (Predicted Rating): {predicted_rating:.2f} / 5.0")

if __name__ == "__main__":
    main()