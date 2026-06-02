import os
import argparse
import numpy as np
import pandas as pd
import yaml
from src.data_loader import MovieLens1MLoader
from src.model import MemoryBasedCF
from src.metrics import RecommenderEvaluator

def parse_arguments():
    # Giữ lại argparse duy nhất cho việc điều khiển luồng chạy (Mode)
    parser = argparse.ArgumentParser(description="Hệ thống Khuyến nghị Phim Memory-Based CF tối ưu trên MovieLens 1M")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help="Luồng chạy: 'train' để tính mới trọng số, 'predict' để tải lại file trọng số sẵn có")
    return parser.parse_args()

def load_system_config(config_path='config.yaml'):
    """Hàm helper đọc tệp cấu hình YAML an toàn"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[Lỗi] Không tìm thấy tệp cấu hình tại: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    cfg = load_system_config('config.yaml')
    args = parse_arguments()
    np.random.seed(cfg['execution']['seed'])  # Đồng bộ seed từ file config
    
    # Lấy giá trị top_k từ config để code gọn gàng hơn
    top_k = cfg['evaluation']['top_k']
    
    print(f"\n=========================================================")
    print(f" KHỞI CHẠY HỆ THỐNG GỢI Ý PHIM CHUẨN HÓA (MOVIELENS 1M)")
    print(f" Thuật toán: {cfg['model']['cf_mode'].upper()}-based CF")
    print(f" Luồng chạy: {args.mode.upper()}")
    print(f" Cấu hình:  Neighbors={cfg['model']['n_neighbors']} | Top-K={top_k}")
    print(f"=========================================================\n")
    
    # 1. Đọc và cấu trúc ma trận dữ liệu (Lấy path động từ config)
    loader = MovieLens1MLoader(data_dir=cfg['storage']['data_dir'])
    if not os.path.exists(os.path.join(loader.data_dir, 'ratings.dat')):
        print(f"[Lỗi] Không thấy thư mục dữ liệu tại '{loader.data_dir}'. Vui lòng giải nén bộ MovieLens 1M vào đây.")
        return
        
    print("-> Bước 1: Đang nạp dữ liệu thô MovieLens 1M...")
    ratings, movies = loader.load_raw_data()
    print(f"-> Thống kê động: Hệ thống nhận diện {loader.n_users} Người dùng & {loader.n_items} Vật phẩm Phim.")
    
    print("-> Bước 2: Đang chia tách dữ liệu dạng Leave-One-Out...")
    rate_train, rate_test = loader.train_test_split_loo(ratings)
    
    print("-> Bước 3: Đang chuyển dịch xây dựng Ma trận tiện ích thưa (Sparse Matrix)...")
    utility_sparse, utility_norm_sparse, user_means = loader.build_utility_matrix(rate_train)
    
    # 2. Khởi tạo thực thể mô hình
    model = MemoryBasedCF(mode=cfg['model']['cf_mode'], n_neighbors=cfg['model']['n_neighbors'])
    
    # Tạo đường dẫn checkpoint động dựa theo cf_mode trong config
    checkpoint_path = os.path.join(cfg['storage']['checkpoint_dir'], f"{cfg['model']['cf_mode']}_cf_checkpoint.pkl")
    
    # 3. Phân luồng logic xử lý dựa trên tham số dòng lệnh đầu vào (--mode)
    if args.mode == 'train':
        model.compute_similarity(utility_norm_sparse)
        model.save_checkpoint(checkpoint_path)
    elif args.mode == 'predict':
        if not model.load_checkpoint(checkpoint_path):
            print(f"[Cảnh báo] Không tìm thấy file trọng số tại '{checkpoint_path}'. Chuyển sang luồng Train tự động...")
            model.compute_similarity(utility_norm_sparse)
            model.save_checkpoint(checkpoint_path)

    # 4. Đánh giá và kiểm định chất lượng mô hình
    print(f"-> Bước 4: Đang chuẩn bị tập mẫu âm ({cfg['evaluation']['n_negatives']} Negatives) để phục vụ kiểm tra...")
    user_pos, user_neg = RecommenderEvaluator.build_ranking_data(
        rate_train, rate_test, n_items=loader.n_items, n_neg=cfg['evaluation']['n_negatives']
    )
    
    # SỬA LỖI: Thay args.top_k bằng biến top_k lấy từ config
    rmse, precision, recall, ndcg, mrr = RecommenderEvaluator.evaluate_model(
        model, rate_test, user_pos, user_neg, utility_sparse, utility_norm_sparse, user_means, K=top_k
    )
    
    # SỬA LỖI: Hiển thị chỉ số động theo config
    print(f"\n[KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM ĐỊNH]")
    print(f"  RMSE              : {rmse:.4f}")
    print(f"  Precision@{top_k}       : {precision:.4f}")
    print(f"  Recall@{top_k}          : {recall:.4f}")
    print(f"  NDCG@{top_k}            : {ndcg:.4f}")
    print(f"  MRR@{top_k}             : {mrr:.4f}")
    
    # 5. Thực hiện tác vụ Dự đoán cho một thực thể ngẫu nhiên thực tế (Inference)
    print(f"\n[DỰ ĐOÁN ĐIỂM SỐ NGẪU NHIÊN (INFERENCE)]")
    rand_user_idx = np.random.choice(list(user_pos.keys()))
    
    start_idx = utility_sparse.indptr[rand_user_idx]
    end_idx = utility_sparse.indptr[rand_user_idx + 1]
    rated_by_user = utility_sparse.indices[start_idx:end_idx]
    unrated_items = np.setdiff1d(np.arange(loader.n_items), rated_by_user)
    
    if len(unrated_items) > 0:
        rand_item_idx = np.random.choice(unrated_items)
        
        utility_csc = utility_sparse.tocsc() if model.mode == 'user' else None
        utility_norm_csc = utility_norm_sparse.tocsc() if model.mode == 'user' else None
        
        predicted_score = model.predict_for_user_items(
            rand_user_idx, [rand_item_idx], utility_sparse, utility_norm_sparse, user_means,
            utility_csc=utility_csc, utility_norm_csc=utility_norm_csc
        )[0]
        
        movie_id_actual = rand_item_idx + 1
        movie_title = movies[movies['movie_id'] == movie_id_actual]['movie_title'].values
        title_str = movie_title[0] if len(movie_title) > 0 else f"ID {movie_id_actual}"
        
        print(f"  User ID ngẫu nhiên  : {rand_user_idx + 1}")
        print(f"  Tên Vật Phẩm (Phim) : {title_str}")
        print(f"  Điểm số Dự Đoán     : {predicted_score:.2f} / 5.0")

if __name__ == "__main__":
    main()