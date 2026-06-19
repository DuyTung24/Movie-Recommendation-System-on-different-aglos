import pandas as pd
import os
import argparse
import yaml 
import numpy as np
from src.data_loader import MovieLensDataLoader
from src.model import MatrixFactorizationModel
from src.optimizer import SGDOptimizer
from src.trainer import Trainer
from src.metrics import Metrics

def load_config(config_path="config.yaml"):
    """Hàm đọc file cấu hình YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. Cấu hình tham số dòng lệnh (CLI Arguments)
    parser = argparse.ArgumentParser(description="Matrix Factorization Recommendation System - MovieLens 1M")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['train', 'predict'], 
        help="Chế độ chạy: 'train' để huấn luyện, 'predict' để đánh giá/dự đoán."
    )
    args = parser.parse_args()

    # Load cấu hình từ file YAML
    config = load_config()
    paths = config['paths']
    hp = config['hyperparameters']

    # 1. Load dữ liệu MovieLens 1M
    loader = MovieLensDataLoader(paths['data_path'])
    full_data = loader.load_and_preprocess()
    num_users, num_items = loader.get_dimensions()
    
    # 2. XỬ LÝ CHIA DỮ LIỆU & CHỐNG DATA LEAKAGE
    split_save_path = paths['model_save_path'].replace('.npz', '_data_split.npz')

    if args.mode == "train":
        print("\n>>> TIẾN HÀNH CHIA NGẪU NHIÊN HOÀN TOÀN (RANDOM SPLIT 80/20)...")
        df = loader.dataset.copy()
        train_list = []
        test_list = []
        
        for user_id, group in df.groupby('user_idx'):
            n_items = len(group)
            if n_items < 5:
                train_list.append(group)
                continue
                
            # Chia ngẫu nhiên không cố định random_state để mỗi lần train là 1 lần xáo trộn mới
            train_group = group.sample(frac=0.8)
            test_group = group.drop(train_group.index)
            
            train_list.append(train_group)
            test_list.append(test_group)
            
        train_df = pd.concat(train_list)
        test_df = pd.concat(test_list)
        
        train_data = train_df[['user_idx', 'item_idx', 'rating']].values
        test_data = test_df[['user_idx', 'item_idx', 'rating']].values

        # LƯU LẠI TẬP TRAIN/TEST
        np.savez(split_save_path, train_data=train_data, test_data=test_data)
        print(f"Đã lưu cố định tập phân chia dữ liệu tại: {split_save_path}")

    elif args.mode == "predict":
        print("\n>>> TẢI LẠI TẬP DỮ LIỆU TỪ LẦN TRAIN GẦN NHẤT (CHỐNG DATA LEAKAGE)...")
        if not os.path.exists(split_save_path):
            raise FileNotFoundError(f"Không tìm thấy file {split_save_path}. Vui lòng chạy mode 'train' trước!")
            
        split_data = np.load(split_save_path)
        train_data = split_data['train_data']
        test_data = split_data['test_data']
        print("Đã tải xong. Đảm bảo Train/Test giống hệt lúc huấn luyện!")

    # 3. KHỞI TẠO MODEL VÀ KIỂM TRA TRỌNG SỐ CŨ
    model = MatrixFactorizationModel(num_users, num_items, num_factors=hp['k'])
    initial_best_ndcg = -1.0 
    
    if args.mode == "predict":
        if os.path.exists(paths['model_save_path']):
            print(f"\n>>> [PREDICT MODE] Đã tải trọng số từ {paths['model_save_path']}")
            model.load_weights(paths['model_save_path'])
            
            print("Đang đánh giá hiệu năng model trên tập Test...")
            old_metrics = Metrics.get_ranking_metrics(model, test_data, train_data, num_items, k=hp['top_k'])
            initial_best_ndcg = old_metrics[f"NDCG@{hp['top_k']}"]
            print(f"Kỷ lục hiện tại (NDCG@{hp['top_k']}): {initial_best_ndcg:.4f}")
        else:
            raise FileNotFoundError(f"Không tìm thấy model tại {paths['model_save_path']}. Hãy chạy mode 'train' trước!")

    elif args.mode == "train":
        if os.path.exists(paths['model_save_path']):
            print(f"\n>>> [TRAIN MODE] Phát hiện file trọng số cũ tại {paths['model_save_path']}.")
            print(">>> Hệ thống sẽ tự động GHI ĐÈ file này bằng mô hình tối ưu mới nhất!")
        else:
            print("\n>>> [TRAIN MODE] Khởi tạo mô hình hoàn toàn mới.")

    # 4. KÍCH HOẠT QUÁ TRÌNH HUẤN LUYỆN
    if args.mode == "train":
        print("\n>>> BẮT ĐẦU CHẾ ĐỘ HUẤN LUYỆN TỐI ƯU...")
        optimizer = SGDOptimizer(model, learning_rate=hp['learning_rate'], regularization=hp['regularization'])
        
        trainer = Trainer(model, optimizer, train_data, test_data, initial_best_score=initial_best_ndcg)
        
        # Gọi hàm train thực sự
        trainer.train(
            epochs=hp['epochs'], 
            save_path=paths['model_save_path'], 
            top_k=hp['top_k'],
            patience=hp.get('patience', 10)
        )

    # 5. DỰ ĐOÁN VÀ ĐÁNH GIÁ CUỐI CÙNG
    print("\n" + "="*30)
    print("DỰ ĐOÁN VÀ ĐÁNH GIÁ CUỐI CÙNG")
    print("="*30)
    
    # Demo thử một cặp User - Movie
    demo_user_id = 1
    demo_movie_id = 1193
    u_idx = loader.user2idx.get(demo_user_id)
    i_idx = loader.item2idx.get(demo_movie_id)
    
    if u_idx is not None and i_idx is not None:
        pred_rating = model.predict(u_idx, i_idx)
        print(f"Dự đoán Rating cho User {demo_user_id} - Movie {demo_movie_id}: {pred_rating:.2f}")

    # Đánh giá toàn bộ tập metrics cuối cùng
    ranking_metrics = Metrics.get_ranking_metrics(model, test_data, train_data, num_items, k=hp['top_k'])
    for name, val in ranking_metrics.items():
        print(f"{name}: {val:.4f}") 

if __name__ == "__main__":
    main()