
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

    # Thiết lập seed cố định cho NumPy
    np.random.seed(42)

    # 2. Load dữ liệu MovieLens 1M
    loader = MovieLensDataLoader(paths['data_path'])
    full_data = loader.load_and_preprocess()
    num_users, num_items = loader.get_dimensions()
    
    # Chia dữ liệu theo phương pháp Leave-One-Out (LOO)
    df = loader.dataset.copy()
    high_rating_df = df[df['rating'] >= 4]
    test_df = high_rating_df.groupby('user_idx').tail(1)
    train_df = df.drop(test_df.index)
    train_data, test_data = train_df.values, test_df.values

    # 3. Khởi tạo Model sử dụng tham số từ block 'hyperparameters' trong YAML
    model = MatrixFactorizationModel(num_users, num_items, num_factors=hp['k'])
    initial_best_ndcg = -1.0 
    
    if os.path.exists(paths['model_save_path']):
        print(f"\n>>> Tìm thấy file trọng số cũ tại {paths['model_save_path']}")
        model.load_weights(paths['model_save_path'])
        
        # Tính thử NDCG của model cũ làm mốc so sánh
        print("Đang đánh giá hiệu năng model cũ làm mốc so sánh...")
        old_metrics = Metrics.get_loo_metrics(model, test_data, train_data, num_items, k=hp['top_k'])
        initial_best_ndcg = old_metrics[f"NDCG@{hp['top_k']}"]
        print(f"Kỷ lục hiện tại (NDCG@{hp['top_k']}): {initial_best_ndcg:.4f}")

    # 4. Kiểm tra chế độ thực thi từ CLI
    if args.mode == "train":
        print("\n>>> BẮT ĐẦU CHẾ ĐỘ HUẤN LUYỆN TỐI ƯU...")
        optimizer = SGDOptimizer(model, learning_rate=hp['learning_rate'], regularization=hp['regularization'])
        
        trainer = Trainer(model, optimizer, train_data, test_data, initial_best_score=initial_best_ndcg)
        # Huấn luyện và tự động lưu đè nếu vượt kỷ lục cũ
        trainer.train(epochs=hp['epochs'], save_path=paths['model_save_path'], top_k=hp['top_k'])
        
    elif args.mode == "predict":
        print("\n>>> CHẾ ĐỘ PREDICT: ĐÃ LOAD TRỌNG SỐ, KHÔNG HUẤN LUYỆN LẠI.")

    # 5. DỰ ĐOÁN VÀ ĐÁNH GIÁ CHUNG
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
    loo_metrics = Metrics.get_loo_metrics(model, test_data, train_data, num_items, k=hp['top_k'])
    for name, val in loo_metrics.items():
        print(f"{name}: {val:.4f}")

if __name__ == "__main__":
    main()