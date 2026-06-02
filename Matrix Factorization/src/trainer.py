
import numpy as np
import time
from src.metrics import Metrics

class Trainer:
    def __init__(self, model, optimizer, train_data, test_data=None, initial_best_score=-1.0):
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        
        # Biến để theo dõi kết quả tốt nhất
        self.best_ndcg = initial_best_score
        self.model.mu = np.mean(self.train_data[:, 2])

    def train(self, epochs=10, save_path="best_model.npz", top_k=5, patience=10):
        """
        Thực hiện vòng lặp huấn luyện và lưu lại trọng số tốt nhất dựa trên NDCG.
        Tích hợp cơ chế Early Stopping dựa trên tham số patience.
        """
        num_items = self.model.num_items
        patience_counter = 0  # Khởi tạo bộ đếm số epoch không cải tiến

        print(f"Bắt đầu huấn luyện với {epochs} epochs (Early Stopping patience = {patience})...")
        print("-" * 50)

        for epoch in range(epochs):
            start_time = time.time()
            np.random.shuffle(self.train_data)
            
            # 1. Cập nhật trọng số bằng SGD qua từng sample (Python thuần)
            for u, i, rating in self.train_data:
                self.optimizer.step(int(u), int(i), rating)
            
            # 2. Đánh giá sau mỗi epoch
            train_rmse = np.sqrt(self.optimizer.calculate_loss(self.train_data))
            epoch_info = f"Epoch {epoch + 1}/{epochs} | Train RMSE: {train_rmse:.4f}"
            
            if self.test_data is not None:
                test_rmse = np.sqrt(self.optimizer.calculate_loss(self.test_data))
                epoch_info += f" | Test RMSE: {test_rmse:.4f}"
                
                # Tính NDCG dựa trên hàm LOO metrics đã tối ưu bằng dict tra cứu O(1)
                loo_metrics = Metrics.get_loo_metrics(
                    self.model, self.test_data, self.train_data, n_items=num_items, k=top_k
                )
                current_ndcg = loo_metrics[f'NDCG@{top_k}']
                epoch_info += f" | NDCG@{top_k}: {current_ndcg:.4f}"

                # --- LOGIC EARLY STOPPING ---
                if current_ndcg > self.best_ndcg:
                    self.best_ndcg = current_ndcg
                    self.model.save_weights(save_path)
                    epoch_info += " [NEW_BEST SAVED]"
                    patience_counter = 0  # Reset bộ đếm về 0 khi có cải tiến hiệu năng
                else:
                    patience_counter += 1  # Tăng bộ đếm nếu hiệu năng không vượt qua best cũ
                    epoch_info += f" (Best: {self.best_ndcg:.4f} | No Improve: {patience_counter}/{patience})"
            
            duration = time.time() - start_time
            print(f"{epoch_info} | Time: {duration:.2f}s")
            
            # Kiểm tra điều kiện dừng sớm
            if patience_counter >= patience:
                print(f"\n[Early Stopping] NDCG không cải tiến sau {patience} epochs liên tiếp.")
                print(f"Hệ thống tự động dừng huấn luyện sớm tại Epoch {epoch + 1} để tránh Overfitting!")
                break