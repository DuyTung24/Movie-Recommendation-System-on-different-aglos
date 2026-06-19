# src/trainer.py (ĐÃ SỬA)

import numpy as np
import time
from src.metrics import Metrics

class Trainer:
    def __init__(self, model, optimizer, train_data, test_data=None, initial_best_score=-1.0):
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.best_ndcg = initial_best_score
        self.model.mu = np.mean(self.train_data[:, 2])

    def train(self, epochs=10, save_path="best_model.npz", top_k=5, patience=10):
        num_items = self.model.num_items
        patience_counter = 0

        print(f"Bắt đầu huấn luyện với {epochs} epochs (Early Stopping = {patience})...")
        print("-" * 50)

        for epoch in range(epochs):
            start_time = time.time()
            
            # Đã xáo trộn ngẫu nhiên, không có seed cố định
            np.random.shuffle(self.train_data)
            
            # --- CHỈ HUẤN LUYỆN TRÊN EXPLICIT RATINGS (Không Negative Sampling) ---
            for u, i, rating in self.train_data:
                self.optimizer.step(int(u), int(i), float(rating))
            
            # Đánh giá sau mỗi epoch
            train_rmse = np.sqrt(self.optimizer.calculate_loss(self.train_data))
            epoch_info = f"Epoch {epoch + 1}/{epochs} | Train RMSE: {train_rmse:.4f}"
            
            if self.test_data is not None:
                test_rmse = np.sqrt(self.optimizer.calculate_loss(self.test_data))
                epoch_info += f" | Test RMSE: {test_rmse:.4f}"
                
                ranking_metrics = Metrics.get_ranking_metrics(
                    self.model, self.test_data, self.train_data, n_items=num_items, k=top_k
                )
                current_ndcg = ranking_metrics[f'NDCG@{top_k}']
                epoch_info += f" | NDCG@{top_k}: {current_ndcg:.4f}"

                # Lọc Early Stopping
                if current_ndcg > self.best_ndcg:
                    self.best_ndcg = current_ndcg
                    self.model.save_weights(save_path)
                    epoch_info += " [NEW_BEST SAVED]"
                    patience_counter = 0
                else:
                    patience_counter += 1
                    epoch_info += f" (Best: {self.best_ndcg:.4f} | No Improve: {patience_counter}/{patience})"
            
            duration = time.time() - start_time
            print(f"{epoch_info} | Time: {duration:.2f}s")
            
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Dừng sớm tại Epoch {epoch + 1}!")
                break