
import numpy as np
import math
from collections import defaultdict

class Metrics:
    @staticmethod
    def calculate_rmse(model, data):
        """Tính Root Mean Square Error cho các bài toán dự đoán điểm số (Rating)"""
        mse = 0
        for u, i, rating in data:
            prediction = model.predict(int(u), int(i))
            mse += (rating - prediction) ** 2
        return np.sqrt(mse / len(data))

    @staticmethod
    def get_ranking_metrics(model, test_data, train_data, n_items, k=10):
        sum_precision = 0 # Bổ sung biến lưu tổng Precision
        sum_recall = 0
        sum_ndcg = 0
        sum_mrr = 0
        
        # 1. Nhóm các phim đã xem trong Train
        user_train_items = defaultdict(set)
        for u, i, _ in train_data:
            user_train_items[int(u)].add(int(i))
            
        # 2. Nhóm các phim trong Test để làm Ground Truth 
        user_test_items = defaultdict(set)
        for u, i, rating in test_data:
            if rating >= 4.0:
                user_test_items[int(u)].add(int(i))
                
        valid_users = 0
        
        # 3. Đánh giá cho từng người dùng có dữ liệu test
        for u_idx, true_items in user_test_items.items():
            if len(true_items) == 0:
                continue 
                
            valid_users += 1
            train_items = user_train_items.get(u_idx, set())
            
            # --- FULL RANKING BẰNG VECTORIZATION ---
            scores = model.mu + model.b_u[u_idx] + model.b_i + model.P[u_idx, :].dot(model.Q.T)
            
            if len(train_items) > 0:
                scores[list(train_items)] = -np.inf 
            
            # Lấy vị trí của K bộ phim có điểm số cao nhất
            top_k_indices = np.argsort(scores)[::-1][:k]
            
            # Tính số lượng dự đoán trúng (Hits)
            hits = len(set(top_k_indices).intersection(true_items))
            
            # --- TÍNH PRECISION@K VÀ RECALL@K ---
            sum_precision += hits / k               # Thêm mới ở đây
            sum_recall += hits / len(true_items)
            
            # --- TÍNH MRR@K ---
            mrr = 0
            for rank, item in enumerate(top_k_indices):
                if item in true_items:
                    mrr = 1 / (rank + 1)
                    break 
            sum_mrr += mrr
            
            # --- TÍNH NDCG@K ---
            dcg = 0
            for rank, item in enumerate(top_k_indices):
                if item in true_items:
                    dcg += 1 / math.log2(rank + 2)
            
            idcg = 0
            for rank in range(min(len(true_items), k)):
                idcg += 1 / math.log2(rank + 2)
                
            if idcg > 0:
                sum_ndcg += dcg / idcg

        if valid_users == 0:
            return {f'Precision@{k}': 0, f'Recall@{k}': 0, f'MRR@{k}': 0, f'NDCG@{k}': 0}

        # Trả về thêm Precision@K
        return {
            f'Precision@{k}': sum_precision / valid_users,
            f'Recall@{k}': sum_recall / valid_users,
            f'MRR@{k}': sum_mrr / valid_users,
            f'NDCG@{k}': sum_ndcg / valid_users
        }