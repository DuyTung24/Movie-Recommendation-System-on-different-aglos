
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
    def get_loo_metrics(model, test_data, train_data, n_items, k=5, n_neg=99):
        hits = 0
        sum_mrr = 0
        sum_ndcg = 0
        num_users = len(test_data)
        
        all_items = set(range(n_items))

        # --- BƯỚC TỐI ƯU CỐT LÕI: Nhóm trước các item đã tương tác trong train_data ---
        # Quét train_data đúng 1 lần duy nhất, chi phí O(N) thay vì O(U * N)
        user_train_items = defaultdict(set)
        for u, i, _ in train_data:
            user_train_items[int(u)].add(int(i))
        # --------------------------------------------------------------------------

        for u_idx, pos_item_idx, _ in test_data:
            u_idx, pos_item_idx = int(u_idx), int(pos_item_idx)
            
            # Tối ưu: Tra cứu Dict với độ phức tạp O(1) cực kỳ nhanh
            train_items = user_train_items[u_idx]
            
            # Negative sampling: 99 items user chưa xem
            candidates = list(all_items - train_items - {pos_item_idx})
            neg_items = np.random.choice(candidates, n_neg, replace=False)
            
            # Tập items để rank (100 items)
            test_items = np.append(neg_items, pos_item_idx)
            
            # Dự đoán score cho 100 items này
            scores = []
            for i_idx in test_items:
                scores.append(model.predict(u_idx, i_idx))
            
            # Sắp xếp và tìm rank của pos_item_idx
            ranked_indices = np.argsort(scores)[::-1]
            rank = np.where(test_items[ranked_indices] == pos_item_idx)[0][0]

            if rank < k:
                hits += 1
                sum_mrr += 1 / (rank + 1)
                sum_ndcg += 1 / math.log2(rank + 2)

        return {
            f'Hit Ratio@{k}': hits / num_users,
            f'MRR@{k}': sum_mrr / num_users,
            f'NDCG@{k}': sum_ndcg / num_users
        }