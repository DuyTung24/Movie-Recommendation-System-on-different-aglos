import numpy as np

class RecommenderEvaluator:
    @staticmethod
    def build_ranking_data(rate_train, rate_test, n_items, n_neg=99):
        """
        Chuẩn bị tập dữ liệu đánh giá Leave-One-Out phối hợp với 99 negative samples ngẫu nhiên.
        """
        user_pos = {}
        user_neg = {}
        all_items = set(range(n_items))
        
        # Tạo bản đồ các item đã xem trong tập train để tìm nhanh candidates
        train_dict = {}
        for u, i, _ in rate_train:
            u_idx = u - 1
            i_idx = i - 1
            if u_idx not in train_dict:
                train_dict[u_idx] = set()
            train_dict[u_idx].add(i_idx)

        for u, i, _ in rate_test:
            u_idx = u - 1
            i_idx = i - 1
            
            user_pos[u_idx] = i_idx
            train_items = train_dict.get(u_idx, set())
            
            # Lấy các item chưa từng tương tác trong tập train và không trùng với mẫu test thực tế
            candidates = list(all_items - train_items - {i_idx})
            if len(candidates) >= n_neg:
                neg_items = np.random.choice(candidates, n_neg, replace=False)
            else:
                neg_items = np.array(candidates)
                
            user_neg[u_idx] = neg_items
            
        return user_pos, user_neg

    @staticmethod
    def evaluate_model(model, rate_test, user_pos, user_neg, utility_sparse, utility_norm_sparse, user_means, K=5):
        """
        Đánh giá toàn diện mô hình bằng các phép tính lướt qua dữ liệu thưa, tránh dựng ma trận kết quả dense.
        """
        print("-> Đang tính toán các chỉ số đánh giá (Metrics) trên kiến trúc On-Demand...")
        
        # Ép kiểu CSC trước một lần duy nhất nếu chạy User-Based nhằm tránh thắt nút cổ chai hiệu năng
        utility_csc = utility_sparse.tocsc() if model.mode == 'user' else None
        utility_norm_csc = utility_norm_sparse.tocsc() if model.mode == 'user' else None

        # 1. Tính toán RMSE trên tập Test thực tế
        mse = 0
        count = 0
        for u, i, r in rate_test:
            u_idx = u - 1
            i_idx = i - 1
            pred_rating = model.predict_for_user_items(
                u_idx, [i_idx], utility_sparse, utility_norm_sparse, user_means,
                utility_csc=utility_csc, utility_norm_csc=utility_norm_csc
            )[0]
            mse += (pred_rating - r) ** 2
            count += 1
        rmse = np.sqrt(mse / count) if count > 0 else 0
        
        # 2. Tính toán Ranking Metrics (Precision@K, Recall@K, NDCG@K, MRR@K)
        precisions, recalls, ndcgs, mrrs = [], [], [], []
        
        for u_idx in user_pos:
            pos_item = user_pos[u_idx]
            neg_items = user_neg[u_idx]
            
            # Gom cụm 100 items (1 điểm Positive và 99 điểm Negative)
            items_to_predict = np.append(neg_items, pos_item)
            
            # Thực hiện dự đoán song song cho cụm 100 ứng viên này
            scores = model.predict_for_user_items(
                u_idx, items_to_predict, utility_sparse, utility_norm_sparse, user_means,
                utility_csc=utility_csc, utility_norm_csc=utility_norm_csc
            )
            
            # Đọc ra chỉ số sắp xếp giảm dần theo điểm số dự báo
            ranked_indices = items_to_predict[np.argsort(scores)[::-1]]
            top_k = ranked_indices[:K]
            
            # Đếm Hits
            hit = int(pos_item in top_k)
            precisions.append(hit / K)
            recalls.append(hit)
            
            # Tìm vị trí xếp hạng chính xác của điểm Positive thực tế
            rank_positions = np.where(ranked_indices == pos_item)[0] + 1
            rank = rank_positions[0] if len(rank_positions) > 0 else len(ranked_indices) + 1
            
            if rank <= K:
                ndcgs.append(1.0 / np.log2(rank + 1))
                mrrs.append(1.0 / rank)
            else:
                ndcgs.append(0.0)
                mrrs.append(0.0)
                
        return rmse, np.mean(precisions), np.mean(recalls), np.mean(ndcgs), np.mean(mrrs)