import numpy as np

class RecommenderEvaluator:
    @staticmethod
    def build_ranking_data(rate_train, rate_test, n_items):
        """
        Chuẩn bị tập dữ liệu đánh giá Random 80 - 20
        Mỗi user sẽ có một DANH SÁCH các item test (Positives) và các item chưa xem (Negatives).
        """
        user_pos = {}
        user_neg = {}
        all_items = set(range(n_items))
        
        train_dict = {}
        for u, i, _ in rate_train:
            u_idx, i_idx = u - 1, i - 1
            if u_idx not in train_dict:
                train_dict[u_idx] = set()
            train_dict[u_idx].add(i_idx)

        # Ghi nhận TẤT CẢ items mà user đã xem trong tập Test
        test_dict = {}
        for u, i, _ in rate_test:
            u_idx, i_idx = u - 1, i - 1
            if u_idx not in test_dict:
                test_dict[u_idx] = set()
            test_dict[u_idx].add(i_idx)

        for u_idx, pos_items in test_dict.items():
            user_pos[u_idx] = list(pos_items)
            train_items = train_dict.get(u_idx, set())
            
            # Negatives = Toàn bộ item - Item đã xem ở Train - Item sẽ xem ở Test
            candidates = list(all_items - train_items - pos_items)
            user_neg[u_idx] = np.array(candidates)
            
        return user_pos, user_neg

    @staticmethod
    def evaluate_model(model, rate_test, user_pos, user_neg, utility_sparse, utility_norm_sparse, user_means, item_means, K=5):
        print("-> Đang tính toán RMSE và các chỉ số Ranking (Random 80-20)...")
        
        utility_csc = utility_sparse.tocsc() if model.mode == 'user' else None
        utility_norm_csc = utility_norm_sparse.tocsc() if model.mode == 'user' else None

        # 1. RMSE tính trên tập Test
        mse, count = 0, 0
        for u, i, r in rate_test:
            u_idx, i_idx = u - 1, i - 1
            pred_rating = model.predict_for_user_items(
                u_idx, [i_idx], utility_sparse, utility_norm_sparse, user_means, item_means,
                utility_csc=utility_csc, utility_norm_csc=utility_norm_csc
            )[0]
            mse += (pred_rating - r) ** 2
            count += 1
        rmse = np.sqrt(mse / count) if count > 0 else 0
        
        # 2. Tính toán Ranking Metrics
        precisions, recalls, ndcgs, mrrs = [], [], [], []
        
        for u_idx, pos_items in user_pos.items():
            pos_items_arr = np.array(pos_items)
            neg_items = user_neg[u_idx]
            
            items_to_predict = np.concatenate([neg_items, pos_items_arr])
            
            scores = model.predict_for_user_items(
                u_idx, items_to_predict, utility_sparse, utility_norm_sparse, user_means, item_means,
                utility_csc=utility_csc, utility_norm_csc=utility_norm_csc
            )
            
            ranked_indices = items_to_predict[np.argsort(scores)[::-1]]
            top_k = ranked_indices[:K]
            
            # Đếm Hits
            hits = np.intersect1d(top_k, pos_items_arr)
            hit_count = len(hits)
            
            precisions.append(hit_count / K)
            recalls.append(hit_count / len(pos_items_arr))
            
            # Tính MRR (Dựa trên vị trí rank của item đúng đầu tiên)
            ranks = np.where(np.isin(ranked_indices, pos_items_arr))[0] + 1
            if len(ranks) > 0 and ranks[0] <= K:
                mrrs.append(1.0 / ranks[0])
            else:
                mrrs.append(0.0)
                
            # Tính NDCG@K cho nhiều items
            dcg = 0.0
            for i, item in enumerate(top_k):
                if item in pos_items_arr:
                    dcg += 1.0 / np.log2(i + 2)  # Vị trí i=0 tương ứng rank=1
                    
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(pos_items_arr), K)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
                
        return rmse, np.mean(precisions), np.mean(recalls), np.mean(ndcgs), np.mean(mrrs)