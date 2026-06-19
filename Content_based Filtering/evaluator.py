import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_rmse(model, test_df, item_features):
    predictions, truths = [], []
    for _, row in test_df.iterrows():
        u = int(row['u_idx'])
        m = int(row['m_idx'])
        truths.append(row['rating'])
        pred = model.predict(u, [m], item_features)[0]
        predictions.append(pred)
    return np.sqrt(mean_squared_error(truths, predictions))

def evaluate_top_n(model, train_df, test_df, item_features, n_items, k=5):
    precisions, recalls, ndcgs, mrrs = [], [], [], []
    
    # 1. Tạo dict các item user đã xem trong quá khứ (train) và ground truth (test)
    train_interacted = train_df.groupby('u_idx')['m_idx'].apply(set).to_dict()
    test_interacted = test_df.groupby('u_idx')['m_idx'].apply(set).to_dict()
    
    all_items = np.arange(n_items)
    
    # Chỉ đánh giá trên những user hợp lệ (có dữ liệu ở cả train và test)
    valid_users = set(train_interacted.keys()).intersection(set(test_interacted.keys()))
    
    for u in valid_users:
        user_train_items = train_interacted[u]
        ground_truth = test_interacted[u]
        
        # 2. Tập ứng viên: toàn bộ items chưa từng xem trong Train
        candidates = np.setdiff1d(all_items, list(user_train_items))
        
        if len(candidates) == 0:
            continue
            
        # 3. Dự đoán điểm cho tất cả ứng viên của user này trong 1 thao tác vector
        preds = model.predict(u, candidates, item_features)
        
        # 4. Lấy index của Top K item có điểm cao nhất
        # np.argsort sắp xếp tăng dần, [-k:] lấy K phần tử cuối, [::-1] đảo ngược thành giảm dần
        top_k_indices = np.argsort(preds)[-k:][::-1]
        top_k_items = candidates[top_k_indices]
        
        # 5. Kiểm tra Hit (Item được recommend có nằm trong Ground Truth của User không)
        hits = [1 if item in ground_truth else 0 for item in top_k_items]
        hits_count = sum(hits)
        
        # Tính toán Metrics
        precisions.append(hits_count / k)
        recalls.append(hits_count / len(ground_truth))
        
        # Tính NDCG@K
        dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(k, len(ground_truth)))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
        
        # Tính MRR@K (Chỉ quan tâm đến thứ hạng của item đúng đầu tiên)
        mrr = 0
        for idx, hit in enumerate(hits):
            if hit == 1:
                mrr = 1 / (idx + 1)
                break
        mrrs.append(mrr)
        
    return {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"NDCG@{k}": np.mean(ndcgs),
        f"MRR@{k}": np.mean(mrrs)
    }