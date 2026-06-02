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

def evaluate_top_n(model, train_df, test_df, item_features, n_items, k=5, n_negatives=99):
    hits, ndcgs, mrrs = [], [], []
    
    # Lấy danh sách item đã tương tác để tránh lấy nhầm vào tập negative
    interacted = train_df.groupby('u_idx')['m_idx'].apply(set).to_dict()
    all_items = np.arange(n_items)
    
    for _, row in test_df.iterrows():
        u = int(row['u_idx'])
        test_item = int(row['m_idx'])
        
        user_interacted = interacted.get(u, set())
        user_interacted.add(test_item)
        
        # Sampling 99 negative items
        candidates = np.setdiff1d(all_items, list(user_interacted))
        if len(candidates) >= n_negatives:
            neg_samples = np.random.choice(candidates, size=n_negatives, replace=False)
        else:
            neg_samples = candidates
            
        items_to_eval = np.append(neg_samples, test_item)
        
        # Dự đoán rating cho 100 items này
        preds = model.predict(u, items_to_eval, item_features)
        
        # Lấy rank của các items (sắp xếp giảm dần)
        rank_indices = np.argsort(preds)[::-1]
        ranked_items = items_to_eval[rank_indices]
        
        # Vị trí của test_item sau khi xếp hạng (1-based)
        rank = np.where(ranked_items == test_item)[0][0] + 1
        
        # Tính toán metrics
        hit = 1 if rank <= k else 0
        hits.append(hit)
        
        ndcgs.append(1 / np.log2(rank + 1) if rank <= k else 0)
        mrrs.append(1 / rank if rank <= k else 0)
        
    return {
        f"Precision@{k}": np.mean(hits) / k,
        f"Recall@{k}": np.mean(hits), # Vì Leave-one-out chỉ có 1 ground-truth
        f"NDCG@{k}": np.mean(ndcgs),
        f"MRR@{k}": np.mean(mrrs)
    }