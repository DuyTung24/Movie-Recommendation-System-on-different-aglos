import torch
import numpy as np

def evaluate_model(model, test_ratings, test_negatives, K_list, device):
    """
    Hàm đánh giá hiệu năng mô hình theo chiến lược RANDOM SPLIT 80/20.
    Tận dụng tính toán song song xử lý Batch để đo đạc chính xác Precision, Recall, NDCG, MRR.
    """ 
    model.eval()
    
    # Lấy danh sách các user có phát sinh tương tác trong tập Test
    eval_users = list(test_ratings.keys())
    num_users = len(eval_users)
    
    if hasattr(model, 'num_items'):
        num_items = model.num_items
    elif hasattr(model, 'item_embedding'):
        num_items = model.item_embedding.num_embeddings
    elif hasattr(model, 'item_embedding_gmf'):
        num_items = model.item_embedding_gmf.num_embeddings
    else:
        num_items = max([max(items) for items in test_ratings.values()]) + 1

    # các chỉ số
    sum_metrics = {K: {'precision': 0, 'recall': 0, 'ndcg': 0, 'mrr': 0} for K in K_list}
    valid_user_count = 0
    user_batch_size = 512

    with torch.no_grad():
        for start_u in range(0, num_users, user_batch_size):
            end_u = min(start_u + user_batch_size, num_users)
            batch_user_ids = eval_users[start_u:end_u]
            curr_batch_size = len(batch_user_ids)
            
            user_tensor = torch.tensor(batch_user_ids, dtype=torch.long).to(device).unsqueeze(1)
            user_tensor = user_tensor.expand(-1, num_items).reshape(-1)
            item_tensor = torch.arange(num_items, dtype=torch.long).to(device).unsqueeze(0)
            item_tensor = item_tensor.expand(curr_batch_size, -1).reshape(-1)

            predictions = model(user_tensor, item_tensor)
            predictions = predictions.view(curr_batch_size, num_items)
            
            for i, u in enumerate(batch_user_ids):
                if u in test_negatives:
                    train_items_tensor = torch.tensor(test_negatives[u], dtype=torch.long).to(device)
                    predictions[i, train_items_tensor] = -1e9
                    
            #Tìm Top-K Items lớn nhất
            max_K = max(K_list)
            _, top_k_indices = torch.topk(predictions, max_K, dim=1)
            top_k_indices = top_k_indices.cpu().numpy()
            
            #Tính toán các Metric
            for i, u in enumerate(batch_user_ids):
                gt_items = set(test_ratings[u])
                num_gt = len(gt_items)
                if num_gt == 0: continue
                
                valid_user_count += 1
                top_k_list = top_k_indices[i]
                
                for K in K_list:
                    top_k_preds = top_k_list[:K]
                    hits = [1 if item in gt_items else 0 for item in top_k_preds]
                    
                    # Precision & Recall
                    hit_count = sum(hits)
                    sum_metrics[K]['precision'] += hit_count / K
                    sum_metrics[K]['recall'] += hit_count / num_gt
                    
                    # MRR
                    mrr = 0.0
                    for rank_idx, is_hit in enumerate(hits):
                        if is_hit == 1:
                            mrr = 1.0 / (rank_idx + 1)
                            break
                    sum_metrics[K]['mrr'] += mrr
                    
                    # NDCG tính theo Ideal DCG
                    dcg = 0.0
                    for rank_idx, is_hit in enumerate(hits):
                        if is_hit == 1:
                            dcg += 1.0 / np.log2(rank_idx + 2)
                            
                    idcg = 0.0
                    for rank_idx in range(min(num_gt, K)):
                        idcg += 1.0 / np.log2(rank_idx + 2)
                        
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    sum_metrics[K]['ndcg'] += ndcg
                    
    # Trung bình hóa kết quả cho Output
    final_metrics = {}
    for K in K_list:
        final_metrics[K] = {
            'precision': sum_metrics[K]['precision'] / valid_user_count,
            'recall': sum_metrics[K]['recall'] / valid_user_count,
            'ndcg': sum_metrics[K]['ndcg'] / valid_user_count,
            'mrr': sum_metrics[K]['mrr'] / valid_user_count
        }
        
    return final_metrics