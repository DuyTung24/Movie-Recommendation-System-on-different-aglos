import torch
import numpy as np

def evaluate_model(model, test_ratings, test_negatives, K_list, device):
    """
    Hàm đánh giá hiệu năng mô hình đã được VECTOR HÓA HOÀN TOÀN (Fully Vectorized).
    Loại bỏ hoàn toàn vòng lặp tuần tự từng user, tận dụng tối đa sức mạnh tính toán song song của CUDA/C++.
    """
    # Chuyển mô hình sang chế độ Đánh giá (Tắt Dropout, Batch Norm)
    model.eval()
    
    num_users = len(test_ratings)
    
    # =====================================================================
    # BƯỚC 1: GỘP KHỐI DỮ LIỆU TOÀN BỘ USERS (BATCHING ALL USERS AT ONCE)
    # =====================================================================
    user_inputs = []
    item_inputs = []
    
    for idx in range(num_users):
        user_id = test_ratings[idx][0]
        gt_item = test_ratings[idx][1]      # Mẫu dương thực tế (Ground Truth)
        neg_items = test_negatives[idx]     # 99 mẫu âm của user này
        
        # Ghép mẫu dương vào đầu danh sách 99 mẫu âm -> 100 sản phẩm cần xếp hạng
        items = [gt_item] + list(neg_items)
        
        user_inputs.extend([user_id] * len(items))
        item_inputs.extend(items)
        
    # Ép thành PyTorch Tensor một lần duy nhất rồi đẩy thẳng lên GPU (hoặc CPU)
    user_tensor = torch.tensor(user_inputs, dtype=torch.long, device=device)
    item_tensor = torch.tensor(item_inputs, dtype=torch.long, device=device)
    
    # =====================================================================
    # BƯỚC 2: DỰ ĐOÁN ĐỒNG LOẠT TRÊN GPU (ONE-SHOT FORWARD PASS)
    # =====================================================================
    with torch.no_grad():
        # Đẩy toàn bộ 6,040 users x 100 items = 604,000 cặp qua mô hình ĐÚNG 1 LẦN
        predictions = model(user_tensor, item_tensor)
        
        # Nắn dòng Tensor (Reshape) về ma trận dạng (Số lượng User, 100)
        # Điểm đặc biệt: Phần tử đầu tiên (cột index 0) của mỗi dòng luôn là điểm của mẫu Dương (GT)
        predictions = predictions.view(num_users, 100)
        
        # Trích xuất điểm số của các mẫu dương: shape (num_users, 1)
        pos_scores = predictions[:, 0].unsqueeze(1)
        
        # THUẬT TOÁN VECTOR XẾP HẠNG (SIÊU TỐC ĐỘ):
        # Đếm xem có bao nhiêu phần tử trong số 100 items có điểm số LỚN HƠN điểm của mẫu dương.
        # Vị trí thứ hạng (1-indexed Rank) = (Số món lớn hơn điểm mẫu dương) + 1
        # Ví dụ: Không có món nào lớn hơn điểm mẫu dương -> Rank = 0 + 1 = 1 (Đứng đầu bảng Top-1)
        rank = (predictions > pos_scores).sum(dim=1) + 1
        
    # =====================================================================
    # BƯỚC 3: TÍNH TOÁN CÁC CHỈ SỐ SONG SONG QUA TENSORES (VECTORIZED METRICS)
    # =====================================================================
    final_metrics = {}
    
    for K in K_list:
        # Tạo mặt nạ kiểm tra (Mask): Nếu Rank <= K thì nhận 1.0 (Hit), ngược lại nhận 0.0 (Miss)
        hit = (rank <= K).float()
        
        # Tính toán trung bình toàn bộ ma trận ngay trên nền tảng tính toán lõi cứng
        recall = hit.mean().item()
        precision = (hit / K).mean().item()
        
        # NDCG = 1.0 / log2(Rank + 1) áp dụng cho các User có 'hit'
        ndcg_tensor = hit / torch.log2(rank.float() + 1.0)
        ndcg = ndcg_tensor.mean().item()
        
        # MRR = 1.0 / Rank áp dụng cho các User có 'hit'
        mrr_tensor = hit / rank.float()
        mrr = mrr_tensor.mean().item()
        
        # Lưu kết quả trả về đúng định dạng cũ để không làm lỗi file trainer.py
        final_metrics[K] = {
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg,
            'mrr': mrr
        }
        
    return final_metrics