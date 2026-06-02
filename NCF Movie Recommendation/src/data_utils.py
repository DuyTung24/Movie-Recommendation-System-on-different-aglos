import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

class NCFDataset(Dataset):
    """
    Kế thừa lớp torch.utils.data.Dataset để đóng gói dữ liệu huấn luyện.
    ĐÃ TỐI ƯU: Ép kiểu Tensor một lần duy nhất tại __init__ để giải phóng áp lực tính toán cho Multi-workers.
    """
    def __init__(self, user_input, item_input, labels):
        # Chuyển toàn bộ mảng NumPy sang PyTorch Tensor ngay từ đầu
        self.user_input = torch.tensor(user_input, dtype=torch.long)
        self.item_input = torch.tensor(item_input, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Trả về trực tiếp phần tử từ Tensor có sẵn, tốc độ đọc luồng cực nhanh
        return self.user_input[index], self.item_input[index], self.labels[index]


def load_rating_file_as_matrix(filename, test_ratings=None):
    """
    Đọc tệp dữ liệu huấn luyện (chuẩn định dạng UserID::MovieID::Rating::Timestamp).
    Chuyển đổi dữ liệu từ Explicit (1-5 sao) về Implicit Feedback (0 và 1) 
    và dựng thành ma trận thưa DOK (Dictionary of Keys).
    """

    exclude_set = set()
    if test_ratings is not None:
        # Chuyển test_ratings thành tập hợp các cặp (user, item) để tra cứu O(1)
        for u, i in test_ratings:
            exclude_set.add((u, i))
    
    num_users, num_items = 0, 0
    
    # Bước 1: Quét file lần một dựa theo dấu phân tách '::' để tìm ID lớn nhất của User và Item
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                arr = line.split("::")
                u, i, rating = int(arr[0]), int(arr[1]), float(arr[2])
                
                if rating >= 3.0:  # Chỉ tính những tương tác tích cực vào việc xác định kích thước ma trận
                    num_users = max(num_users, u)
                    num_items = max(num_items, i)
                
    # Tạo ma trận thưa lưu tương tác ẩn kích thước (num_users + 1, num_items + 1)
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    
    # Bước 2: Quét file lần hai để ép toàn bộ điểm số về nhãn Implicit 1.0
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                arr = line.split("::")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                
                if rating < 3.0:
                    continue
                    
                if (user, item) in exclude_set:
                    continue
                
                # BẤT KỂ số sao người dùng đánh giá (arr[2] từ 1 đến 5 sao),
                # cứ có lịch sử xuất hiện tương tác thì được gán nhãn cố định là 1.0 (Implicit Positive)
                mat[user, item] = 1.0
                
    return mat


def get_train_instances(train_matrix, num_negatives):
    """
    Thuật toán Lấy mẫu âm động (Dynamic Negative Sampling) SIÊU TỐC ĐỘ.
    Chuyển dịch sang cấu trúc ma trận dòng CSR và ép khối mảng qua NumPy,
    triệt tiêu hoàn toàn độ trễ đơn nhân CPU trước mỗi Epoch.
    """
    user_input = []
    item_input = []
    labels = []
    
    num_users, num_items = train_matrix.shape
    # Chuyển đổi sang ma trận thưa định dạng CSR để cắt lát dòng (User slicing) cực nhanh
    train_matrix_csr = train_matrix.tocsr()
    
    for u in range(num_users):
        # Trích xuất toàn bộ sản phẩm tương tác dương của User u bằng C-level speed
        pos_items = train_matrix_csr[u].indices
        num_pos = len(pos_items)
        if num_pos == 0:
            continue
            
        # 1. Tích hợp nhanh toàn bộ khối mẫu dương (Nhãn 1.0)
        user_input.extend([u] * num_pos)
        item_input.extend(pos_items)
        labels.extend([1.0] * num_pos)
        
        # 2. Xử lý lấy mẫu âm hàng loạt (Vectorized Negative Sampling)
        num_neg_needed = num_pos * num_negatives
        pos_set = set(pos_items) # Dùng set() để tối ưu tốc độ tìm kiếm O(1)
        
        # Sinh dư lượng mẫu ngẫu nhiên dạng mảng NumPy để lọc bộ lọc nhanh
        neg_candidates = np.random.randint(0, num_items, size=num_neg_needed * 2)
        filtered_negs = [j for j in neg_candidates if j not in pos_set]
        
        # Vòng lặp dự phòng cực hiếm khi mảng bốc ngẫu nhiên bị trùng lặp nhiều
        while len(filtered_negs) < num_neg_needed:
            extra_candidates = np.random.randint(0, num_items, size=num_neg_needed)
            filtered_negs.extend([j for j in extra_candidates if j not in pos_set])
            
        chosen_negs = filtered_negs[:num_neg_needed]
        
        # Tích hợp nhanh toàn bộ khối mẫu âm (Nhãn 0.0)
        user_input.extend([u] * num_neg_needed)
        item_input.extend(chosen_negs)
        labels.extend([0.0] * num_neg_needed)
        
    return np.array(user_input), np.array(item_input), np.array(labels)