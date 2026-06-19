import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

class NCFDataset(Dataset):
    """
    Kế thừa lớp torch.utils.data.Dataset để đóng gói dữ liệu huấn luyện.
    """
    def __init__(self, user_input, item_input, labels):
        self.user_input = torch.tensor(user_input, dtype=torch.long)
        self.item_input = torch.tensor(item_input, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Trả về trực tiếp phần tử từ Tensor có sẵn
        return self.user_input[index], self.item_input[index], self.labels[index]


def load_rating_file_as_matrix(filename, test_ratings=None):
    exclude_set = set()
    if test_ratings is not None:
        for u, items in test_ratings.items():
            for i in items:
                exclude_set.add((u, i))
    
    num_users, num_items = 0, 0
    
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                arr = line.split("::")
                u, i, rating = int(arr[0]), int(arr[1]), float(arr[2])
                
                if rating >= 3.0:  # Chỉ tính những tương tác tích cực vào việc xác định kích thước ma trận
                    num_users = max(num_users, u)
                    num_items = max(num_items, i)
                
    # Tạo ma trận thưa lưu tương tác ẩn kích thước
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    
    # Quét file lần hai để ép toàn bộ điểm số về nhãn Implicit 1.0
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                arr = line.split("::")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                
                if rating < 3.0:
                    continue
                    
                if (user, item) in exclude_set:
                    continue
                #chỉ khi có rating >= 3.0 sao thì mới tính là một positive implicit rating
                mat[user, item] = 1.0
                
    return mat


def get_train_instances(train_matrix, num_negatives):
    """
    Lấy mẫu âm động.
    """
    user_input = []
    item_input = []
    labels = []
    
    num_users, num_items = train_matrix.shape
    train_matrix_csr = train_matrix.tocsr()
    
    for u in range(num_users):
        # Trích xuất toàn bộ sản phẩm tương tác dương của User u
        pos_items = train_matrix_csr[u].indices
        num_pos = len(pos_items)
        if num_pos == 0:
            continue
            
        # Tích hợp nhanh toàn bộ khối mẫu dương (Nhãn 1.0)
        user_input.extend([u] * num_pos)
        item_input.extend(pos_items)
        labels.extend([1.0] * num_pos)
        
        num_neg_needed = num_pos * num_negatives
        pos_set = set(pos_items)
        
        neg_candidates = np.random.randint(0, num_items, size=num_neg_needed * 2)
        filtered_negs = [j for j in neg_candidates if j not in pos_set]
        
        # Vòng lặp dự phòng khi mảng bốc ngẫu nhiên bị trùng lặp nhiều
        while len(filtered_negs) < num_neg_needed:
            extra_candidates = np.random.randint(0, num_items, size=num_neg_needed)
            filtered_negs.extend([j for j in extra_candidates if j not in pos_set])
            
        chosen_negs = filtered_negs[:num_neg_needed]
        
        # Tích hợp toàn bộ khối mẫu âm (Nhãn 0.0)
        user_input.extend([u] * num_neg_needed)
        item_input.extend(chosen_negs)
        labels.extend([0.0] * num_neg_needed)
        
    return np.array(user_input), np.array(item_input), np.array(labels)