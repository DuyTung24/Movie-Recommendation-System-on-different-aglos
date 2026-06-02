import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MemoryBasedCF:
    def __init__(self, mode='item', n_neighbors=30):
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.similarity_matrix = None

    def compute_similarity(self, utility_norm_sparse):
        print(f"-> Đang tính toán ma trận tương đồng Cosine cho chế độ: {self.mode.upper()}...")
        if self.mode == 'item':
            # Item-based: Tính độ tương đồng giữa các cột (Items)
            sim = cosine_similarity(utility_norm_sparse.T)
        else:
            # User-based: Tính độ tương đồng giữa các hàng (Users)
            sim = cosine_similarity(utility_norm_sparse)
            
        np.fill_diagonal(sim, -np.inf)  # Loại bỏ việc tự tương đồng với chính mình
        self.similarity_matrix = sim.astype(np.float32)
        print("-> Tính toán ma trận tương đồng hoàn tất.")
        return self.similarity_matrix

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'mode': self.mode,
            'n_neighbors': self.n_neighbors,
            'similarity_matrix': self.similarity_matrix
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"-> Đã lưu trọng số tương đồng vào file: {path}")

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.mode = checkpoint['mode']
        self.n_neighbors = checkpoint['n_neighbors']
        self.similarity_matrix = checkpoint['similarity_matrix']
        print(f"-> Đã tải thành công trọng số tương đồng từ file: {path}")
        return True

    def predict_for_user_items(self, u_idx, item_indices, utility_sparse, utility_norm_sparse, user_means, utility_csc=None, utility_norm_csc=None):
        """
        Ý TƯỞNG 3: Dự đoán Rating theo yêu cầu (On-demand/Lazy Prediction).
        Chỉ tính toán rating cho danh sách item cụ thể của user cụ thể để tối ưu bộ nhớ tuyệt đối.
        """
        item_indices = np.atleast_1d(item_indices)
        preds = np.zeros(len(item_indices), dtype=np.float32)
        
        if self.mode == 'item':
            # Lấy các item mà user u_idx ĐÃ rate trong tập huấn luyện
            start_idx = utility_sparse.indptr[u_idx]
            end_idx = utility_sparse.indptr[u_idx + 1]
            
            rated_items = utility_sparse.indices[start_idx:end_idx]
            rated_vals_norm = utility_norm_sparse.data[start_idx:end_idx]
            
            if len(rated_items) == 0:
                return np.full(len(item_indices), user_means[u_idx], dtype=np.float32)
                
            # Trích xuất độ tương đồng giữa các item cần dự đoán với các item đã xem
            sim_slice = self.similarity_matrix[item_indices[:, None], rated_items]
            
            # Ý TƯỞNG 4: Vector hóa việc chọn K lân cận lớn nhất cùng lúc
            k = min(self.n_neighbors, len(rated_items))
            top_k_indices = np.argsort(sim_slice, axis=1)[:, -k:]
            
            for idx in range(len(item_indices)):
                k_neighbors = top_k_indices[idx]
                top_sim = sim_slice[idx, k_neighbors]
                
                denom = np.sum(np.abs(top_sim)) + 1e-8
                num = np.dot(top_sim, rated_vals_norm[k_neighbors])
                preds[idx] = (num / denom) + user_means[u_idx]
                
        else: # User-based
            sim_vec = self.similarity_matrix[u_idx]
            
            # Tối ưu hóa: Truy vấn cột nhanh bằng định dạng CSC (chỉ chuyển đổi 1 lần bên ngoài)
            if utility_csc is None:
                utility_csc = utility_sparse.tocsc()
            if utility_norm_csc is None:
                utility_norm_csc = utility_norm_sparse.tocsc()
                
            for idx, j in enumerate(item_indices):
                start_idx = utility_csc.indptr[j]
                end_idx = utility_csc.indptr[j + 1]
                
                users_rated = utility_csc.indices[start_idx:end_idx]
                users_vals_norm = utility_norm_csc.data[start_idx:end_idx]
                
                if len(users_rated) == 0:
                    preds[idx] = user_means[u_idx]
                    continue
                    
                sim_slice = sim_vec[users_rated]
                k = min(self.n_neighbors, len(users_rated))
                top_k_idx = np.argsort(sim_slice)[-k:]
                
                top_sim = sim_slice[top_k_idx]
                denom = np.sum(np.abs(top_sim)) + 1e-8
                num = np.dot(top_sim, users_vals_norm[top_k_idx])
                preds[idx] = (num / denom) + user_means[u_idx]
                
        return np.clip(preds, 1, 5)