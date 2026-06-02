# src/model.py

import numpy as np

class MatrixFactorizationModel:
    """
    Định nghĩa mô hình Matrix Factorization với Biases.
    Dự đoán rating được tính bằng: r_hat = mu + b_u + b_i + P_u * Q_i^T
    """
    def __init__(self, num_users, num_items, num_factors=20):
        """
        Khởi tạo các tham số cho mô hình.
        
        Args:
            num_users (int): Tổng số người dùng.
            num_items (int): Tổng số vật phẩm (phim).
            num_factors (int): Số lượng đặc trưng ẩn (K - latent factors).
        """
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors

        # Khởi tạo ngẫu nhiên các ma trận đặc trưng ẩn (Latent Factors)
        # Sử dụng phân phối chuẩn với độ lệch chuẩn nhỏ để khởi tạo
        self.P = np.random.normal(scale=1./self.num_factors, size=(self.num_users, self.num_factors))
        self.Q = np.random.normal(scale=1./self.num_factors, size=(self.num_items, self.num_factors))

        # Khởi tạo các bias bằng 0
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.mu = 0

    def predict(self, u, i):
        """
        Dự đoán rating cho một cặp (user index, item index).
        Công thức: \hat{r}_{ui} = \mu + b_u + b_i + p_u^T \cdot q_i
        """
        # Kiểm tra index hợp lệ (trong trường hợp dữ liệu test có user/item mới)
        if u < self.num_users and i < self.num_items:
            prediction = self.mu + self.b_u[u] + self.b_i[i] + self.P[u, :].dot(self.Q[i, :].T)
        else:
            # Nếu gặp user/item lạ, trả về global bias (trung bình)
            prediction = self.mu
            
        return prediction

    def get_full_matrix(self):
        """
        Tái tạo lại toàn bộ ma trận Rating (chỉ dùng khi num_users và num_items nhỏ).
        """
        return self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :] + self.P.dot(self.Q.T)

    def save_weights(self, path):
        """Lưu các trọng số sau khi train."""
        np.savez(path, P=self.P, Q=self.Q, b_u=self.b_u, b_i=self.b_i, mu=self.mu)

    def load_weights(self, path):
        """Load trọng số đã lưu."""
        weights = np.load(path)
        self.P = weights['P']
        self.Q = weights['Q']
        self.b_u = weights['b_u']
        self.b_i = weights['b_i']
        # Đảm bảo lấy đúng giá trị số cho mu
        self.mu = weights['mu'].item() if weights['mu'].ndim == 0 else weights['mu']
        print(f"--- Đã tải trọng số từ {path} ---")