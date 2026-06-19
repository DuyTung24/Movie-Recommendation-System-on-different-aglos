# src/optimizer.py

import numpy as np

class SGDOptimizer:
    """
    Triển khai thuật toán Stochastic Gradient Descent cho Matrix Factorization.
    """
    def __init__(self, model, learning_rate=0.01, regularization=0.02):
        """
        Args:
            model: Instance của class MatrixFactorizationModel.
            learning_rate (float): Tốc độ học (gamma).
            regularization (float): Hệ số điều tiết (lambda) để tránh overfitting.
        """
        self.model = model
        self.lr = learning_rate
        self.lamda = regularization

    def step(self, u, i, rating):
        """
        Thực hiện cập nhật trọng số cho MỘT sample (u, i, rating).
        """
        # 1. Tính toán dự đoán hiện tại
        prediction = self.model.predict(u, i)
        
        # 2. Tính sai số (Error)
        error = rating - prediction

        # 3. Cập nhật Global Bias (ít khi dùng regularization cho mu)
        self.model.mu += self.lr * error

        # 4. Cập nhật User Bias và Item Bias
        # Công thức: b = b + lr * (error - lamda * b)
        self.model.b_u[u] += self.lr * (error - self.lamda * self.model.b_u[u])
        self.model.b_i[i] += self.lr * (error - self.lamda * self.model.b_i[i])

        # 5. Cập nhật Ma trận Latent Factors P và Q
        # Lưu lại giá trị cũ của P_u để cập nhật Q_i một cách độc lập trong cùng một step
        p_u_old = self.model.P[u, :].copy()
        
        # Công thức: p_u = p_u + lr * (error * q_i - lamda * p_u)
        self.model.P[u, :] += self.lr * (error * self.model.Q[i, :] - self.lamda * self.model.P[u, :])
        
        # Công thức: q_i = q_i + lr * (error * p_u - lamda * q_i)
        self.model.Q[i, :] += self.lr * (error * p_u_old - self.lamda * self.model.Q[i, :])

        return error ** 2

    def calculate_loss(self, data):
        """
        Tính toán tổng lỗi (MSE) trên một tập dữ liệu (train hoặc test).
        """
        total_sq_error = 0
        for u, i, rating in data:
            u, i = int(u), int(i)
            prediction = self.model.predict(u, i)
            total_sq_error += (rating - prediction) ** 2
        
        return total_sq_error / len(data)