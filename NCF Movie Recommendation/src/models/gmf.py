import torch
import torch.nn as nn

class GMF(nn.Module):
    """
    Generalized Matrix Factorization (GMF) Model
    Kế thừa nn.Module của PyTorch, ánh tác ý tưởng cốt lõi từ paper Neural Collaborative Filtering (WWW 2017).
    """
    def __init__(self, num_users, num_items, num_factors):
        """
        num_users: Tổng số lượng người dùng trong hệ thống (Hệ số kích thước không gian nhúng)
        num_items: Tổng số lượng sản phẩm/phim trong hệ thống
        num_factors: Kích thước của không gian ẩn (Embedding size - Latent dimensions)
        """
        super(GMF, self).__init__()
        
        # 1. Khởi tạo tầng Không gian nhúng ẩn (Latent Embeddings) cho User và Item
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)
        
        # 2. Tầng Tuyến tính Dự đoán (Prediction Layer) để ánh xạ từ vector tích sang 1 giá trị score duy nhất
        self.prediction_layer = nn.Linear(num_factors, 1)
        
        # 3. Hàm kích hoạt Hàm mũ Sigmoid để chuẩn hóa đầu ra về khoảng xác suất [0, 1]
        self.sigmoid = nn.Sigmoid()
        
        # Kích hoạt hàm khởi tạo trọng số mặc định
        self._init_weights()

    def _init_weights(self):
        """
        Khởi tạo trọng số ngẫu nhiên ban đầu cho các tham số mạng.
        """
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        nn.init.normal_(self.prediction_layer.weight, std=0.01)
        if self.prediction_layer.bias is not None:
            nn.init.zeros_(self.prediction_layer.bias)

    def forward(self, user_indices, item_indices):
        """
        Luồng lan truyền tiến (Forward Pass) của mạng GMF.
        user_indices: Tensor chứa danh sách các ID của người dùng dạng Batch
        item_indices: Tensor chứa danh sách các ID của sản phẩm dạng Batch
        """
        # Trích xuất vector ẩn từ tầng nhúng dựa trên ID đầu vào
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        # Thực hiện phép nhân từng phần tử
        # Đây chính là lõi mở rộng Generalized Matrix Factorization thay thế cho phép nhân vô hướng Inner Product truyền thống
        element_product = torch.mul(user_embed, item_embed)
        
        # Đẩy qua tầng dự đoán để áp công thức tính tổng trọng số tuyến tính h^T
        logits = self.prediction_layer(element_product)
        
        # Áp dụng hàm Sigmoid
        output = self.sigmoid(logits)
        return output.view(-1)