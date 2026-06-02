import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Model cho Hệ thống gợi ý.
    Ánh xạ tư tưởng từ bài báo Neural Collaborative Filtering (WWW 2017).
    """
    def __init__(self, num_users, num_items, layers=[64, 32, 16, 8]):
        """
        Khởi tạo mạng MLP.
        - num_users: Tổng số lượng người dùng.
        - num_items: Tổng số lượng sản phẩm.
        - layers: Mảng cấu trúc các tầng ẩn. 
                  Lưu ý: layers[0] chính là tổng kích thước của (user_embedding + item_embedding).
                  => Do đó, kích thước embedding thực tế của mỗi bên sẽ là layers[0] // 2.
        """
        super(MLP, self).__init__()
        
        # 1. Khởi tạo tầng nhúng (Embeddings)
        # Kích thước embedding = 64 // 2 = 32
        embed_size = layers[0] // 2
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.item_embedding = nn.Embedding(num_items, embed_size)
        
        # 2. Xây dựng động (Dynamic) cấu trúc các tầng ẩn MLP dạng tháp (Tower Pattern)
        mlp_modules = []
        for i in range(len(layers) - 1):
            input_size = layers[i]
            output_size = layers[i + 1]
            
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU()) # Hàm kích hoạt phi tuyến ReLU
            
        # Đóng gói danh sách các tầng lại thành một mạng Sequential hoàn chỉnh
        self.mlp_network = nn.Sequential(*mlp_modules)
        
        # 3. Tầng tuyến tính dự đoán đầu ra (từ tầng ẩn cuối cùng -> 1 giá trị)
        self.prediction_layer = nn.Linear(layers[-1], 1)
        
        # 4. Hàm kích hoạt Sigmoid đưa điểm số về xác suất [0, 1] cho Implicit Feedback
        self.sigmoid = nn.Sigmoid()
        
        # 5. Kích hoạt khởi tạo trọng số
        self._init_weights()

    def _init_weights(self):
        """
        Khởi tạo trọng số ngẫu nhiên ban đầu để tối ưu hóa quá trình học.
        """
        # Embeddings: Phân phối chuẩn (Normal distribution) với std=0.01 như GMF
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # Các tầng ẩn MLP: Do sử dụng hàm kích hoạt ReLU, việc khởi tạo He/Kaiming Uniform
        # là bắt buộc để tránh hiện tượng triệt tiêu đạo hàm (Vanishing Gradient).
        for layer in self.mlp_network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
        # Tầng dự đoán đầu ra - Chuẩn hóa lại cho giống paper
        nn.init.normal_(self.prediction_layer.weight, std=0.01)
        if self.prediction_layer.bias is not None:
            nn.init.zeros_(self.prediction_layer.bias)

    def forward(self, user_indices, item_indices):
        """
        Luồng lan truyền tiến (Forward Pass).
        """
        # Bước A: Trích xuất vector đặc trưng từ tầng nhúng (shape: [batch_size, embed_size])
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        # Bước B: Ghép chuỗi (Concatenation) hai vector lại với nhau.
        # Đây là điểm KHÁC BIỆT CỐT LÕI so với GMF (dùng phép nhân).
        # Kết quả sẽ có kích thước: embed_size + embed_size = layers[0]
        vector_concat = torch.cat([user_embed, item_embed], dim=-1)
        
        # Bước C: Đẩy vector đã ghép qua toàn bộ cấu trúc tháp MLP (gồm các tầng Linear + ReLU)
        hidden_output = self.mlp_network(vector_concat)
        
        # Bước D: Đẩy qua tầng cuối cùng tính Logit, bọc Sigmoid và làm phẳng (Flatten) mảng
        logits = self.prediction_layer(hidden_output)
        output = self.sigmoid(logits)
        
        return output.view(-1)