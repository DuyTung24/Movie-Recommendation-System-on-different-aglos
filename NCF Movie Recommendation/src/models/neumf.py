import torch
import torch.nn as nn

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF)
    Mô hình kết hợp sức mạnh của GMF (tuyến tính) và MLP (phi tuyến sâu).
    """
    def __init__(self, num_users, num_items, num_factors=8, layers=[64, 32, 16, 8]):
        super(NeuMF, self).__init__()
        
        # ==========================================
        # NHÁNH 1: KHỞI TẠO EMBEDDING CHO GMF
        # ==========================================
        self.user_embedding_gmf = nn.Embedding(num_users, num_factors)
        self.item_embedding_gmf = nn.Embedding(num_items, num_factors)
        
        # ==========================================
        # NHÁNH 2: KHỞI TẠO EMBEDDING VÀ TẦNG ẨN CHO MLP
        # ==========================================
        embed_size_mlp = layers[0] // 2
        self.user_embedding_mlp = nn.Embedding(num_users, embed_size_mlp)
        self.item_embedding_mlp = nn.Embedding(num_items, embed_size_mlp)
        
        mlp_modules = []
        for i in range(len(layers) - 1):
            mlp_modules.append(nn.Linear(layers[i], layers[i+1]))
            mlp_modules.append(nn.ReLU())
        self.mlp_network = nn.Sequential(*mlp_modules)
        
        # ==========================================
        # NHÁNH 3: TẦNG DỰ ĐOÁN HỢP NHẤT TỔNG (FUSION LAYER)
        # ==========================================
        # Kích thước đầu vào = kích thước đầu ra GMF (num_factors) + kích thước đầu ra MLP (layers[-1])
        self.prediction_layer = nn.Linear(num_factors + layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
        # Khởi tạo trọng số mặc định (dùng khi KHÔNG bật pretrain)
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số khi train NeuMF từ đầu (From scratch)"""
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
        for layer in self.mlp_network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
        nn.init.kaiming_uniform_(self.prediction_layer.weight, a=1)
        if self.prediction_layer.bias is not None:
            nn.init.zeros_(self.prediction_layer.bias)

    def forward(self, user_indices, item_indices):
        """Luồng lan truyền tiến song song"""
        # --- Xử lý nhánh GMF ---
        user_gmf = self.user_embedding_gmf(user_indices)
        item_gmf = self.item_embedding_gmf(item_indices)
        vector_gmf = torch.mul(user_gmf, item_gmf) # Nhân Element-wise
        
        # --- Xử lý nhánh MLP ---
        user_mlp = self.user_embedding_mlp(user_indices)
        item_mlp = self.item_embedding_mlp(item_indices)
        vector_mlp = torch.cat([user_mlp, item_mlp], dim=-1) # Ghép chuỗi Concatenate
        vector_mlp = self.mlp_network(vector_mlp) # Đẩy qua các tầng ẩn ReLU
        
        # --- Hợp nhất hai nhánh ---
        vector_fusion = torch.cat([vector_gmf, vector_mlp], dim=-1)
        
        # --- Đưa ra dự đoán ---
        logits = self.prediction_layer(vector_fusion)
        output = self.sigmoid(logits)
        
        return output.view(-1)

    def load_pretrain_weights(self, gmf_model, mlp_model):
        """
        Nạp trọng số Pre-train từ mô hình GMF và MLP đã hội tụ.
        Đây là kỹ thuật mấu chốt để NeuMF đạt hiệu năng tối đa (state-of-the-art).
        """
        # 1. Nạp trọng số Embedding cho nhánh GMF
        self.user_embedding_gmf.weight.data.copy_(gmf_model.user_embedding.weight.data)
        self.item_embedding_gmf.weight.data.copy_(gmf_model.item_embedding.weight.data)
        
        # 2. Nạp trọng số Embedding cho nhánh MLP
        self.user_embedding_mlp.weight.data.copy_(mlp_model.user_embedding.weight.data)
        self.item_embedding_mlp.weight.data.copy_(mlp_model.item_embedding.weight.data)
        
        # 3. Nạp trọng số cho các tầng ẩn của MLP
        for (m_neumf, m_mlp) in zip(self.mlp_network, mlp_model.mlp_network):
            if isinstance(m_neumf, nn.Linear) and isinstance(m_mlp, nn.Linear):
                m_neumf.weight.data.copy_(m_mlp.weight.data)
                if m_neumf.bias is not None:
                    m_neumf.bias.data.copy_(m_mlp.bias.data)
                    
        # 4. TRỌNG TÂM: Hợp nhất tầng dự đoán (Prediction Layer)
        # Theo bài báo gốc, trọng số tầng cuối được ghép lại và nhân với hệ số alpha = 0.5
        gmf_weight = gmf_model.prediction_layer.weight.data
        mlp_weight = mlp_model.prediction_layer.weight.data
        
        gmf_bias = gmf_model.prediction_layer.bias.data
        mlp_bias = mlp_model.prediction_layer.bias.data
        
        # Ghép trọng số của 2 nhánh (Shape: [1, num_factors] + [1, layers[-1]] -> [1, num_factors + layers[-1]])
        concat_weight = torch.cat([gmf_weight, mlp_weight], dim=1)
        concat_bias = gmf_bias + mlp_bias
        
        # Copy vào layer dự đoán của NeuMF và chia đôi (nhân 0.5)
        self.prediction_layer.weight.data.copy_(0.5 * concat_weight)
        self.prediction_layer.bias.data.copy_(0.5 * concat_bias)