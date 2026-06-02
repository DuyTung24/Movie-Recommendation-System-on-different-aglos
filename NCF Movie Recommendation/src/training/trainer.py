from logging import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

from src.data_utils import NCFDataset, get_train_instances
from src.evaluation.metrics import evaluate_model

def run_training(model, train_matrix, test_ratings, test_negatives, config, device):
    """
    Hàm huấn luyện chính - ĐÃ CẬP NHẬT LUẬT LƯU THEO NDCG@10 TỐT NHẤT TẠI CÁC EPOCH
    """
    model.to(device)
    criterion = nn.BCELoss()
    
    # if config['model'].lower() == 'neumf':
    #     print("[*] Áp dụng thuật toán SGD (Stochastic Gradient Descent) cho mô hình NeuMF.")
    #     optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    # else:
    #     print(f"[*] Áp dụng thuật toán Adam cho mô hình {config['model'].upper()}.")
    #     optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    print(f"[*] Áp dụng thuật toán Adam cho mô hình {config['model'].upper()}.")
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # -----------------------------------------------------------------
    # THAY ĐỔI TẠI ĐÂY: Chuyển từ best_hr sang theo dõi best_ndcg
    # -----------------------------------------------------------------
    best_ndcg = 0.0        # Điểm NDCG@10 cao nhất thu được
    best_epoch = -1        # Epoch đạt kỷ lục NDCG
    patience_counter = 0   # Đếm số epoch không cải thiện để Early Stopping

    print("[*] Đang đánh giá năng lực của trọng số hiện tại (Baseline)...")
    baseline_metrics = evaluate_model(model, test_ratings, test_negatives, config['top_k'], device)
    best_ndcg = baseline_metrics[10]['ndcg']
    print(f"    -> Kỷ lục NDCG@10 hiện tại đang được giữ mức: {best_ndcg:.4f}\n")
    # =========================================================================
    
    print(f"\n--- BẮT ĐẦU TRAINING MÔ HÌNH: {config['model'].upper()} ---")
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # Sinh mẫu âm động đầu mỗi epoch
        user_input, item_input, labels = get_train_instances(train_matrix, config['num_negatives'])
        train_dataset = NCFDataset(user_input, item_input, labels)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=12,          # Điểm ngọt cho hệ thống 24 luồng, không chiếm dụng hết tài nguyên của Main Thread
            pin_memory=True          # Ghim chặt luồng RAM phân phối trực tiếp sang bộ nhớ GPU không qua trung gian
        )
        
        model.train()
        epoch_loss = 0.0
        for batch_user, batch_item, batch_labels in train_loader:
            batch_user, batch_item, batch_labels = batch_user.to(device), batch_item.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_user, batch_item)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        
        # Đánh giá mô hình định kỳ sau mỗi epoch
        metrics = evaluate_model(model, test_ratings, test_negatives, config['top_k'], device)
        
        # Trích xuất kết quả tại mốc K=10 để làm thước đo lưu checkpoint
        hr_10 = metrics[10]['recall']
        ndcg_10 = metrics[10]['ndcg']
        
        train_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d} [{train_time:.1f}s] | Loss: {avg_loss:.4f} | Recall@10: {hr_10:.4f} | NDCG@10: {ndcg_10:.4f}")
        
        # -----------------------------------------------------------------
        # THAY ĐỔI TẠI ĐÂY: Logic so sánh kỷ lục dựa vào NDCG@10
        # -----------------------------------------------------------------
        if ndcg_10 > best_ndcg:
            best_ndcg = ndcg_10
            best_epoch = epoch + 1 # Lưu lại chính xác số Epoch tốt nhất (chạy từ 1)
            patience_counter = 0
            
            # Tiến hành ghi đè/lưu trọng số của Epoch hoàng kim này vào ổ cứng
            os.makedirs(os.path.dirname(config['checkpoint_path']), exist_ok=True)
            torch.save(model.state_dict(), config['checkpoint_path'])
            print(f"  --> [KỶ LỤC MỚI] Đã lưu trọng số tốt nhất của Epoch {best_epoch} (NDCG@10: {best_ndcg:.4f})")
        else:
            patience_counter += 1
            
        # Kích hoạt dừng sớm nếu vượt quá ngưỡng chịu đựng
        if patience_counter >= config['patience']:
            print(f"\n[!] Kích hoạt Early Stopping tại Epoch {epoch+1}. Không có cải thiện NDCG sau {config['patience']} epochs.")
            break
            
    print(f"--- KẾT THÚC QUÁ TRÌNH TRAIN MÔ HÌNH {config['model'].upper()} ---")
    print(f"[*] Trọng số tối ưu nhất được bảo tồn từ Epoch: {best_epoch} với NDCG@10 = {best_ndcg:.4f}\n")