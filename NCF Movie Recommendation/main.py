import os
import argparse
import torch
import yaml  # Thư viện dùng để đọc file cấu hình .yaml động
import numpy as np
# Import các hàm xử lý từ các file mô-đun trong hệ thống src/
from src.data_utils import load_rating_file_as_matrix
from src.models.gmf import GMF
from src.models.mlp import MLP
from src.models.neumf import NeuMF
from src.training.trainer import run_training
from src.evaluation.metrics import evaluate_model

def parse_arguments():
    """
    Cơ chế nhận tham số từ dòng lệnh Terminal.
    Giúp linh hoạt chuyển đổi giữa Train/Predict và lựa chọn Model mà không cần sửa mã nguồn.
    """
    parser = argparse.ArgumentParser(description="Hệ thống Gợi ý phim NCF (PyTorch) - Cấu hình nâng cao")
    
    # 1. Chế độ chạy: Huấn luyện mới hoặc Chỉ nạp trọng số tối ưu ra dự đoán
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help="Chọn 'train' để huấn luyện mới, hoặc 'predict' để load trọng số ra dự đoán/đánh giá.")
    
    # 2. Lựa chọn mô hình kiến trúc mạng
    parser.add_argument('--model', type=str, default='neumf', choices=['gmf', 'mlp', 'neumf'],
                        help="Lựa chọn mô hình để chạy xử lý (gmf, mlp, hoặc neumf).")
    
    # 3. Đường dẫn thực tế đến file tập dữ liệu ratings.dat của bạn
    parser.add_argument('--data_path', type=str, default='data/raw/ratings.dat',
                        help="Đường dẫn đến file dữ liệu ratings.dat thực tế.")
    
    return parser.parse_args()

def main():
    # =========================================================================
    # BƯỚC 1: THU THẬP THAM SỐ VÀ TỰ ĐỘNG NẠP FILE CONFIG CONFIGS/ TƯƠNG ỨNG
    # =========================================================================
    args = parse_arguments()
    
    # Tạo đường dẫn động dựa trên mô hình được chọn, ví dụ: configs/gmf.yaml
    config_file_path = f"configs/{args.model}_config.yaml" 
    
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"[LỖI HỆ THỐNG] Không tìm thấy file cấu hình của mô hình tại: {config_file_path}\n"
                                f"Vui lòng đảm bảo bạn đã tạo thư mục 'configs/' và đặt các file cấu hình tương ứng vào đó.")
        
    print(f"[HỆ THỐNG] Đang tải siêu tham số (Hyperparameters) từ file: {config_file_path}")
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Đồng bộ các tham số vận hành hệ thống từ Terminal vào cấu hình chung
    config['model'] = args.model
    # config['checkpoint_path'] = f"checkpoints/best_{args.model}_model.pth"
    
    # Tự động cấu hình thiết bị: Ưu tiên tối đa Card đồ họa NVIDIA (CUDA), nếu không có sẽ chạy CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HỆ THỐNG] Thiết bị tính toán hiện tại: {device}")
    
    # =========================================================================
    # BƯỚC 2: TẢI TẬP DỮ LIỆU TỪ THƯ MỤC DATA/RAW/
    # =========================================================================
    
    
    processed_dir = config['processed_dir']
    print("[1/3] Đang phân tách tập dữ liệu Test phục vụ giao thức kiểm toán Leave-One-Out...")
    try:
        test_ratings = np.load(os.path.join(processed_dir, 'test_ratings.npy'))
        test_negatives = np.load(os.path.join(processed_dir, 'test_negatives.npy'))
        print(f"    -> Đã nạp xong {len(test_ratings)} mẫu test.")
    except FileNotFoundError:
        print("[LỖI] Không tìm thấy file .npy tại thư mục processed. Vui lòng chạy preprocess.py trước!")
        return

    print(f"[2/3] Đang đọc tập dữ liệu từ: {args.data_path} ...")
    train_matrix = load_rating_file_as_matrix(args.data_path, test_ratings=test_ratings)
    num_users, num_items = train_matrix.shape
    print(f"    -> Thống kê dữ liệu: Tổng số Users = {num_users} | Tổng số Items (Phim) = {num_items}")
    # =========================================================================
    # BƯỚC 3: KHỞI TẠO MÔ HÌNH DỰA TRÊN THÔNG SỐ ĐỌC ĐƯỢC TỪ FILE CONFIG 
    # =========================================================================
    print(f"[3/3] Khởi tạo kiến trúc mạng NCF được chỉ định: {args.model.upper()}...")
    if args.model == 'gmf':
        model = GMF(num_users=num_users, num_items=num_items, num_factors=config['num_factors'])
    elif args.model == 'mlp':
        model = MLP(num_users=num_users, num_items=num_items, layers=config['layers'])
    elif args.model == 'neumf':
        model = NeuMF(num_users=num_users, num_items=num_items, 
                      num_factors=config['num_factors'], layers=config['layers'])
        
        # [ĐÃ SỬA LỖI SỐ 3]: Bổ sung cơ chế kích hoạt Pre-train từ YAML
        if config.get('use_pretrain'):
            print("[*] Đang nạp trọng số Pre-train từ các nhánh GMF và MLP độc lập...")
            try:
                gmf_sub = GMF(num_users, num_items, config['num_factors'])
                mlp_sub = MLP(num_users, num_items, config['layers'])
                
                gmf_sub.load_state_dict(torch.load(config['gmf_pretrain_path'], map_location=device))
                mlp_sub.load_state_dict(torch.load(config['mlp_pretrain_path'], map_location=device))
                
                model.load_pretrain_weights(gmf_sub, mlp_sub)
                print("[+] Nạp và hợp nhất hệ số Alpha=0.5 thành công cho NeuMF!")
            except FileNotFoundError:
                print("[CẢNH BÁO] Không tìm thấy file pretrain. Vui lòng train GMF và MLP trước. NeuMF sẽ chạy với trọng số ngẫu nhiên.")

    # =========================================================================
    # CHẾ ĐỘ CHẠY A: HUẤN LUYỆN (CÓ HỖ TRỢ RESUME TRAINING)
    # =========================================================================
    if args.mode == 'train':
        print(f"\n>>> KÍCH HOẠT CHẾ ĐỘ: HUẤN LUYỆN MÔ HÌNH ({args.model.upper()}) <<<")
        
        # [CƠ CHẾ MỚI]: Kiểm tra xem đã từng train chưa, nếu có thì nạp lại trọng số cũ
        if os.path.exists(config['checkpoint_path']):
            print(f"[INFO] Phát hiện checkpoint cũ tại {config['checkpoint_path']}.")
            print("       -> Đang nạp trọng số để tiếp tục tối ưu từ kết quả trước đó...")
            model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))
        else:
            print("[INFO] Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ đầu (From Scratch).")
            
        # Đảm bảo model đã nằm trên thiết bị (GPU/CPU)
        model.to(device)
        
        # Gọi luồng huấn luyện
        run_training(model, train_matrix, test_ratings, test_negatives, config, device)
    # =========================================================================
    # CHẾ ĐỘ CHẠY B: CHỈ LOAD LẠI TRỌNG SỐ TỐT NHẤT ĐỂ DỰ ĐOÁN (PREDICT MODE)
    # =========================================================================
    elif args.mode == 'predict':
        print(f"\n>>> KÍCH HOẠT CHẾ ĐỘ: TRÍCH XUẤT TRỌNG SỐ TỐI ƯU & DỰ ĐOÁN TỐC ĐỘ Cao ({args.model.upper()}) <<<")
        
        # Kiểm tra sự tồn tại của file checkpoint .pth tối ưu nhất lưu theo tiêu chí NDCG
        if not os.path.exists(config['checkpoint_path']):
            print(f"[LỖI THIẾT LẬP] Không tìm thấy file checkpoint lưu trọng số tại: {config['checkpoint_path']}")
            print("Vui lòng thực hiện huấn luyện mô hình bằng lệnh '--mode train' trước để tạo ra file kỷ lục này.")
            return
            
        # Nạp trạng thái trọng số của Epoch xuất sắc nhất thu được trong quá trình Train vào mô hình rỗng
        print(f"[*] Đang nạp trọng số tốt nhất từ file: {config['checkpoint_path']} ...")
        checkpoint = torch.load(config['checkpoint_path'], map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        print("[+] Khớp nối trọng số thành công! Hệ thống đã sẵn sàng đưa ra dự đoán.")
        
        # Hành động 1: Chạy đánh giá toàn diện lại các mốc chỉ số Top-K dựa theo cấu hình mốc K trong file config
        print("\n--- BẢNG ĐÁNH GIÁ CHẤT LƯỢNG GỢI Ý TẠI CÁC MỐC TOP-K (MÔ HÌNH TỐI ƯU) ---")
        final_metrics = evaluate_model(model, test_ratings, test_negatives, config['top_k'], device)
        
        for K in config['top_k']:
            print(f"Mốc Top-{K:02d} Gợi ý | Precision@{K}: {final_metrics[K]['precision']:.4f} "
                  f"| Recall@{K}: {final_metrics[K]['recall']:.4f} "
                  f"| NDCG@{K}: {final_metrics[K]['ndcg']:.4f} "
                  f"| MRR@{K}: {final_metrics[K]['mrr']:.4f}")
            
        # Hành động 2: Demo tính toán dự đoán thực tế cho một User bất kỳ để chứng minh với hội đồng
        print("\n--- DEMO THỰC THI GỢI Ý THỜI GIAN THỰC CHO USER 10 ---")
        demo_user = 10
        # Tính toán xác suất với 5 bộ phim ngẫu nhiên (ví dụ ID: 100, 200, 500, 1193, 2376)
        demo_items = torch.tensor([100, 200, 500, 1193, 2376], dtype=torch.long).to(device)
        user_tensor = torch.full((len(demo_items),), demo_user, dtype=torch.long).to(device)
        
        model.eval()
        with torch.no_grad():
            scores = model(user_tensor, demo_items)
            
        print(f"Xác suất User {demo_user} sẽ yêu thích và click xem các bộ phim tương ứng là:")
        for item, score in zip(demo_items.cpu().numpy(), scores.cpu().numpy()):
            print(f"  - Sản phẩm Phim (ID {item:4d}) --> Điểm số ưa thích dự đoán: {score*100:6.2f}%")

if __name__ == '__main__':
    main()