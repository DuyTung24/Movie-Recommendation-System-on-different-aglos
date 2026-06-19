import numpy as np
import pandas as pd
import os

def generate_random_split(data_path, output_dir):
    """
    Chia tập dữ liệu theo chiến lược PER-USER RANDOM SPLIT 80/20.
    Đảm bảo mỗi user đều có 80% dữ liệu để học và 20% dữ liệu để đánh giá.
    """
    # 1. Đọc dữ liệu và lọc Implicit Threshold (>= 3)
    df = pd.read_csv(data_path, sep='::', header=None, names=['user', 'item', 'rating', 'timestamp'], engine='python')
    df = df[df['rating'] >= 3.0]
    
    train_list = []
    test_list = []
    
    # 2. Duyệt qua từng User để chia ngẫu nhiên tỉ lệ 80/20 độc lập
    for user, group in df.groupby('user'):
        # Xáo trộn các tương tác của riêng user này
        shuffled_group = group.sample(frac=1).reset_index(drop=True)
        split_idx = int(len(shuffled_group) * 0.8)
        
        # Thao tác fail-safe: Nếu user chỉ có 1 tương tác duy nhất, giữ lại cho tập train
        if split_idx == 0 and len(shuffled_group) > 0:
            split_idx = 1
            
        train_list.append(shuffled_group.iloc[:split_idx])
        test_list.append(shuffled_group.iloc[split_idx:])
        
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    train_dict = train_df.groupby('user')['item'].apply(list).to_dict()
    test_dict = test_df.groupby('user')['item'].apply(list).to_dict()
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'test_ratings.npy'), test_dict)
    np.save(os.path.join(output_dir, 'test_negatives.npy'), train_dict)
    
    print(f"[*] Đã tạo thành công Per-user Random Split 80/20)")
    print(f"    - Tổng tương tác Train: {len(train_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    generate_random_split('data/raw/ratings.dat', 'data/processed/')