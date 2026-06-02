import numpy as np
import pandas as pd
import os

def generate_leave_one_out(data_path, output_dir):
    # 1. Đọc dữ liệu
    # Định dạng giả định: UserID::ItemID::Rating::Timestamp
    df = pd.read_csv(data_path, sep='::', header=None, names=['user', 'item', 'rating', 'timestamp'], engine='python')
    
    #Chỉ coi những rating >= 3 mới tương đương với tương tác tích cực(tức = 1) với implicit feedback
    df = df[df['rating'] >= 3.0]  # Chỉ giữ lại các tương tác tích cực (rating >= 3)

    # 2. Sắp xếp theo User và Thời gian
    df = df.sort_values(by=['user', 'timestamp'])
    
    # Tạo cấu trúc lưu trữ
    test_ratings = []
    test_negatives = []
    
    # Tương tác cho train (cần lưu lại để loại trừ khi lấy mẫu âm)
    train_dict = {} 

    # 3. Leave-one-out: Lấy mẫu cuối cùng của mỗi user làm Test
    for user, group in df.groupby('user'):
        items = group['item'].tolist()
        
        # Mẫu cuối là Test
        test_item = items[-1]
        test_ratings.append([user, test_item])
        
        # Các mẫu còn lại là Train
        train_items = items[:-1]
        train_dict[user] = set(train_items)
        
        # 4. Lấy 99 mẫu âm (Negative Sampling)
        # Lấy ngẫu nhiên 99 items mà user chưa từng tương tác
        negatives = []
        all_items = df['item'].unique()
        while len(negatives) < 99:
            neg = np.random.choice(all_items)
            if neg not in train_dict[user] and neg != test_item:
                negatives.append(neg)
        test_negatives.append(negatives)

    # 5. Lưu xuống đĩa
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'test_ratings.npy'), np.array(test_ratings))
    np.save(os.path.join(output_dir, 'test_negatives.npy'), np.array(test_negatives))
    
    print(f"Đã tạo xong file tại {output_dir}")
    print(f"Tổng số User xử lý: {len(test_ratings)}")

if __name__ == "__main__":
    generate_leave_one_out('data/raw/ratings.dat', 'data/processed/')