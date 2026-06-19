# src/data_loader.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MovieLensDataLoader:
    """
    Class chịu trách nhiệm load và tiền xử lý dữ liệu MovieLens-100k.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None
        self.num_users = 0
        self.num_items = 0
        
        # Lưu trữ mapping để có thể map ngược lại từ index (0 -> N-1) ra ID gốc của MovieLens
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}

    def load_and_preprocess(self):
        """
        Đọc file ratings.dat của MovieLens-1M và thực hiện mapping các ID sang mảng index liên tục.
        """
        print(f"Đang đọc dữ liệu từ: {self.file_path}...")
        
        # Cập nhật: MovieLens-1M ratings.dat dùng ký tự '::' để phân tách các cột
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(self.file_path, sep='::', names=columns, engine='python')

        # Tạo dictionary mapping cho User
        unique_users = df['user_id'].unique()
        self.user2idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx2user = {idx: user for user, idx in self.user2idx.items()}
        self.num_users = len(unique_users)

        # Tạo dictionary mapping cho Item (Movie)
        unique_items = df['item_id'].unique()
        self.item2idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        self.num_items = len(unique_items)

        # Apply mapping vào dataframe
        df['user_idx'] = df['user_id'].map(self.user2idx)
        df['item_idx'] = df['item_id'].map(self.item2idx)

        # Chỉ giữ lại các cột cần thiết cho Matrix Factorization
        # Bổ sung cột timestamp để phục vụ chia Global Temporal Split
        self.dataset = df[['user_idx', 'item_idx', 'rating']]
        print(f"Đã load xong! Tổng số Users: {self.num_users}, Tổng số Items: {self.num_items}")
        
        return self.dataset

    def get_dimensions(self):
        """
        Trả về kích thước ma trận (số lượng users, số lượng items) để khởi tạo model.
        """
        return self.num_users, self.num_items
