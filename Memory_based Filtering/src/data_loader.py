import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class MovieLens1MLoader:
    def __init__(self, data_dir='./data/ml-1m'):
        self.data_dir = data_dir
        self.n_users = 0
        self.n_items = 0

    def load_raw_data(self):
        rating_path = os.path.join(self.data_dir, 'ratings.dat')
        movie_path = os.path.join(self.data_dir, 'movies.dat')
        
        # Đọc dữ liệu MovieLens 1M
        ratings = pd.read_csv(rating_path, sep='::', engine='python', 
                              names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')
        movies = pd.read_csv(movie_path, sep='::', engine='python', 
                             names=['movie_id', 'movie_title', 'genres'], encoding='latin-1')
        
        #Tự động trích xuất kích thước thực tế từ dữ liệu
        self.n_users = ratings['user_id'].max()
        self.n_items = ratings['movie_id'].max()
        
        return ratings, movies

    def train_test_split_temporal(self, ratings, test_size=0.2):
        """
        Phân tách Temporal 80-20 theo từng User: Sắp xếp theo thời gian cá nhân,
        giữ lại 20% rating mới nhất của MỖI user cho tập Test để tránh Cold Start.
        """
        # Sắp xếp theo user trước, sau đó đến thời gian
        ratings = ratings.sort_values(by=['user_id', 'timestamp'])
        
        train_dfs = []
        test_dfs = []
        
        # Chia tách 80/20 trên từng nhóm user
        for u_id, group in ratings.groupby('user_id'):
            split_idx = int(len(group) * (1 - test_size))
            
            # Đảm bảo user có ít nhất 1 tương tác ở tập train
            if split_idx == 0 and len(group) > 0:
                split_idx = 1
                
            train_dfs.append(group.iloc[:split_idx])
            test_dfs.append(group.iloc[split_idx:])
            
        train_df = pd.concat(train_dfs)
        test_df = pd.concat(test_dfs)
        
        rate_train = train_df[['user_id', 'movie_id', 'rating']].to_numpy()
        rate_test = test_df[['user_id', 'movie_id', 'rating']].to_numpy()
        
        return rate_train, rate_test

    def build_utility_matrix(self, rate_train, mode = 'item'):
        rows = rate_train[:, 0] - 1
        cols = rate_train[:, 1] - 1
        data = rate_train[:, 2].astype(np.float32)
        
        utility_sparse = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

        user_sums = np.array(utility_sparse.sum(axis=1)).flatten()
        user_counts = np.array((utility_sparse > 0).sum(axis=1)).flatten()
        user_means = np.zeros(self.n_users, dtype=np.float32)
        nonzero_users = user_counts > 0
        user_means[nonzero_users] = user_sums[nonzero_users] / user_counts[nonzero_users]
        

        item_sums = np.array(utility_sparse.sum(axis=0)).flatten()
        item_counts = np.array((utility_sparse > 0).sum(axis=0)).flatten()
        item_means = np.zeros(self.n_items, dtype=np.float32)
        nonzero_items = item_counts > 0
        item_means[nonzero_items] = item_sums[nonzero_items] / item_counts[nonzero_items]
        
        if mode == 'user':
            normalized_data = data - user_means[rows]
        else:
            normalized_data = data - item_means[cols]
        
        utility_norm_sparse = csr_matrix((normalized_data, (rows, cols)), shape=(self.n_users, self.n_items))

        global_mean = data.mean() if len(data) > 0 else 3.5
        
        return utility_sparse, utility_norm_sparse, user_means, item_means, global_mean

    def train_test_split_random(self, ratings, test_size=0.2):
        """
        Phân tách Random 80-20 theo từng User: Xáo trộn ngẫu nhiên dữ liệu 
        của mỗi user trước khi cắt để đảm bảo phân phối ngẫu nhiên.
        """
        
        train_dfs = []
        test_dfs = []
        
        # Chia tách 80/20 ngẫu nhiên trên từng nhóm user
        for u_id, group in ratings.groupby('user_id'):
            # Xáo trộn các dòng của user này
            shuffled_group = group.sample(frac=1).reset_index(drop=True)
            
            split_idx = int(len(shuffled_group) * (1 - test_size))
            
            # Đảm bảo user có ít nhất 1 tương tác ở tập train
            if split_idx == 0 and len(shuffled_group) > 0:
                split_idx = 1
                
            train_dfs.append(shuffled_group.iloc[:split_idx])
            test_dfs.append(shuffled_group.iloc[split_idx:])
            
        train_df = pd.concat(train_dfs)
        test_df = pd.concat(test_dfs)
        
        rate_train = train_df[['user_id', 'movie_id', 'rating']].to_numpy()
        rate_test = test_df[['user_id', 'movie_id', 'rating']].to_numpy()
        
        return rate_train, rate_test

    def save_splits(self, rate_train, rate_test, save_dir):
        """Lưu trữ tập Train/Test cứng ra ổ đĩa để chặn Data Leakage"""
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'rate_train.npy'), rate_train)
        np.save(os.path.join(save_dir, 'rate_test.npy'), rate_test)

    def load_splits(self, load_dir):
        """Tải lại chính xác tập Train/Test đã được chia từ pha Train"""
        train_path = os.path.join(load_dir, 'rate_train.npy')
        test_path = os.path.join(load_dir, 'rate_test.npy')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            return None, None
            
        return np.load(train_path), np.load(test_path)