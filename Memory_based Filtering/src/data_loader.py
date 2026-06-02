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
        
        # Đọc dữ liệu phân tách bằng ký tự '::' của MovieLens 1M
        ratings = pd.read_csv(rating_path, sep='::', engine='python', 
                              names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')
        movies = pd.read_csv(movie_path, sep='::', engine='python', 
                             names=['movie_id', 'movie_title', 'genres'], encoding='latin-1')
        
        # Ý TƯỞNG 1: Tự động trích xuất kích thước thực tế từ dữ liệu
        self.n_users = ratings['user_id'].max()
        self.n_items = ratings['movie_id'].max()
        
        return ratings, movies

    def train_test_split_loo(self, ratings):
        """
        Phân tách Leave-One-Out: Sắp xếp theo thời gian, 
        giữ lại rating cuối cùng của mỗi user cho tập kiểm định (Test).
        """
        ratings = ratings.sort_values(by='timestamp')
        test_df = ratings.groupby('user_id').tail(1)
        train_df = ratings.drop(test_df.index)
        
        rate_train = train_df[['user_id', 'movie_id', 'rating']].to_numpy()
        rate_test = test_df[['user_id', 'movie_id', 'rating']].to_numpy()
        
        return rate_train, rate_test

    def build_utility_matrix(self, rate_train):
        """
        Ý TƯỞNG 2: Khởi tạo hoàn toàn dưới dạng Ma trận thưa (Sparse Matrix) để tối ưu bộ nhớ.
        """
        # Chuyển ID (bắt đầu từ 1) thành Index mảng (bắt đầu từ 0)
        rows = rate_train[:, 0] - 1
        cols = rate_train[:, 1] - 1
        data = rate_train[:, 2].astype(np.float32)
        
        utility_sparse = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))
        
        # Tính toán User Mean hiệu năng cao từ ma trận thưa
        user_sums = np.array(utility_sparse.sum(axis=1)).flatten()
        user_counts = np.array((utility_sparse > 0).sum(axis=1)).flatten()
        
        user_means = np.zeros(self.n_users, dtype=np.float32)
        nonzero_users = user_counts > 0
        user_means[nonzero_users] = user_sums[nonzero_users] / user_counts[nonzero_users]
        
        # Ý TƯỞNG 4: Vector hóa hoàn toàn bước trừ chuẩn hóa để giữ nguyên tính chất thưa
        normalized_data = data - user_means[rows]
        utility_norm_sparse = csr_matrix((normalized_data, (rows, cols)), shape=(self.n_users, self.n_items))
        
        return utility_sparse, utility_norm_sparse, user_means