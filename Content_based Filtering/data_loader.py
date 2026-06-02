import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer

def load_data(path='./data/ml-1m'):
    # Đọc dữ liệu Movies
    m_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(f'{path}/movies.dat', sep='::', names=m_cols, engine='python', encoding='latin-1')
    
    # ML-1M phân tách thể loại bằng '|'. Ta chuyển thành one-hot rồi áp dụng TF-IDF
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genres'])
    
    transformer = TfidfTransformer(smooth_idf=True, norm='l2')
    item_features = transformer.fit_transform(genre_matrix).toarray()
    
    # Map movie_id (số nguyên không liên tục) về index 0-based
    movie2idx = {m: i for i, m in enumerate(movies['movie_id'])}
    
    # Đọc dữ liệu Ratings
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(f'{path}/ratings.dat', sep='::', names=r_cols, engine='python', encoding='latin-1')
    
    # Map user_id về index 0-based
    user2idx = {u: i for i, u in enumerate(ratings['user_id'].unique())}
    
    ratings['u_idx'] = ratings['user_id'].map(user2idx)
    ratings['m_idx'] = ratings['movie_id'].map(movie2idx)
    
    return ratings, item_features, user2idx, movie2idx

def leave_one_out_split(ratings):
    # Sắp xếp theo timestamp để lấy rating cuối cùng làm test set
    ratings = ratings.sort_values(['u_idx', 'timestamp'])
    
    # Lấy ra bộ test (1 item cuối cho mỗi user)
    test_df = ratings.groupby('u_idx').tail(1)
    train_df = ratings.drop(test_df.index)
    
    return train_df, test_df