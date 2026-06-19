import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
def split_data_per_user(ratings, test_size=0.2, random_state=42):
    """
    Chia dữ liệu theo từng user bằng phương thức ngẫu nhiên (Per-User Random Split).
    Mỗi user sẽ được xáo trộn toàn bộ danh sách tương tác một cách ngẫu nhiên,
    sau đó trích ra đúng test_size% làm tập test, phần còn lại làm tập train.
    """
    train_list = []
    test_list = []
    
    # Gom nhóm theo từng user để đảm bảo phân bổ đều trên từng cá nhân
    grouped = ratings.groupby('u_idx')
    
    for u_idx, group in grouped:
        # Bỏ hoàn toàn logic kiểm tra và sort theo timestamp..
        shuffled_group = group.sample(frac=1).reset_index(drop=True)
            
        n_test = max(1, int(len(shuffled_group) * test_size)) # Đảm bảo user nào cũng có ít nhất 1 item test
        n_train = len(shuffled_group) - n_test
        
        # Cắt ngẫu nhiên sau khi đã xáo trộn
        train_list.append(shuffled_group.iloc[:n_train])
        test_list.append(shuffled_group.iloc[n_train:])
        
    return pd.concat(train_list).reset_index(drop=True), pd.concat(test_list).reset_index(drop=True)

def random_80_20_split(ratings, ratio=0.8):
    """
    Hàm bọc thực hiện chia dữ liệu ngẫu nhiên theo tỷ lệ 80/20 trên từng User (Per-User).
    Đồng bộ tham số ratio từ main truyền xuống vào cấu trúc chia.
    """
    test_size = round(1.0 - ratio, 2) # Nếu ratio = 0.8 thì test_size = 0.2
    train_df, test_df = split_data_per_user(ratings, test_size=test_size)
    return train_df, test_df

def load_data(path='./data/ml-1m'):
    # Đọc dữ liệu Movies
    m_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(f'{path}/movies.dat', sep='::', names=m_cols, engine='python', encoding='latin-1')
    
    # ML-1M phân tách thể loại. Chuyển thành one-hot rồi áp dụng TF-IDF
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