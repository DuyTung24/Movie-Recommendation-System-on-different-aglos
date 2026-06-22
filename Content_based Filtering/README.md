# Content-Based Filtering Movie Recommendation

This folder implements a **Content-Based Filtering** movie recommender using the MovieLens 1M dataset. The model learns a separate preference profile for each user from movie genre features, then predicts ratings for unseen movies.

## 1. Algorithm Overview

Content-Based Filtering recommends items based on item attributes. In this project:

- Each movie is represented by its genre vector.
- Genres are converted to one-hot features, then transformed with TF-IDF.
- A separate Ridge Regression model is trained for each user.
- Predicted ratings are computed from movie feature vectors and the learned user preference weights.

General prediction formula:

```text
rating_hat(user, movie) = item_features(movie) * W_user + b_user
```

## 2. Folder Structure

```text
Content_based Filtering/
├── main.py
├── data_loader.py
├── model.py
├── evaluator.py
├── data/
│   └── ml-1m/
│       ├── movies.dat
│       ├── ratings.dat
│       └── users.dat
└── weights/
    └── cb_weights.npz
```

## 3. Input Data

The project uses the MovieLens 1M dataset located at:

```text
data/ml-1m/
```

Required files:

- `movies.dat`: movie metadata and genres.
- `ratings.dat`: user ratings for movies.
- `users.dat`: user metadata, included with the dataset but not central to this model.

## 4. Required Libraries

```bash
pip install numpy pandas scikit-learn
```

## 5. How to Run

Train a new model:

```bash
python main.py --mode train
```

This command will:

- Load the MovieLens 1M dataset.
- Build TF-IDF movie features from genres.
- Split ratings into an 80/20 train/test set per user.
- Train a Ridge Regression preference model for each user.
- Save model weights to `weights/cb_weights.npz`.
- Save the train/test split to `weights/train_split.csv` and `weights/test_split.csv`.

Run prediction and evaluation using saved weights:

```bash
python main.py --mode predict
```

The `predict` mode requires `train` mode to be run first so that the files in `weights/` exist.

## 6. Output

After running, the program prints:

- `RMSE`: rating prediction error.
- `Precision@20`, `Recall@20`, `NDCG@20`, `MRR@20`: Top-K ranking metrics.
- A sample predicted rating for a random user and random movie.

## 7. Important Files

- `main.py`: main entry point for training, prediction, and evaluation.
- `data_loader.py`: loads MovieLens data, creates TF-IDF features, and splits train/test data.
- `model.py`: implements `ContentBasedRecommender`.
- `evaluator.py`: computes RMSE and Top-K metrics.

## 8. Notes

This algorithm works well when items have meaningful content features, such as movie genres. Its strength is recommending based on item attributes, but it may perform poorly when the available content features are too limited.
