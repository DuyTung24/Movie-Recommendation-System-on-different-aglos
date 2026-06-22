# Memory-Based Collaborative Filtering Movie Recommendation

This folder implements a movie recommender using **Memory-Based Collaborative Filtering** on the MovieLens 1M dataset. The model computes cosine similarity between users or items, then predicts ratings from the nearest neighbors.

## 1. Algorithm Overview

Memory-Based Collaborative Filtering does not learn latent vectors like Matrix Factorization. Instead, it works directly on the rating matrix:

- `user-based`: finds users whose behavior is similar to the target user.
- `item-based`: finds movies whose rating patterns are similar to the target movie.
- Cosine similarity is used to measure similarity.
- The K nearest neighbors are used to predict ratings.

The default configuration currently uses `user-based` filtering in `config.yaml`.

## 2. Folder Structure

```text
Memory_based Filtering/
├── main.py
├── config.yaml
├── src/
│   ├── data_loader.py
│   ├── metrics.py
│   └── model.py
├── data/
│   └── ml-1m/
│       ├── movies.dat
│       ├── ratings.dat
│       ├── users.dat
│       └── README
└── checkpoints/
```

## 3. Input Data

The MovieLens 1M dataset must be located at:

```text
data/ml-1m/
```

Main files:

- `ratings.dat`: user ratings for movies.
- `movies.dat`: movie metadata used to display movie titles in the demo.
- `users.dat`: MovieLens 1M user metadata.

## 4. Required Libraries

```bash
pip install numpy pandas scipy scikit-learn pyyaml
```

## 5. Configuration

The `config.yaml` file contains the main settings:

```yaml
model:
  cf_mode: "user"
  n_neighbors: 30

evaluation:
  top_k: 20
  test_size: 0.2
```

Parameter meanings:

- `cf_mode`: use `"user"` for user-based CF or `"item"` for item-based CF.
- `n_neighbors`: number of nearest neighbors used for prediction.
- `top_k`: K value for ranking metrics.
- `test_size`: evaluation data ratio.

## 6. How to Run

Train or recompute similarity:

```bash
python main.py --mode train
```

This command will:

- Load the MovieLens 1M dataset.
- Split data into a random 80/20 train/test set.
- Build a sparse utility matrix.
- Compute cosine similarity based on `cf_mode`.
- Save a checkpoint to `checkpoints/`.
- Evaluate the model on the test set.

Run prediction using a saved checkpoint:

```bash
python main.py --mode predict
```

If the checkpoint does not exist, the program will recompute similarity and save a new checkpoint.

## 7. Output

The program prints:

- `RMSE`: rating prediction error.
- `Precision@K`, `Recall@K`, `NDCG@K`, `MRR@K`: recommendation ranking quality.
- A sample predicted rating for a random user and movie.

## 8. Important Files

- `main.py`: main entry point for training, prediction, and evaluation.
- `src/data_loader.py`: loads MovieLens data, splits data, and builds the utility matrix.
- `src/model.py`: computes cosine similarity and predicts ratings.
- `src/metrics.py`: builds ranking evaluation data and computes metrics.
- `config.yaml`: configures user-based/item-based mode and evaluation parameters.

## 9. Notes

Memory-Based CF is easy to understand and explain because recommendations are based on similar users or similar items. Its downside is that the similarity matrix can require significant memory as the number of users or items grows.
