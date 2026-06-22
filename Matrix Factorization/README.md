# Matrix Factorization Movie Recommendation

This folder implements a movie recommender using **Matrix Factorization with biases** on the MovieLens 1M dataset. The model learns latent vectors for users and items, then uses them to predict ratings and rank movie recommendations.

## 1. Algorithm Overview

Matrix Factorization decomposes a large rating matrix into two latent factor matrices:

- `P`: user latent factor matrix.
- `Q`: item latent factor matrix.
- `b_u`: user bias.
- `b_i`: item bias.
- `mu`: global average rating.

Prediction formula:

```text
rating_hat(u, i) = mu + b_u + b_i + P_u dot Q_i
```

The model is optimized with SGD and uses regularization to reduce overfitting.

## 2. Folder Structure

```text
Matrix Factorization/
├── main.py
├── config.yaml
├── matrix_factorization_model.npz
├── src/
│   ├── data_loader.py
│   ├── metrics.py
│   ├── model.py
│   ├── optimizer.py
│   └── trainer.py
└── data/
    └── raw/
        └── ml-1m/
            ├── movies.dat
            ├── ratings.dat
            ├── users.dat
            └── README
```

## 3. Input Data

The MovieLens 1M dataset path is configured in `config.yaml`:

```yaml
paths:
  data_path: "data/raw/ml-1m/ratings.dat"
```

The main required file is `ratings.dat`. The files `movies.dat` and `users.dat` are kept with the dataset for additional lookup if needed.

## 4. Required Libraries

```bash
pip install numpy pandas pyyaml
```

## 5. Configuration

Main parameters are stored in `config.yaml`:

```yaml
hyperparameters:
  k: 20
  learning_rate: 0.001
  regularization: 0.02
  epochs: 300
  top_k: 20
  patience: 50
```

Parameter meanings:

- `k`: number of latent factors.
- `learning_rate`: SGD learning rate.
- `regularization`: penalty coefficient to reduce overfitting.
- `epochs`: maximum number of training epochs.
- `top_k`: K value for ranking evaluation.
- `patience`: number of non-improving epochs allowed before early stopping.

## 6. How to Run

Train the model:

```bash
python main.py --mode train
```

This command will:

- Load and preprocess MovieLens 1M.
- Split data into a random 80/20 train/test set per user.
- Save the data split to `matrix_factorization_model_data_split.npz`.
- Train Matrix Factorization with SGD.
- Save the best model to `matrix_factorization_model.npz`.

Evaluate and predict using the saved model:

```bash
python main.py --mode predict
```

The `predict` mode requires:

- `matrix_factorization_model.npz`
- `matrix_factorization_model_data_split.npz`

## 7. Output

The program prints:

- `Train RMSE` and `Test RMSE` for each epoch.
- `NDCG@K` during training.
- A demo rating prediction for `User 1 - Movie 1193`.
- Final ranking metrics such as `Precision@K`, `Recall@K`, `NDCG@K`, and `MRR@K`.

## 8. Important Files

- `main.py`: controls train/predict flow, data splitting, and evaluation.
- `src/model.py`: defines Matrix Factorization with biases.
- `src/optimizer.py`: updates parameters with SGD.
- `src/trainer.py`: training loop, early stopping, and best-model saving.
- `src/metrics.py`: computes ranking metrics.

## 9. Notes

Matrix Factorization uses community rating behavior, so it often performs better than pure Content-Based Filtering when there is enough interaction data. However, it can still struggle with cold-start users or movies that have no ratings.
