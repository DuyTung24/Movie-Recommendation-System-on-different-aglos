# Neural Collaborative Filtering Movie Recommendation

This folder implements a movie recommender using **Neural Collaborative Filtering (NCF)** with PyTorch. The project supports three architectures: `GMF`, `MLP`, and `NeuMF`.

## 1. Algorithm Overview

Neural Collaborative Filtering learns user-item interactions with neural networks instead of relying only on a linear dot product.

Supported models:

- `GMF`: Generalized Matrix Factorization, which learns user/item embeddings and combines them with element-wise multiplication.
- `MLP`: Multi-Layer Perceptron, which concatenates user/item embeddings and passes them through fully connected layers.
- `NeuMF`: combines the GMF and MLP branches to capture both linear and non-linear interactions.

This project treats the task as implicit feedback: ratings greater than or equal to 3 are considered positive interactions.

## 2. Folder Structure

```text
NCF Movie Recommendation/
├── main.py
├── preprocess.py
├── requirements.txt
├── configs/
│   ├── gmf_config.yaml
│   ├── mlp_config.yaml
│   └── neumf_config.yaml
├── src/
│   ├── data_utils.py
│   ├── models/
│   │   ├── gmf.py
│   │   ├── mlp.py
│   │   └── neumf.py
│   ├── training/
│   │   └── trainer.py
│   └── evaluation/
│       └── metrics.py
├── data/
│   ├── raw/
│   │   └── ratings.dat
│   └── processed/
│       ├── test_ratings.npy
│       └── test_negatives.npy
└── checkpoints/
    ├── gmf_best.pth
    ├── mlp_best.pth
    └── neumf_best.pth
```

## 3. Input Data

The raw rating file must be located at:

```text
data/raw/ratings.dat
```

Expected MovieLens-compatible format:

```text
user_id::item_id::rating::timestamp
```

Before training, generate the processed evaluation files:

```bash
python preprocess.py
```

This command will:

- Read `data/raw/ratings.dat`.
- Treat ratings `>= 3.0` as positive implicit feedback.
- Split interactions per user into an 80/20 train/test split.
- Save `test_ratings.npy` and `test_negatives.npy` to `data/processed/`.

## 4. Required Libraries

Install dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

Main libraries include:

```bash
pip install numpy pandas scipy pyyaml torch
```

## 5. Configuration

Each model has its own config file:

- `configs/gmf_config.yaml`
- `configs/mlp_config.yaml`
- `configs/neumf_config.yaml`

Important parameters:

- `num_factors`: embedding size for GMF.
- `layers`: hidden layer structure for MLP/NeuMF.
- `lr`: learning rate.
- `batch_size`: batch size.
- `epochs`: maximum number of training epochs.
- `num_negatives`: number of negative samples used during training.
- `patience`: early stopping patience.
- `top_k`: ranking evaluation cutoffs.
- `checkpoint_path`: location of the best saved model.

For `NeuMF`, pretraining can be enabled:

```yaml
use_pretrain: true
gmf_pretrain_path: "checkpoints/gmf_best.pth"
mlp_pretrain_path: "checkpoints/mlp_best.pth"
```

When pretraining is enabled, train `GMF` and `MLP` first, then train `NeuMF`.

## 6. How to Run

Step 1: Preprocess the data.

```bash
python preprocess.py
```

Step 2: Train models.

```bash
python main.py --mode train --model gmf
python main.py --mode train --model mlp
python main.py --mode train --model neumf
```

You can train only one model if a full comparison is not needed. Because `neumf` uses `use_pretrain: true` by default, it works best after `gmf` and `mlp` checkpoints already exist.

Step 3: Predict and evaluate saved models.

```bash
python main.py --mode predict --model gmf
python main.py --mode predict --model mlp
python main.py --mode predict --model neumf
```

You can also pass a custom data path:

```bash
python main.py --mode train --model neumf --data_path data/raw/ratings.dat
```

## 7. Output

The program prints:

- The active device: `cuda` if a GPU is available, otherwise `cpu`.
- Training progress and early stopping status.
- `Precision@K`, `Recall@K`, `NDCG@K`, and `MRR@K`.
- A demo prediction showing the probability that a user will like several sample movies.

## 8. Important Files

- `main.py`: main entry point for model selection, training, prediction, and evaluation.
- `preprocess.py`: creates processed data for the evaluation protocol.
- `src/models/gmf.py`: GMF architecture.
- `src/models/mlp.py`: MLP architecture.
- `src/models/neumf.py`: NeuMF architecture.
- `src/training/trainer.py`: training loop, negative sampling, and checkpoint saving.
- `src/evaluation/metrics.py`: ranking metric computation.

## 9. Notes

NCF is useful when you want to learn non-linear interactions between users and items. It requires more computation than traditional recommendation methods, but it is more flexible for large datasets and complex user behavior.
