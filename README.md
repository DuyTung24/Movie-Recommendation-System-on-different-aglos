# Movie-Recommendation-System-on-different-algos

This repository contains multiple movie recommendation system implementations built with different recommendation algorithms. Each folder is a standalone experiment using the MovieLens dataset and includes its own README with detailed setup, training, prediction, and evaluation instructions.

## Algorithms Included

| Folder | Algorithm | Main Idea |
| --- | --- | --- |
| `Content_based Filtering/` | Content-Based Filtering | Recommends movies from item content features such as genres. |
| `Memory_based Filtering/` | Memory-Based Collaborative Filtering | Uses user-user or item-item cosine similarity to predict ratings. |
| `Matrix Factorization/` | Matrix Factorization | Learns latent user and item vectors with bias terms. |
| `NCF Movie Recommendation/` | Neural Collaborative Filtering | Uses neural networks to model user-item interactions. |

## Repository Structure

```text
Movie-Recommendation-System-on-different-algos/
├── Content_based Filtering/
├── Memory_based Filtering/
├── Matrix Factorization/
├── NCF Movie Recommendation/
└── README.md
```

## Dataset

The projects are designed around the MovieLens dataset, mainly MovieLens 1M. Each algorithm folder keeps its own expected data path, so check the README inside that folder before running.

Common MovieLens files include:

- `ratings.dat`: user-movie rating interactions.
- `movies.dat`: movie metadata and genres.
- `users.dat`: user metadata.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/<your-username>/Movie-Recommendation-System-on-different-algos.git
cd Movie-Recommendation-System-on-different-algos
```

Install dependencies according to the algorithm you want to run. For example:

```bash
pip install numpy pandas scikit-learn scipy pyyaml
```

For the Neural Collaborative Filtering project, install its requirements:

```bash
cd "NCF Movie Recommendation"
pip install -r requirements.txt
```

## Running Each Algorithm

### Content-Based Filtering

```bash
cd "Content_based Filtering"
python main.py --mode train
python main.py --mode predict
```

### Memory-Based Collaborative Filtering

```bash
cd "Memory_based Filtering"
python main.py --mode train
python main.py --mode predict
```

### Matrix Factorization

```bash
cd "Matrix Factorization"
python main.py --mode train
python main.py --mode predict
```

### Neural Collaborative Filtering

```bash
cd "NCF Movie Recommendation"
python preprocess.py
python main.py --mode train --model gmf
python main.py --mode train --model mlp
python main.py --mode train --model neumf
python main.py --mode predict --model neumf
```

## Evaluation Metrics

The projects evaluate recommendation quality using a mix of rating prediction and ranking metrics:

- `RMSE`: measures rating prediction error.
- `Precision@K`: percentage of recommended items that are relevant.
- `Recall@K`: percentage of relevant items recovered in the recommendation list.
- `NDCG@K`: ranking quality with position-aware gain.
- `MRR@K`: rank position of the first relevant recommendation.

## Algorithm Comparison

| Algorithm | Strengths | Limitations |
| --- | --- | --- |
| Content-Based Filtering | Works with item metadata and is easy to explain. | Depends heavily on the quality of item features. |
| Memory-Based CF | Simple, intuitive, and directly based on similar users/items. | Similarity matrices can become large and expensive. |
| Matrix Factorization | Learns compact latent representations and usually performs well on sparse ratings. | Struggles with cold-start users or items. |
| Neural CF | Captures non-linear user-item interactions. | Requires more compute and more careful tuning. |

## Notes

- Each algorithm folder is independent and can be run separately.
- Saved model files and checkpoints are stored inside the corresponding algorithm folder.
- Paths in the code are relative to each algorithm folder, so run commands from inside the target folder.
- For detailed documentation, open the `README.md` file inside each algorithm folder.
