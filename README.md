# Personalized Online Course Recommender System

Capstone project of the IBM Machine Learning Professional Certificate. For a fictional online learning platform, *AI Training Room*, I built and compared three content-based recommenders and three collaborative-filtering models to help learners find relevant new courses.

## Data

IBM Skills Network course datasets, loaded directly from IBM's public course storage in the notebook:

- **307 courses** with titles, descriptions and 14 genre tags
- **33,901 learners** and **233,306 enrollments** with ratings (2 or 3)
- Pre-computed learner genre profiles and a 1,000-learner test set

## Exploratory data analysis

- The largest genres are BackendDev (78 courses), MachineLearning (69) and Database (60).
- The median learner took 6 courses, and **8,320 learners (25%) took only one**, which is a cold-start challenge.
- The top 20 courses account for **63.3% of enrollments**, which creates a risk of popularity bias.

## Content-based recommenders (unsupervised)

| Method | How it scores unseen courses | Avg. new courses per user |
|---|---|---|
| User profile + genres | Dot product of learner genre profile and course genre vector | 53.4 / 16.7 / 5.9 (threshold 10 / 20 / 30) |
| Course similarity (bag of words) | Highest cosine similarity of course text to a course already taken | 36.7 / 17.3 / 5.6 (threshold 0.5 / 0.6 / 0.7) |
| Clustering (PCA + K-Means) | Popular courses within the learner's cluster (9 PCA components, k = 15) | 11.8 / 5.7 / 3.8 (min. share 10% / 20% / 30%) |

## Collaborative filtering (supervised): test RMSE

All models use the same 70/30 split, so the scores are directly comparable.

| Model | Test RMSE | vs baseline |
|---|---|---|
| Baseline (global mean) | 0.2115 | – |
| KNN item-based (k = 10/20/40) | 0.1933 | 8.6% |
| NMF (30 factors) | 0.1894 | 10.4% |
| **Neural network embeddings (dim = 8)** | **0.1521** | **28.1%** |

The **neural embedding model** (Keras, with user and course embeddings, bias terms, L2 regularisation and early stopping) is recommended for rating prediction.

## Limitations and next steps

Popularity bias, cold-start learners, ratings that are almost all 3, and near-duplicate catalogue entries. Next steps: evaluate with real clicks and ranking metrics (Precision@K, Recall@K, NDCG), add behavioural signals, improve cold-start handling, and de-duplicate the catalogue.

## Files

- `capstone_recommender.ipynb`: full analysis and models
- `Capstone_Presentation.pdf`: final presentation

## Tools

Python, pandas, scikit-learn, Surprise, TensorFlow / Keras, WordCloud, Matplotlib
