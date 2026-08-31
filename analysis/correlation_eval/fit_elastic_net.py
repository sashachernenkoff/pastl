r"""
Enformer-ElasticNet baseline: grid-search an ElasticNet regressor on frozen
Enformer embeddings.

Loads the per-split embedding tables written by analysis/predict.py --mode
enformer (embeddings_{train,valid,test}.pkl), pools each 896x3072 embedding to a
fixed-length vector (gene-mask pooling with --use_gene_mask, else the central 3
bins), standardizes features, and grid-searches ElasticNet over alpha x l1_ratio,
selecting the combination with the highest validation R^2. Writes test-set
predictions and a tuning log.

Outputs:
  {enformer_dir}/predictions.pkl     protein_id, true_label, prediction
  {logs_dir}/elastic_net_tuning.txt

Usage:
  python fit_elastic_net.py \
      --enformer_dir out/predictions/enformer \
      --use_gene_mask \
      --logs_dir out/analysis/correlation_eval/logs
"""
import argparse
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_PREDICTIONS_DIR = "out/predictions"
_LOGS_DIR = "out/analysis/correlation_eval/logs"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enformer_dir", default=f"{_PREDICTIONS_DIR}/enformer",
                        help="Directory containing the embeddings_{split}.pkl files")
    parser.add_argument("--use_gene_mask", action="store_true",
                        help="If set, pools the full 896x3072 embedding using the gene mask. Otherwise pools the central 3 bins.")
    parser.add_argument("--logs_dir", default=_LOGS_DIR,
                        help="Directory for tuning logs")
    args = parser.parse_args()

    ensure_dir(args.logs_dir)

    print("Loading datasets...")
    train_df = pd.read_pickle(Path(args.enformer_dir) / "embeddings_train.pkl")
    val_df = pd.read_pickle(Path(args.enformer_dir) / "embeddings_valid.pkl")
    test_df = pd.read_pickle(Path(args.enformer_dir) / "embeddings_test.pkl")

    def pool_embeddings(df, use_gene_mask):
        if use_gene_mask:
            emb = np.stack(df["embedding_genemask"].values)
        else:
            emb = np.stack(df["embedding_3bin"].values)
        return emb

    X_train = pool_embeddings(train_df, args.use_gene_mask)
    y_train = train_df["true_label"].values

    X_val = pool_embeddings(val_df, args.use_gene_mask)
    y_val = val_df["true_label"].values

    X_test = pool_embeddings(test_df, args.use_gene_mask)
    y_test = test_df["true_label"].values

    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Grid search over ElasticNet hyperparameters using the validation set
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]

    best_r2 = -float("inf")
    best_params = None
    best_model = None

    print(f"Tuning ElasticNet across {len(alphas) * len(l1_ratios)} combinations...")
    for alpha in alphas:
        for l1 in l1_ratios:
            # We use a smaller max_iter for speed during tuning if needed, but 1000 is default
            model = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            val_preds = model.predict(X_val)
            r2 = r2_score(y_val, val_preds)
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = (alpha, l1)
                best_model = model

    alpha, l1 = best_params
    print(f"Best Validation R²: {best_r2:.4f} (alpha={alpha}, l1_ratio={l1})")

    # Predict on test set
    test_preds = best_model.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_pearson = pearsonr(y_test, test_preds)[0]
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test Pearson r: {test_pearson:.4f}")

    # Save test predictions
    out_df = pd.DataFrame({
        "protein_id": test_df["protein_id"],
        "true_label": test_df["true_label"],
        "prediction": test_preds
    })
    
    out_file = Path(args.enformer_dir) / "predictions.pkl"
    out_df.to_pickle(out_file)
    print(f"Saved ElasticNet test predictions to {out_file}")

    # Save log
    with open(Path(args.logs_dir) / "elastic_net_tuning.txt", "w") as f:
        f.write(f"Elastic Net Tuning Results\n")
        f.write(f"Best Validation R2: {best_r2:.4f}\n")
        f.write(f"Best Alpha: {alpha}\n")
        f.write(f"Best L1 Ratio: {l1}\n")
        f.write(f"Test R2: {test_r2:.4f}\n")
        f.write(f"Test Pearson r: {test_pearson:.4f}\n")

if __name__ == "__main__":
    main()
