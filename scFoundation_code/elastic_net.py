import numpy as np
import pandas as pd

from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_recall_fscore_support,
)
from scipy.stats import pearsonr

# ===== 1. Load data =====
data = np.load("scfound_logFC_elasticnet_data.npz", allow_pickle=True)

X = data["X"]          # (n_genes, emb_dim)
Y = data["Y"]          # (n_genes, n_targets)
genes = data["genes"]  # (n_genes,)
targets = data["targets"]  # (n_targets,)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("n_genes:", X.shape[0], "n_features:", X.shape[1], "n_targets:", Y.shape[1])

# ===== 2. Train/test split over genes =====
X_train, X_test, Y_train, Y_test, genes_train, genes_test = train_test_split(
    X, Y, genes, test_size=0.2, random_state=0, shuffle=True
)

print("Train genes:", X_train.shape[0])
print("Test genes:", X_test.shape[0])

# ===== 3. Scale features =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 4. Fit MultiTask Elastic Net =====
alpha = 0.1
l1_ratio = 0.5

print(f"Fitting MultiTaskElasticNet(alpha={alpha}, l1_ratio={l1_ratio}) ...")
enet = MultiTaskElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    fit_intercept=True,
    max_iter=5000,
    random_state=0,
)
enet.fit(X_train_scaled, Y_train)

print("coef_ shape:", enet.coef_.shape)       # (n_targets, n_features)
print("intercept_ shape:", enet.intercept_.shape)

# ===== 5. Predictions =====
Y_train_pred = enet.predict(X_train_scaled)
Y_test_pred = enet.predict(X_test_scaled)

# ===== 6. Regression metrics (overall) =====
train_r2 = r2_score(Y_train, Y_train_pred, multioutput="uniform_average")
test_r2 = r2_score(Y_test, Y_test_pred, multioutput="uniform_average")

train_mse = mean_squared_error(Y_train, Y_train_pred, multioutput="uniform_average")
test_mse  = mean_squared_error(Y_test, Y_test_pred, multioutput="uniform_average")

train_mae = mean_absolute_error(Y_train, Y_train_pred, multioutput="uniform_average")
test_mae  = mean_absolute_error(Y_test, Y_test_pred, multioutput="uniform_average")

print("\n=== Overall regression metrics ===")
print(f"Train R^2: {train_r2:.4f}")
print(f"Test  R^2: {test_r2:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test  MSE: {test_mse:.4f}")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test  MAE: {test_mae:.4f}")

# ===== 7. Per-task Pearson correlation on test set =====
rows = []
for j in range(Y_test.shape[1]):
    yt = Y_test[:, j]
    yp = Y_test_pred[:, j]
    if np.allclose(yt, yt[0]) or np.allclose(yp, yp[0]):
        r = np.nan
    else:
        r, _ = pearsonr(yt, yp)
    rows.append({"target": targets[j], "pearson_r": r})

per_task_corr_df = pd.DataFrame(rows)
per_task_corr_df.to_csv("per_task_pearson.csv", index=False)
print("\nSaved per-task Pearson correlations to per_task_pearson.csv")
print("Test median Pearson r (ignoring NaN):",
      np.nanmedian(per_task_corr_df["pearson_r"].values))

# ===== 8. Sign-based classification metrics (up vs down) on test set =====
# Flatten all genes × targets
y_true_flat = Y_test.ravel()
y_pred_flat = Y_test_pred.ravel()

# Up-regulated (logFC > 0) = 1, else 0
sign_true = (y_true_flat > 0).astype(int)
sign_pred = (y_pred_flat > 0).astype(int)

acc = accuracy_score(sign_true, sign_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    sign_true, sign_pred, average="binary", zero_division=0
)

print("\n=== Sign-based classification metrics (test set) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 score : {f1:.4f}")
