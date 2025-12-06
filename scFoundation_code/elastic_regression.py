# %%
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse

# %%
adata = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad")
adata


# %%
adata_ctrl = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/adata_ctrl_sub.h5ad")
adata_ctrl

# %%
import numpy as np

# 1. Get gene symbols from both AnnData objects
genes_ctrl = adata_ctrl.var["gene_name"].astype(str).values
genes_full = adata.var["gene_name"].astype(str).values

print("n_ctrl_genes:", len(genes_ctrl))
print("n_full_genes:", len(genes_full))

# 2. Find overlap using gene_name, not var_names
genes_set_full = set(genes_full)
genes_to_use = [g for g in genes_ctrl if g in genes_set_full]

print(f"adata_ctrl requested {len(genes_ctrl)} genes; "
      f"{len(genes_to_use)} found in full adata.")

missing = [g for g in genes_ctrl if g not in genes_set_full]
if missing:
    print(f"{len(missing)} missing in adata (examples):", missing[:10])

# 3. Get indices in full adata by gene_name, then subset
#    Make a mapping: gene_name -> index in adata.var
name_to_idx_full = {g: i for i, g in enumerate(genes_full)}
idx_full = [name_to_idx_full[g] for g in genes_to_use]

adata_subset = adata[:, idx_full].copy()
print("Filtered adata shape:", adata_subset.shape)


# %%
adata_subset

# %%
group_key = "guide_target"   # column in .obs with perturbation target

X_means = []
obs_rows = []

# choose which matrix to average (raw counts layer if present, else .X)
use_layer = "counts" if "counts" in adata_subset.layers else None

for tgt in adata_subset.obs[group_key].unique():
    mask = adata_subset.obs[group_key] == tgt
    sub = adata_subset[mask]

    # pick matrix
    if use_layer is not None:
        X = sub.layers[use_layer]
    else:
        X = sub.X

    # mean over cells (rows)
    if sparse.issparse(X):
        mean_expr = np.asarray(X.mean(axis=0)).ravel()
    else:
        mean_expr = X.mean(axis=0)

    X_means.append(mean_expr)
    obs_rows.append({
        group_key: tgt,
        "n_cells": sub.n_obs,
    })

X_means = np.vstack(X_means)
obs_new = pd.DataFrame(obs_rows).set_index(group_key)

# one row per perturbation target, columns = genes in adata_subset
adata_subset_by_target = sc.AnnData(
    X=X_means,
    obs=obs_new,
    var=adata_subset.var.copy(),
)

print("adata_subset:", adata_subset.shape,
      "→ by target:", adata_subset_by_target.shape)

# %%
out_fname = "/net/dali/home/mscbio/yul700/ML_project/adata_subset.h5ad"
adata_subset.write_h5ad(out_fname, compression="lzf")

# %%
adata_subset = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/adata_subset.h5ad")
adata_subset

# %%
adata = adata_subset          # just to have a short name
group_key = "guide_target"    # perturbation ID column in .obs

# choose which matrix to use (raw counts if available, else X)
use_layer = "counts" if "counts" in adata.layers else None

X_means = []
obs_rows = []

for tgt in adata.obs[group_key].unique():
    mask = adata.obs[group_key] == tgt
    sub = adata[mask]

    # pick matrix
    X = sub.layers[use_layer] if use_layer is not None else sub.X

    # average over cells
    if sparse.issparse(X):
        mean_expr = np.asarray(X.mean(axis=0)).ravel()
    else:
        mean_expr = X.mean(axis=0)

    X_means.append(mean_expr)
    obs_rows.append({
        group_key: tgt,
        "n_cells": sub.n_obs,
        # keep control flag if present
        "control": bool(sub.obs["control"].iloc[0]) if "control" in sub.obs else False,
    })

X_means = np.vstack(X_means)
obs_new = pd.DataFrame(obs_rows).set_index(group_key)

adata_by_target = sc.AnnData(
    X=X_means,
    obs=obs_new,
    var=adata.var.copy(),
)

print("By-target:", adata_by_target.shape)

# %%
if "control" in adata_by_target.obs.columns:
    ctrl_mask = adata_by_target.obs["control"].astype(bool).values
    assert ctrl_mask.sum() > 0, "No control rows found in adata_by_target.obs['control']"

    X_ctrl = adata_by_target.X[ctrl_mask]
    # average across any control targets
    if sparse.issparse(X_ctrl):
        ctrl_mean = np.asarray(X_ctrl.mean(axis=0)).ravel()
    else:
        ctrl_mean = X_ctrl.mean(axis=0)
else:
    # fallback: if you have a specific guide_target label that is control
    control_target = "NT"  # <- change this to your actual control name
    ctrl_idx = np.where(adata_by_target.obs_names == control_target)[0]
    assert len(ctrl_idx) > 0, f"Control target {control_target!r} not found"

    X_ctrl = adata_by_target.X[ctrl_idx]
    if sparse.issparse(X_ctrl):
        ctrl_mean = np.asarray(X_ctrl.mean(axis=0)).ravel()
    else:
        ctrl_mean = X_ctrl.mean(axis=0)


# %%
eps = 1e-6  # pseudocount

X_tgt = adata_by_target.X
if sparse.issparse(X_tgt):
    X_tgt = X_tgt.A  # to dense

# broadcast ctrl_mean to all targets
logFC = np.log2((X_tgt + eps) / (ctrl_mean[None, :] + eps))  # shape: (n_targets, n_genes)

print("logFC shape:", logFC.shape)  # (n_targets, n_genes)


# %%
adata_by_target.layers["logFC_vs_ctrl"] = logFC

# %%
logFC_df = pd.DataFrame(
    logFC,
    index=adata_by_target.obs_names,      # guide_target
    columns=adata.var["gene_name"],    # genes
)
logFC_df.head()


# %%
genes_emb = adata_ctrl.var["gene_name"].astype(str).values   # length 1696

# gene names for logFC (columns)
genes_log = logFC_df.columns.astype(str).values              # length 1696

# intersection (in case order differs or some miss)
common_genes = sorted(set(genes_emb) & set(genes_log))
print("common genes:", len(common_genes))

# %%
# maps: gene_name -> position
idx_emb_map = {g: i for i, g in enumerate(genes_emb)}
idx_log_map = {g: i for i, g in enumerate(genes_log)}

# same gene order for both X and Y
gene_order = common_genes
idx_emb = [idx_emb_map[g] for g in gene_order]
idx_log = [idx_log_map[g] for g in gene_order]


# %%
# X: gene embeddings, shape (n_genes, emb_dim)
X = adata_ctrl.varm["scFoundation_gene_emb"][idx_emb, :]   # (n_genes, emb_dim)

# Y: logFC, we want (n_genes, n_targets)
# logFC_df is (n_targets, n_genes) → subset columns, then transpose
Y = logFC_df.iloc[:, idx_log].T.values                     # (n_genes, n_targets)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# %%
targets = logFC_df.index.astype(str).values  # (1837,)

np.savez(
    "scfound_logFC_elasticnet_data.npz",
    X=X.astype(np.float32),       # (n_genes, emb_dim)
    Y=Y.astype(np.float32),       # (n_genes, n_targets)
    genes=np.array(gene_order),   # (n_genes,)
    targets=targets,              # (n_targets,)
)

# %%
data = np.load("scfound_logFC_elasticnet_data.npz", allow_pickle=True)

X = data["X"]          # (n_genes, emb_dim)
Y = data["Y"]          # (n_genes, n_targets)
genes = data["genes"]  # gene names in row order of X/Y
targets = data["targets"]  # target names in column order of Y


# %%
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.preprocessing import StandardScaler




# %%
# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# %%
enet_cv = MultiTaskElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    alphas=None,      # let sklearn choose
    cv=5,
    n_jobs=-1,
    random_state=0,
)
enet_cv.fit(X_scaled, Y)

print("Best alpha:", enet_cv.alpha_)
print("Best l1_ratio:", enet_cv.l1_ratio_)

# %%
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import StandardScaler

# 1) Scale features (very important for elastic net)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) Define the multitask elastic net
enet = MultiTaskElasticNet(
    alpha=0.1,      # overall regularization strength
    l1_ratio=0.5,   # 0 = ridge, 1 = lasso, in between = elastic net
    fit_intercept=True,
    max_iter=5000,
    random_state=0,
)

# 3) Fit
enet.fit(X_scaled, Y)

print("W shape:", enet.coef_.shape)      # (n_tasks, n_features) = (1837, 512)
print("b shape:", enet.intercept_.shape) # (n_tasks,)


# %%



