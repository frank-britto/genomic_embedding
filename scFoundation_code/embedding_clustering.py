# %%
import scanpy as sc
import numpy as np

# %%
adata = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/fibro_complete_scFoundation_kmeans10.h5ad")
adata

# %%
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    flavor="seurat",
    subset=False
)

# %%
hvg_mask = adata.var["highly_variable"]
adata = adata[:, hvg_mask].copy()
adata

# %%
hvg_genes = adata.var_names
hvg_genes

# %%
# hvg_genes.to_series().to_csv("/net/dali/home/mscbio/yul700/ML_project/hvg_3000_genes.txt",
#                              index=False,
#                              header=False)

# %%
pert_mask = (adata.obs["control"] == 0)
adata_pert = adata[pert_mask].copy()

# %%
adata_pert

# %%


# %%
from sklearn.model_selection import train_test_split

idx = np.arange(adata_pert.n_obs)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=0)
adata_train = adata_pert[train_idx].copy()
adata_test  = adata_pert[test_idx].copy()


# %%
sc.pp.neighbors(
    adata_train,
    use_rep="X_scFoundation",   # this tells scanpy to use obsm['X_scFoundation']
    n_neighbors=15,
    metric="euclidean"
)
sc.tl.umap(adata_train)

# %%
sc.tl.leiden(adata_train, resolution=1.0)

# %%
sc.pl.umap(adata_train, color=["leiden"], save="_train_leiden.png")

# %%
# save full AnnData object
adata_train.write("fibroblast_scFoundation_pertubed_training_leiden.h5ad")

# %%
# save full AnnData object
adata_test.write("fibroblast_scFoundation_pertubed_testing_leiden.h5ad")

# %%
train_cells = adata.obs_names[train_idx]
test_cells  = adata.obs_names[test_idx]

# %%
train_cells

# %%
train_cells.to_series(name="cell_id").to_csv("train_cells.csv", index=False)
test_cells.to_series(name="cell_id").to_csv("test_cells.csv", index=False)


# %%
adata = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/adata_tf_classes.h5ad")
adata

# %%
# In adata: nice for plotting / value_counts
adata.obs["tf_path_cluster"] = adata.obs["tf_path_cluster"].astype("category")


# %%
sc.pl.umap(adata, color=["tf_path_cluster"], save="_tf_path_cluster.png")

# %%



