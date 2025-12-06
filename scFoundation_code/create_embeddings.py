# %%
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


H5 = "./fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad"
os.makedirs("./scFoundation/inputs", exist_ok=True)

adata = sc.read_h5ad(H5)
if "gene_name" in adata.var.columns:
    adata.var_names = adata.var["gene_name"].astype(str)
    adata.var_names_make_unique()

adata

# %%
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
hvg_mask = adata.var["highly_variable"].values
adata_hvg = adata[:, hvg_mask].copy()
adata_hvg

# %%
GENE_TSV = "./scFoundation/OS_scRNA_gene_index.19264.tsv"
gene_df = pd.read_csv(
    GENE_TSV,
    sep="\t",
    header=0,
    usecols=["gene_name"],
    encoding="utf-8-sig"
)

gene_list = (gene_df["gene_name"]
             .astype(str).str.strip()
             .loc[lambda s: s.ne("")] 
             .dropna()
             .drop_duplicates(keep="first"))

print("gene count:", len(gene_list))
assert len(gene_list) == 19264
gene_list = gene_list.tolist()

# %%
def make_expression_npy(adata, gene_list, out_path, rows=1024, dtype=np.float32):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Map target genes to columns in adata
    src = pd.Index(adata.var_names.astype(str))
    gene_list = pd.Index(pd.Series(gene_list, dtype=str))
    take = src.get_indexer(gene_list)              
    present_mask = take >= 0
    present_cols = take[present_mask]

    # Get X as CSR
    X = adata.X.tocsr() if hasattr(adata.X, "tocsr") else sp.csr_matrix(adata.X)

    n_cells = adata.n_obs
    G = len(gene_list)

    # Create memmap-backed .npy
    mm = open_memmap(out_path, mode="w+", dtype=dtype, shape=(n_cells, G))
    mm[:] = np.array(0, dtype=dtype)

    # Write in blocks
    for s in tqdm(range(0, n_cells, rows), desc=f"write -> {os.path.basename(out_path)}"):
        e = min(s + rows, n_cells)
        if present_cols.size:
            block = X[s:e, present_cols].toarray().astype(dtype, copy=False)
            mm[s:e, present_mask] = block

    # Flush & close
    del mm

    arr = np.load(out_path, mmap_mode="r")
    print("Done:", arr.shape, arr.dtype)


# %%
out_path = "./scFoundation/inputs/fibroblast_HVG.npy"
make_expression_npy(adata_hvg, gene_list, out_path, rows=1024, dtype=np.float32)

# %%
INPUT = "/net/dali/home/mscbio/yul700/ML_project/scFoundation/inputs/fibroblast_X_float32_clean.npy"
OUTPUT = "/net/dali/home/mscbio/yul700/ML_project/scFoundation/outputs/single_cell_data"

%cd /net/dali/home/mscbio/yul700/ML_project/scFoundation/model
!python -u ./get_embedding.py \
  --task_name fibro_CRISPRa \
  --input_type singlecell \
  --output_type cell \
  --pool_type all \
  --data_path "$INPUT" \
  --pre_normalized T \
  --version rde \
  --save_path "$OUTPUT" \
  --tgthighres f1 2>&1 | tee ./run.log


# %%
EMB_HVG = "/net/dali/home/mscbio/yul700/ML_project/scFoundation/outputs/single_cell_data/fibro_HVG_01B-resolution_singlecell_cell_embedding_f1_resolution.npy"

embedding = np.load(EMB_HVG, mmap_mode="r")
assert adata_hvg.n_obs == embedding.shape[0], f"Row mismatch: adata_hvg={adata_hvg.n_obs}, emb={embedding.shape[0]}"
adata_hvg.obsm["X_scFoundation"] = np.asarray(embedding)

# %%
def evaluate_kmeans(E_std, k_values=range(5, 21), n_init=50, sample_for_sil=10000):
    results = []
    n = E_std.shape[0]
    idx = np.random.RandomState(42).choice(n, size=min(sample_for_sil, n), replace=False)

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        labels = km.fit_predict(E_std)
        sil = silhouette_score(E_std[idx], labels[idx])
        ch  = calinski_harabasz_score(E_std, labels)
        db  = davies_bouldin_score(E_std, labels)
        results.append({"k": k, "silhouette": sil, "calinski_harabasz": ch,
                        "davies_bouldin": db, "inertia": km.inertia_})
    return results

# %%
embedding_std = normalize(StandardScaler().fit_transform(adata_hvg.obsm["X_scFoundation"]))

# %%
k = 10
labels = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(embedding_std)

# %%
adata_hvg.obs[f"kmeans_{k}"] = pd.Categorical(labels)
adata_hvg

# %%
fname = f"fibro_hvg_scFoundation_kmeans{k}.h5ad"
adata_hvg.write_h5ad(fname, compression="lzf")

# %%
adata_hvg = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/fibro_hvg_scFoundation_kmeans10.h5ad")

# %%
adata = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/fibro_complete_scFoundation_kmeans10.h5ad")
adata

# %%
sc.pp.neighbors(adata, use_rep="X_scFoundation", n_neighbors=15, metric="euclidean")
sc.tl.umap(adata, random_state=42)

# %%
sc.settings.figdir = "./figs"
sc.pl.umap(adata, color=["kmeans_10"], save="_kmeans10_complete.png")

# %%
sc.pp.neighbors(adata_hvg, use_rep="X_scFoundation", n_neighbors=15, metric="euclidean")
sc.tl.umap(adata_hvg, random_state=42)

# %%
sc.settings.figdir = "./figs"
sc.pl.umap(adata_hvg, color=["kmeans_10"], save="_kmeans10_HGV.png")


