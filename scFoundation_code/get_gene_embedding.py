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

# %%
# paths
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
adata_ctrl = adata_hvg[adata_hvg.obs["control"]].copy()
adata_ctrl

# %%
n_sub = min(250, adata_ctrl.n_obs)
idx = np.random.choice(adata_ctrl.n_obs, size=n_sub, replace=False)
adata_ctrl_sub = adata_ctrl[idx].copy()

# %%
GENE_TSV = "/net/dali/home/mscbio/yul700/ML_project/scFoundation/OS_scRNA_gene_index.19264.tsv"
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
out_path = "/net/dali/home/mscbio/yul700/ML_project/scFoundation/inputs/fibroblast_HVG_control_1000.npy"
make_expression_npy(adata_ctrl_sub, gene_list, out_path, rows=1024, dtype=np.float32)

# %%
INPUT = "/net/dali/home/mscbio/yul700/ML_project/scFoundation/inputs/fibroblast_HVG_control_1000.npy"
OUTPUT = "/net/dali/home/mscbio/yul700/ML_project/scFoundation/outputs/single_cell_data"

%cd /net/dali/home/mscbio/yul700/ML_project/scFoundation/model
!python -u ./get_embedding.py \
  --task_name fibro_ctrl_gene_embeddings \
  --input_type singlecell \
  --output_type gene_batch \
  --data_path "$INPUT" \
  --pre_normalized T \
  --save_path "$OUTPUT" \
  --tgthighres f1 2>&1 | tee ./run_gene_ctrl.log


# %%
EMB_GENE = "/net/dali/home/mscbio/yul700/ML_project/scFoundation/outputs/single_cell_data/fibro_ctrl_gene_embeddings_01B-resolution_singlecell_gene_embedding_f1_resolution.npy"

gene_embedding = np.load(EMB_GENE, mmap_mode="r")
print(gene_embedding.shape)

gene_emb_avg = gene_embedding.mean(axis=0)
print(gene_emb_avg.shape)

# %%
gene_emb_avg

# %%
src = pd.Index([str(g) for g in gene_list])                # gene order for gene_emb_avg
vars_idx = pd.Index(adata_ctrl_sub.var_names.astype(str)) 

# %%
matched = src[src.isin(vars_idx)]
n_available = matched.size
n_select = min(3000, n_available)
selected_genes = matched[:n_select]

# get corresponding rows from gene_emb_avg
src_positions = src.get_indexer(selected_genes)           # indices into gene_emb_avg / gene_list
emb_rows = gene_emb_avg[src_positions] 

# %%
emb_dim = emb_rows.shape[1]
all_emb = np.full((adata_ctrl_sub.n_vars, emb_dim), np.nan, dtype=emb_rows.dtype)

# map selected genes into the adata.var positions and assign embeddings
var_positions = vars_idx.get_indexer(selected_genes)      # indices into adata.var_names
all_emb[var_positions] = emb_rows

# %%
adata_ctrl_sub.varm["scFoundation_gene_emb"] = all_emb

print(f"overlap_found={n_available}, added={n_select}, emb_dim={emb_dim}, adata_ctrl_sub.varm['scFoundation_gene_emb'].shape={adata_ctrl_sub.varm['scFoundation_gene_emb'].shape}")

# %%
# keep only genes that have embeddings in adata_ctrl_sub.varm["scFoundation_gene_emb"]
emb = adata_ctrl_sub.varm.get("scFoundation_gene_emb")
if emb is None:
    raise ValueError("adata_ctrl_sub.varm['scFoundation_gene_emb'] not found")

has_emb = ~np.isnan(emb).all(axis=1)   # True for genes with at least one non-NaN value
n_before = adata_ctrl_sub.n_vars
n_after = int(has_emb.sum())
print(f"Filtering genes: before={n_before}, after={n_after}, removed={n_before - n_after}")

# subset to keep only genes with embeddings
adata_ctrl_sub = adata_ctrl_sub[:, has_emb].copy()

# %%
out_fname = "/net/dali/home/mscbio/yul700/ML_project/adata_ctrl_sub.h5ad"
adata_ctrl_sub.write_h5ad(out_fname, compression="lzf")


