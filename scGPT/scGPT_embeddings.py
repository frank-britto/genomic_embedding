#!/usr/bin/env python
# coding: utf-8

# # Computing gene expression embeddings using scGPT

# Importing libraries.

# In[1]:


import warnings
from pathlib import Path
import scanpy as sc
import scgpt as scg
import numpy as np
import pandas as pd
import torch

import os
import pickle

import subprocess
import sys

from scgpt.tasks import embed_data

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter

import gseapy as gp

from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# Just making sure GPU is available.

# In[2]:


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))


# Importing the dataset.

# In[3]:


adata_path = "fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad"
adata = sc.read_h5ad(adata_path)

print(adata)
#print("Layers available:", adata.layers.keys())


# ## Computing the highly variable genes (HVGs)

# Finding the 3000 most variable genes using build-in function in scanpy (based on scGPT tutorial).

# In[4]:


# Select the number of highly variable genes
N_HVG = 3000

# Saving the HVGs as AnnData object
hvg_file = f"adata_hvg_{N_HVG}.h5ad"

# Check if file exists (to avoid recomputing)
if os.path.exists(hvg_file):
    print(f"Loading pre-computed HVG file...")
    adata_hvg = sc.read_h5ad(hvg_file)
    print(f"Highly variable genes: {adata_hvg.shape[1]}")
else:
    print("Computing HVGs...")

    # Standard HVG selection (following scGPT tutorial)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3")
    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    print("Highly variable genes:", adata_hvg.shape[1])

    # Save the computed results
    print(f"Saving HVG results...")
    adata_hvg.write(hvg_file)
    print(f"Saved: {hvg_file}")


# In[5]:


adata_hvg.obs['guide_target']


# ## Regulon-based clustering

# In[11]:


import gseapy as gp

all_libs = gp.get_library_name()
print(all_libs)


# In[12]:


# Annotations libraries
recommended = [
    "GO_BIOLOGICAL_PROCESS_2025",
    "REACTOME_PATHWAYS_2024",
    "KEGG_2021_HUMAN",
    "ChEA_2022",
    "ARCHS4_TFs_Coexp",
    "Tabula_Sapiens",
    "TF_Perturbations_Followed_by_Expression"
]

# Filtering libraries
available = gp.get_library_name()
available_upper = [lib.upper() for lib in available]
libs_to_use = [lib for lib in recommended if lib.upper() in available_upper]

print("Available (subset) recommended libraries:", libs_to_use)
if len(libs_to_use) == 0:
    raise RuntimeError("None of the recommended libraries were found in gseapy.get_library_name().\n"
                       "Run `print(gp.get_library_name())` to inspect available libraries and pick ones present.")

# Map to exact names from available list 
exact_lib_names = []
for want in libs_to_use:
    for lib in available:
        if lib.upper() == want.upper():
            exact_lib_names.append(lib)
            break

print("Using exact library names:", exact_lib_names)

# Building TF list
tf_series = (
    pd.Series(adata_hvg.obs["guide_target"].astype(str).unique())
    .str.upper()
    .replace("NON", np.nan)
    .dropna()
    .reset_index(drop=True)
)
tf_list = tf_series.tolist()
print("Unique TFs (filtered):", len(tf_list))

# Load gene sets (term -> [genes]) for chosen library
pathway_to_genes = {}  # key: "LIB__TERM", value: list of gene symbols (uppercase)

for lib_name in exact_lib_names:
    print(f"Loading library: {lib_name}")
    try:
        lib_dict = gp.get_library(name=lib_name)  # dict: term -> [genes]
    except Exception as e:
        print(f"Could not load {lib_name}: {e}")
        continue

    for term, genes in lib_dict.items():
        term_key = f"{lib_name}__{term}"
        # Some libraries include gene lists as comma-separated strings; handle both
        if isinstance(genes, str):
            gene_list = [g.strip().upper() for g in genes.split(",") if g.strip()]
        else:
            gene_list = [g.upper() for g in genes]
        pathway_to_genes[term_key] = gene_list

print("Total pathways loaded:", len(pathway_to_genes))

# 1 if TF gene symbol appears among pathway members, else 0
pathway_keys = sorted(pathway_to_genes.keys())
tf_path_df = pd.DataFrame(0, index=tf_list, columns=pathway_keys, dtype=np.int8)

for p in pathway_keys:
    genes = set(pathway_to_genes[p])
    # mark TFs that are members of this pathway
    # Note: Many pathways won't include TFs directly; that's fine
    hits = [1 if tf in genes else 0 for tf in tf_list]
    tf_path_df[p] = hits

# Sanity check
n_pathways_with_hits = (tf_path_df.sum(axis=0) > 0).sum()
n_tfs_with_annotations = (tf_path_df.sum(axis=1) > 0).sum()
print(f"Pathways that include at least one TF: {n_pathways_with_hits} / {len(pathway_keys)}")
print(f"TFs annotated with >=1 pathway: {n_tfs_with_annotations} / {len(tf_list)}")

# Filtering pathways to avoid noise
min_tf = 2           # keep pathways that annotate at least 2 TFs
max_tf = int(0.9 * len(tf_list))  # drop if too broad

pathway_mask = (tf_path_df.sum(axis=0) >= min_tf) & (tf_path_df.sum(axis=0) <= max_tf)
tf_path_df_filt = tf_path_df.loc[:, pathway_mask]
print("Filtered pathways kept:", tf_path_df_filt.shape[1])

# Just in case min_tf it's a very harsh threshold, go default to single annotation
if tf_path_df_filt.shape[1] == 0:
    print("No pathways kept after filtering. Relaxing min_tf to 1.")
    min_tf = 1
    pathway_mask = (tf_path_df.sum(axis=0) >= min_tf) & (tf_path_df.sum(axis=0) <= max_tf)
    tf_path_df_filt = tf_path_df.loc[:, pathway_mask]
    print("Filtered pathways kept (relaxed):", tf_path_df_filt.shape[1])

# Computing Jaccard distance
X = tf_path_df_filt.values.astype(bool)  # shape (n_TF, n_pathways)
if X.size == 0 or X.sum() == 0:
    raise RuntimeError("TF × pathway matrix is empty after filtering. Try using different libraries or lowering min_tf.")

jaccard_dist = pairwise_distances(X, metric="jaccard")  

# Hierarchical clustering step
Z = linkage(jaccard_dist, method="average")
k = 20  # desired number of TF clusters (adjustable)
tf_cluster_labels = fcluster(Z, t=k, criterion="maxclust")

# Map to series indexed by TF symbol
tf_cluster_series = pd.Series(tf_cluster_labels, index=tf_path_df_filt.index, name="tf_path_cluster")
print("Cluster distribution (counts):")
print(tf_cluster_series.value_counts().sort_index())

# Mapping each cluster back to each cell
tf_to_cluster = tf_cluster_series.to_dict()
def map_tf_to_cluster(x):
    if not isinstance(x, str):
        return -1
    key = x.upper()
    return int(tf_to_cluster.get(key, -1))

adata_hvg.obs["tf_path_cluster"] = adata_hvg.obs["guide_target"].astype(str).map(map_tf_to_cluster)


# ## Importing scGPT pre-trained model

# In[14]:


MODEL_DIR = Path("./scGPT_human")
MODEL_DIR.mkdir(exist_ok=True)

# Check if vocab.json exists — if yes, we assume checkpoint is already present
if (MODEL_DIR / "vocab.json").exists():
    print("Pretrained scGPT already downloaded in", MODEL_DIR)
else:
    print("Downloading pretrained scGPT to", MODEL_DIR)
    # The pre-trained model is in Google Drive, so we can use gdown to get it
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"])
        import gdown

    # Google Drive folder URL (from scGPT paper)
    gdrive_url = "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
    gdown.download_folder(gdrive_url, output=str(MODEL_DIR), quiet=False, use_cookies=True)

    # Sanity check
    if not (MODEL_DIR / "vocab.json").exists():
        # If by some reason it gets nested, get it back
        nested = list(MODEL_DIR.glob("**/vocab.json"))
        if nested:
            MODEL_DIR = nested[0].parent
            print("Found vocab.json in nested folder:", MODEL_DIR)
        else:
            raise FileNotFoundError("vocab.json not found after download. Probably Google Drive link isn't working anymore!.")
    else:
        print("Pretrained scGPT downloaded successfully to", MODEL_DIR)


# Following the installation tutorial doesn't work as commands are incompatible with Windows. We couldn't get `flash_att` installed with scanpy and the version of pytorch needed for scGPT to work. We use a patch: https://github.com/bowang-lab/scGPT/issues/58

# In[16]:


# Code from the Github responses in https://github.com/bowang-lab/scGPT/issues/58
import platform
if platform.system() == "Windows":
    import os, sys
    # 1) Make sched_getaffinity exist and return empty, so len(...) == 0
    #    (scGPT uses len(os.sched_getaffinity(0)) to pick num_workers)
    os.sched_getaffinity = lambda pid=0: set()

    # 2) Monkey-patch DataLoader to force single-process behavior
    #    This ensures num_workers is always 0 and pin_memory=False (safe for CPU/Windows).
    from types import SimpleNamespace
    try:
        import torch
        import torch.utils.data as _tud
        _orig_DataLoader = _tud.DataLoader

        class _SingleProcessDataLoader(_orig_DataLoader):
            def __init__(self, *args, **kwargs):
                # force no multiprocessing workers
                kwargs['num_workers'] = 0
                # pin_memory True is for CUDA; keep it False to be safe on Windows CPU runs
                kwargs.setdefault('pin_memory', False)
                super().__init__(*args, **kwargs)

        _tud.DataLoader = _SingleProcessDataLoader
        # Also patch the symbol where scGPT might import DataLoader directly
        import torch.utils
        torch.utils.data.DataLoader = _tud.DataLoader

        print("Windows patch applied: sched_getaffinity -> empty, DataLoader -> single-process")
    except Exception as e:
        # torch may not be imported yet; that's fine — we'll ensure patch after torch import.
        print("Windows patch partially applied (torch not yet importable). Will attempt DataLoader patch after torch import.")
        # arrange to patch later if torch isn't importable now
        def _deferred_patch():
            import torch.utils.data as _tud2
            _orig_DataLoader2 = _tud2.DataLoader
            class _SingleProcessDataLoader2(_orig_DataLoader2):
                def __init__(self, *args, **kwargs):
                    kwargs['num_workers'] = 0
                    kwargs.setdefault('pin_memory', False)
                    super().__init__(*args, **kwargs)
            _tud2.DataLoader = _SingleProcessDataLoader2
            import torch.utils
            torch.utils.data.DataLoader = _tud2.DataLoader
            print("Deferred DataLoader patch applied.")
        # attach so user can call after torch becomes available
        sys.modules.setdefault('_deferred_scgpt_loader_patch', SimpleNamespace(apply=_deferred_patch))


# ## Computing the embeddings

# In[17]:


def scgpt_transform(
    adata,
    model_dir,
    embedding_cache_path="scgpt_embeddings.h5ad",
    force_recompute=False,
    batch_size=32,
    n_neighbors=15,
    min_dist=0.1
):
    """
    This function creates the embeddings from scGPT and generates UMAP visualization.
    Creates properly structured AnnData with embeddings in .obsm and gene info preserved.

    Parameters:
    -----------
    adata : AnnData
        Object with the HVGs *suggested*
    model_dir : str or Path
        Path to scGPT model directory 
    embedding_cache_path : str
        Path to save/load cached embeddings (to avoid recomputing)
    force_recompute : bool
        If True, recompute embeddings even if cache exists 
    batch_size : int
        Batch size for scGPT embedding
    n_neighbors : int
        Number of neighbors for UMAP
    min_dist : float
        Minimum distance for UMAP

    Returns:
    --------
    adata_embedded : AnnData
        AnnData with:
        - Shape: (n_cells, n_genes) with proper gene names
        - .obsm['X_scGPT']: embeddings
        - .obsm['X_umap']: UMAP coordinates
    """

    import os
    import scanpy as sc
    import numpy as np
    from scgpt.tasks import embed_data
    import torch

    # Check if cached embeddings exist and user didn't force recomputation
    if os.path.exists(embedding_cache_path) and not force_recompute:
        print(f"\n{'='*60}")
        print(f"Found cached embeddings: {embedding_cache_path}")
        print(f"{'='*60}")

        try:
            adata_embedded = sc.read_h5ad(embedding_cache_path)

            valid = True

            if "X_scGPT" not in adata_embedded.obsm:
                print("Missing 'X_scGPT' in cached file")
                valid = False

            elif adata_embedded.shape[0] != adata.shape[0]:
                print(f"Cell count mismatch: {adata_embedded.shape[0]} vs {adata.shape[0]}")
                valid = False

            elif adata_embedded.shape[1] != adata.shape[1]:
                print(f"Gene count mismatch: {adata_embedded.shape[1]} vs {adata.shape[1]}")
                valid = False

            elif str(adata_embedded.var_names[0]).isdigit():
                print("Gene names are numeric (wrong structure)")
                valid = False

            else:
                print("✓ Structure validated:")
                print(f"  Shape: {adata_embedded.shape} (cells × genes)")
                print(f"  Gene names: {list(adata_embedded.var_names[:5])}")
                print(f"  Embeddings: {adata_embedded.obsm['X_scGPT'].shape}")

            if valid:
                print("Using cached embeddings!\n")

                if "X_umap" not in adata_embedded.obsm:
                    print("Computing UMAP from cached embeddings...")

                    sc.pp.neighbors(
                        adata_embedded,
                        use_rep='X_scGPT',
                        n_neighbors=n_neighbors
                    )

                    sc.tl.umap(adata_embedded, min_dist=min_dist)

                    adata_embedded.write(embedding_cache_path)
                    print("UMAP computed and saved\n")
                else:
                    print("✓ UMAP already computed\n")

                _plot_umap_visualizations(adata_embedded)

                return adata_embedded

            else:
                print("⚠️  Validation failed. Recomputing embeddings...\n")

        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Recomputing embeddings...\n")

    # If we reach here, we need to compute embeddings
    print("Computing scGPT embeddings...")

    if "gene_symbol" not in adata.var.columns:
        if "gene_name" in adata.var.columns:
            adata.var["gene_symbol"] = adata.var["gene_name"]
        else:
            adata.var["gene_symbol"] = adata.var.index

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cols_to_save = [col for col in ['guide_target', 'guide_identity', 'control', 
                                     'gem_group', 'guide_umi_count', 'mt_frac']
                    if col in adata.obs.columns]
    print(f"Preserving obs columns: {cols_to_save}\n")

    print("Running scGPT embedding")
    adata_scgpt_output = embed_data(
        adata,
        model_dir,
        gene_col="gene_symbol",
        obs_to_save=cols_to_save if cols_to_save else None,
        batch_size=batch_size,
        device=device,
        use_fast_transformer=False,
        return_new_adata=True
    )

    print(f"scGPT finished: {adata_scgpt_output.shape}")

    print("\nCreating properly structured AnnData...")

    embeddings = adata_scgpt_output.X.copy()
    print(f"  Extracted embeddings: {embeddings.shape}")

    adata_embedded = adata.copy()

    for col in adata_scgpt_output.obs.columns:
        if col not in adata_embedded.obs.columns:
            adata_embedded.obs[col] = adata_scgpt_output.obs[col]

    adata_embedded.obsm["X_scGPT"] = embeddings

    print(f"\nProperly structured AnnData created:")
    print(f"  Shape: {adata_embedded.shape} (cells × genes)")
    print(f"  Gene names: {list(adata_embedded.var_names[:5])}")
    print(f"  Embeddings in .obsm['X_scGPT']: {adata_embedded.obsm['X_scGPT'].shape}")

    print(f"\nSaving to: {embedding_cache_path}")
    adata_embedded.write(embedding_cache_path)

    print("Computing UMAP from scGPT embeddings...")
    sc.pp.neighbors(adata_embedded, use_rep='X_scGPT', n_neighbors=n_neighbors)
    sc.tl.umap(adata_embedded, min_dist=min_dist)
    print("UMAP computed\n")

    adata_embedded.write(embedding_cache_path)
    print(f"✓ Cache updated with UMAP\n")

    _plot_umap_visualizations(adata_embedded)

    return adata_embedded


def _plot_umap_visualizations(adata):
    """
    Create UMAP visualization plots colored by different features.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    print("Generating UMAP visualizations...")

    n_cells = adata.shape[0]

    if n_cells > 100000:
        print("Large dataset detected. Using rasterization for faster plotting.")

    x = adata.obsm['X_umap'][:, 0]
    y = adata.obsm['X_umap'][:, 1]

    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))

    control_colors = adata.obs['control'].map({True: '#3498db', False: '#e74c3c'})

    axes1[0].scatter(x, y, c=control_colors, s=0.5, alpha=0.5, rasterized=True)
    axes1[0].set_xlabel('UMAP 1')
    axes1[0].set_ylabel('UMAP 2')
    axes1[0].set_title('Control vs Perturbed')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Control'),
        Patch(facecolor='#e74c3c', label='Perturbed')
    ]
    axes1[0].legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

    perturbed_targets = adata.obs.loc[~adata.obs['control'], 'guide_target'].unique()

    if len(perturbed_targets) < 9:
        print(f"Warning: Only {len(perturbed_targets)} perturbations available.")
        sample_targets = list(perturbed_targets)
    else:
        np.random.seed(0)  
        sample_targets = np.random.choice(perturbed_targets, size=9, replace=False)

    print(f"Plotting {len(sample_targets)} random perturbations...")

    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 16))
    axes2 = axes2.flatten()

    for idx, target in enumerate(sample_targets):
        is_target = adata.obs['guide_target'] == target

        axes2[idx].scatter(x, y, c='lightgray', s=0.1, alpha=0.2, rasterized=True)

        axes2[idx].scatter(
            x[is_target],
            y[is_target],
            c='red',
            s=1,
            alpha=0.8,
            rasterized=True
        )

        axes2[idx].set_title(f'{target}\n({is_target.sum()} cells)', fontweight='bold', fontsize=10)
        axes2[idx].set_xticks([])
        axes2[idx].set_yticks([])

    plt.tight_layout()
    plt.show()


# In[13]:


MODEL_DIR = Path("./scGPT_human")

# Generating embeddings with scGPT
adata_embedded = scgpt_transform(
    adata_hvg,
    model_dir=MODEL_DIR,
    embedding_cache_path="scgpt_embeddings_hvg.h5ad",
    force_recompute=False
)


# In[18]:


# (new) embeddings with the regulton-based label
MODEL_DIR = Path("./scGPT_human")

# Generating embeddings with scGPT
adata_embedded = scgpt_transform(
    adata_hvg,
    model_dir=MODEL_DIR,
    embedding_cache_path="scgpt_embeddings_hvg_regulon.h5ad",
    force_recompute=True
)


# ## Leiden clustering and visualization in UMAP

# Running a leiden clustering.

# In[19]:


def cluster_scgpt_embeddings(
    adata_embedded,
    resolution=1.0,
    n_neighbors=15,
    plot_umap=True,
    force_recompute=False,
    save_path=None
):
    """
    Perform Leiden clustering on scGPT embeddings.

    Parameters:
    -----------
    adata_embedded : AnnData
        AnnData with:
        - .obsm['X_scGPT']: scGPT embeddings (n_cells, embedding_dim)
        - .X: gene expression matrix (n_cells, n_genes) - MUST be non-zero for DE!
        - .var: gene metadata with proper gene names
    resolution : float
        Leiden resolution (higher = more clusters)
    n_neighbors : int
        Number of neighbors for graph construction
    plot_umap : bool
        Whether to visualize clusters on UMAP
    force_recompute : bool
        Force recomputation even if clusters exist
    save_path : str or Path
        Path to save updated h5ad (optional)

    Returns:
    --------
    adata_embedded : AnnData
        Updated with .obs['leiden'] containing cluster labels
    """

    # Validate structure
    print("\n1. Validating AnnData structure...")
    if 'X_scGPT' not in adata_embedded.obsm:
        raise ValueError("No 'X_scGPT' in .obsm! Need to run scgpt_transform() first.")

    # (update). Double check we didn't replace genes for the embeddings, but save them differently
    first_gene = str(adata_embedded.var_names[0])
    if first_gene.isdigit():
        raise ValueError(
            "Gene names were lost."
        )

    print(f"Structure valid:")
    print(f"  Shape: {adata_embedded.shape} (cells × genes)")
    print(f"  Gene names: {list(adata_embedded.var_names[:5])}")
    print(f"  Embeddings: {adata_embedded.obsm['X_scGPT'].shape}")

    # Check if clustering already exists to avoid recomputing
    if 'leiden' in adata_embedded.obs and not force_recompute:
        print("\n Leiden clusters already exist")
        n_clusters = adata_embedded.obs['leiden'].nunique()
        print(f"  {n_clusters} clusters found")
        print(f"\nCluster sizes:")
        print(adata_embedded.obs['leiden'].value_counts().sort_index())
        computed_new = False
    else:
        print(f"\n2. Running Leiden clustering (resolution={resolution})...")

        # Compute neighborhood graph on scGPT embeddings
        print(f"   Computing neighbor graph (k={n_neighbors})...")
        sc.pp.neighbors(
            adata_embedded, 
            use_rep='X_scGPT',  # Use embeddings, not raw expression!
            n_neighbors=n_neighbors
        )

        # Run Leiden clustering
        print("   Running Leiden algorithm...")
        sc.tl.leiden(
            adata_embedded, 
            resolution=resolution,
            key_added='leiden'  # Standard key name
        )

        # Summary
        n_clusters = adata_embedded.obs['leiden'].nunique()
        print(f"\n✓ Clustering complete: {n_clusters} clusters found")
        print(f"\nCluster sizes:")
        print(adata_embedded.obs['leiden'].value_counts().sort_index())

        computed_new = True

    # Save if requested
    if save_path is not None and computed_new:
        save_path = Path(save_path)
        print(f"\n3. Saving updated data to: {save_path}")
        adata_embedded.write_h5ad(save_path)
        print("✓ Save complete!")

    # Visualize
    if plot_umap:
        print(f"\n4. Generating UMAP visualization...")

        # Ensure UMAP exists
        if 'X_umap' not in adata_embedded.obsm:
            print("   Computing UMAP coordinates...")
            sc.tl.umap(adata_embedded, min_dist=0.1)

        sc.pl.umap(
            adata_embedded, 
            color='leiden',
            frameon=False,
            title='',
            save='_leiden_umap.png'
        )   
    return adata_embedded


# In[20]:


# We can use the file obtained from this function for the classification task!
adata_embedded = cluster_scgpt_embeddings(
    adata_embedded,
    resolution=0.8,
    save_path="scgpt_embeddings_leiden.h5ad",
    force_recompute = False
)


# ## Enrichment analysis 

# Running a gene expression enrichment analysis to determine if the genes that are differentially expressed on each cluster (with respect to the rest) are enriched in a certain pathway.

# In[16]:


import gseapy

names = gseapy.get_library_name()
reactome_names = [n for n in names if 'reactome' in n.lower()]

print("Reactome-based libraries:")
for lib in reactome_names:
    print(lib)


# In[21]:


def run_differential_expression(
    adata_embedded,
    groupby='leiden',
    method='wilcoxon',
    n_genes=100,
    save_results=True,
    output_dir='de_results'
):
    """
    Run differential expression analysis to find marker genes that are specific to each cluster.

    Parameters:
    -----------
    adata_embedded : AnnData
        Clustered data with expression matrix in .X
    groupby : str
        Column in .obs containing cluster labels
    method : str
        DE test method ('wilcoxon', 't-test', 'logreg')
    n_genes : int
        Number of top genes to store per cluster
    save_results : bool
        Whether to save DE results to CSV
    output_dir : str
        Directory to save results

    Returns:
    --------
    adata_embedded : AnnData
        Updated with DE results in .uns['rank_genes_groups']
    """

    # Some basic info about the amount of clusters we are going to work with
    n_clusters = adata_embedded.obs[groupby].nunique()
    print(f"\nAnalyzing {n_clusters} clusters using {method} test...")
    print(f"Cluster sizes:")
    print(adata_embedded.obs[groupby].value_counts().sort_index())

    # Run differential expression
    print(f"\nRunning Differential Gene Expression Analysis...")
    sc.tl.rank_genes_groups(
        adata_embedded,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        use_raw=False  
    )

    # Saving results for further plotting
    if save_results:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Extract results for each cluster
        result = adata_embedded.uns['rank_genes_groups']
        groups = result['names'].dtype.names

        for cluster in groups:
            df = pd.DataFrame({
                'gene': result['names'][cluster],
                'logfoldchange': result['logfoldchanges'][cluster],
                'pval': result['pvals'][cluster],
                'pval_adj': result['pvals_adj'][cluster],
                'scores': result['scores'][cluster]
            })

            # Save
            out_file = output_dir / f"cluster_{cluster}_markers.csv"
            df.to_csv(out_file, index=False)

        print(f"Saved DEA genes for {len(groups)} clusters")

    return adata_embedded

def plot_cluster_heatmap(
    adata_embedded,
    groupby='leiden',
    n_genes=10,
    figsize=(12, 10),
    save_path=None,
    **kwargs
):
    """
    Plot heatmap of top marker genes per cluster.

    Parameters:
    -----------
    adata_embedded : AnnData
        Data with DE results in .uns['rank_genes_groups']
    groupby : str
        Cluster column
    n_genes : int
        Number of top genes per cluster to show
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure (optional)
    **kwargs : dict
        Additional arguments for sc.pl.rank_genes_groups_heatmap
    """

    if 'rank_genes_groups' not in adata_embedded.uns:
        raise ValueError("Run DEA analysis first.")

    # Plot heatmap
    sc.pl.rank_genes_groups_heatmap(
        adata_embedded,
        n_genes=n_genes,
        groupby=groupby,
        cmap='RdBu_r',
        figsize=figsize,
        show_gene_labels=True,
        dendrogram=True,
        swap_axes=False,  # Genes as rows, clusters as columns
        **kwargs
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

def run_reactome_enrichment(
    adata_embedded,
    groupby='leiden',
    n_genes=100,
    pval_threshold=0.05,
    organism='human',
    save_results=True,
    output_dir='enrichment_results',
    use_gene_col=None  # <- Added
):
    """
    Run Reactome pathway enrichment analysis for each cluster.

    Parameters:
    -----------
    adata_embedded : AnnData
        Data with DE results
    groupby : str
        Cluster column
    n_genes : int
        Number of top genes per cluster to use
    pval_threshold : float
        Adjusted p-value threshold for enrichment
    organism : str
        'human' or 'mouse'
    save_results : bool
        Whether to save results to CSV
    output_dir : str
        Directory to save results
    use_gene_col : str or None
        Column in adata_embedded.var to use for gene symbols (optional)

    Returns:
    --------
    enrichment_results : dict
        Dictionary mapping cluster -> enrichment DataFrame
    """    
    if 'rank_genes_groups' not in adata_embedded.uns:
        raise ValueError("Run DEA analysis first.")

    # Auto-detect gene symbol column if use_gene_col is None
    if use_gene_col is None:
        if 'gene_symbol' in adata_embedded.var.columns:
            use_gene_col = 'gene_symbol'
        elif 'gene_name' in adata_embedded.var.columns:
            use_gene_col = 'gene_name'
        else:
            use_gene_col = False  # fallback, do not convert

    clusters = adata_embedded.obs[groupby].unique()
    n_clusters = len(clusters)
    print(f"\nRunning enrichment for {n_clusters} clusters...")
    print(f"Using top {n_genes} genes per cluster")
    print(f"Organism: {organism}")

    output_dir = Path(output_dir)
    if save_results:
        output_dir.mkdir(exist_ok=True)

    enrichment_results = {}
    result = adata_embedded.uns['rank_genes_groups']

    for cluster in clusters:
        print(f"\n--- Cluster {cluster} ---")

        # Get top marker genes for this cluster
        gene_names = result['names'][str(cluster)][:n_genes]

        # Convert to gene symbols if needed
        if use_gene_col:
            gene_list = []
            for gene_id in gene_names:
                if gene_id in adata_embedded.var_names:
                    idx = adata_embedded.var_names.get_loc(gene_id)
                    symbol = adata_embedded.var.iloc[idx][use_gene_col]
                    if pd.notna(symbol) and symbol != '':
                        gene_list.append(str(symbol))
            print(f"  Converted {len(gene_names)} IDs → {len(gene_list)} gene symbols")
        else:
            gene_list = [str(g) for g in gene_names]

        # Show sample genes
        print(f"  Sample genes: {gene_list[:5]}")
        print(f"  Using {len(gene_list)} marker genes for enrichment")

        if len(gene_list) == 0:
            print(f"  No valid gene symbols found after conversion!")
            enrichment_results[cluster] = pd.DataFrame()
            continue

        try:
            # Run enrichment using Reactome
            print(f"  Querying Reactome database...")
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets='Reactome_Pathways_2024',
                organism='Human' if organism == 'human' else 'Mouse',
                cutoff=pval_threshold,
                no_plot=True
            )
            enr_df = enr.results
            if len(enr_df) > 0:
                enr_df = enr_df[enr_df['Adjusted P-value'] < pval_threshold]
                enrichment_results[cluster] = enr_df
                if save_results:
                    out_file = output_dir / f"cluster_{cluster}_reactome.csv"
                    enr_df.to_csv(out_file, index=False)
            else:
                enrichment_results[cluster] = pd.DataFrame()
        except Exception as e:
            print(f" Error: {e}")
            enrichment_results[cluster] = pd.DataFrame()

    n_enriched = sum(len(df) > 0 for df in enrichment_results.values())
    print(f"\nSummary: {n_enriched}/{n_clusters} clusters have enriched pathways")
    if save_results:
        print(f"Results saved to: {output_dir}/")

    return enrichment_results

def plot_enrichment_dotplot(
    enrichment_results,
    top_n=5,
    figsize=(12, 10),
    save_path=None
):
    """
    Create dot plot of top enriched pathways per cluster.

    Parameters:
    -----------
    enrichment_results : dict
        Results from run_reactome_enrichment()
    top_n : int
        Number of top pathways per cluster to show
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    """

    # Collect top pathways per cluster
    plot_data = []

    for cluster, df in enrichment_results.items():
        if len(df) > 0:
            top_df = df.nsmallest(top_n, 'Adjusted P-value')
            for _, row in top_df.iterrows():
                # Parse overlap - could be "5/100" string or already parsed
                overlap = row['Overlap']
                if isinstance(overlap, str):
                    # Format: "5/100"
                    numerator, denominator = overlap.split('/')
                    gene_ratio = int(numerator) / int(denominator)
                elif isinstance(overlap, (list, tuple)):
                    # Format: [5, 100] or (5, 100)
                    gene_ratio = overlap[0] / overlap[1]
                else:
                    # Already a ratio
                    gene_ratio = float(overlap)

                plot_data.append({
                    'Cluster': f'Cluster {cluster}',
                    'Pathway': row['Term'][:50],  # Truncate long names
                    'Adj_P_value': row['Adjusted P-value'],
                    'Neg_log10_P': -np.log10(row['Adjusted P-value']),
                    'Gene_Ratio': gene_ratio
                })

    if len(plot_data) == 0:
        print("No enriched pathways to plot")
        return

    df_plot = pd.DataFrame(plot_data)

    # Create dot plot
    fig, ax = plt.subplots(figsize=figsize)

    # Pivot for heatmap-style plot
    pivot = df_plot.pivot_table(
        index='Pathway',
        columns='Cluster',
        values='Neg_log10_P',
        fill_value=0
    )

    # Plot
    scatter_data = []
    for cluster in pivot.columns:
        for pathway in pivot.index:
            value = pivot.loc[pathway, cluster]
            if value > 0:
                scatter_data.append({
                    'x': cluster,
                    'y': pathway,
                    'size': value * 20,  # Scale for visibility
                    'color': value
                })

    df_scatter = pd.DataFrame(scatter_data)

    scatter = ax.scatter(
        df_scatter['x'],
        df_scatter['y'],
        s=df_scatter['size'],
        c=df_scatter['color'],
        cmap='Reds',
        edgecolors='black',
        linewidth=0.5,
        alpha=0.8
    )

    # Formatting
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reactome Pathway', fontsize=12, fontweight='bold')
    ax.set_title('Top Enriched Pathways per Cluster', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(fontsize=9)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('-log10(Adj P-value)', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    plt.show()

def export_de_results_to_excel(adata_embedded, groupby='leiden', 
                                output_file='de_results_all_clusters.xlsx'):
    """
    Export all DE results to a single Excel file with one sheet per cluster.

    Parameters:
    -----------
    adata_embedded : AnnData
        Data with DE results
    groupby : str
        Cluster column
    output_file : str
        Output Excel file path
    """

    if 'rank_genes_groups' not in adata_embedded.uns:
        raise ValueError("Run DEA analysis first.")

    print(f"Exporting DE results to {output_file}...")

    result = adata_embedded.uns['rank_genes_groups']
    clusters = result['names'].dtype.names

    # Check if we need to add gene symbols
    first_gene = str(adata_embedded.var_names[0])
    add_symbols = first_gene.startswith('ENSG')

    if add_symbols:
        if 'gene_symbol' in adata_embedded.var.columns:
            symbol_col = 'gene_symbol'
        elif 'gene_name' in adata_embedded.var.columns:
            symbol_col = 'gene_name'
        else:
            print(" No gene symbol column found")
            add_symbols = False

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for cluster in clusters:
            # Extract data
            df = pd.DataFrame({
                'gene_id': result['names'][cluster],
                'logfoldchange': result['logfoldchanges'][cluster],
                'pval': result['pvals'][cluster],
                'pval_adj': result['pvals_adj'][cluster],
                'scores': result['scores'][cluster]
            })

            # Add gene symbols if needed
            if add_symbols:
                symbols = []
                for gene_id in df['gene_id']:
                    if gene_id in adata_embedded.var_names:
                        idx = adata_embedded.var_names.get_loc(gene_id)
                        symbol = adata_embedded.var.iloc[idx][symbol_col]
                        symbols.append(symbol if pd.notna(symbol) else '')
                    else:
                        symbols.append('')
                df.insert(1, 'gene_symbol', symbols)

            # Write to Excel
            sheet_name = f'Cluster_{cluster}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✓ Wrote {len(df)} genes for {sheet_name}")

    print(f"✓ Exported to {output_file}")
    return output_file

# ------------- Pipeline -------------------- #

def complete_de_enrichment_workflow(
    adata_embedded,
    groupby='leiden',
    n_de_genes=100,
    n_heatmap_genes=10,
    n_enrichment_genes=100,
    output_dir='enrichment_results'
):
    """
    Run complete differential expression and enrichment workflow.

    Parameters:
    -----------
    adata_embedded : AnnData
        Clustered data with expression matrix
    groupby : str
        Cluster column
    n_de_genes : int
        Number of DE genes to compute per cluster
    n_heatmap_genes : int
        Number of genes to show in heatmap
    n_enrichment_genes : int
        Number of genes for enrichment analysis
    output_dir : str
        Output directory for all results
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Step 1: Differential Expression
    print("\n Running Differential Expression Analysis")
    adata_embedded = run_differential_expression(
        adata_embedded,
        groupby=groupby,
        n_genes=n_de_genes,
        output_dir=output_dir / 'de_results'
    )

    # Step 2: DEA Heatmap
    print("\n Computing dendrogram for clusters")
    sc.tl.dendrogram(adata_embedded, groupby=groupby)  # <-- recompute dendrogram

    print("\n DEA Heatmap")
    plot_cluster_heatmap(
        adata_embedded,
        groupby=groupby,
        n_genes=n_heatmap_genes,
        save_path=output_dir / 'marker_heatmap.png'
    )

    # Step 3: Enrichment Analysis
    print("\n Running enrichment using Reactome 2024")
    enrichment_results = run_reactome_enrichment(
        adata_embedded,
        groupby=groupby,
        n_genes=n_enrichment_genes,
        output_dir=output_dir / 'enrichment_results'
    )

    # Step 4: Plot Enrichment
    print("\n Visualizing enrichment results")
    plot_enrichment_dotplot(
        enrichment_results,
        save_path=output_dir / 'enrichment_dotplot.png'
    )

    return adata_embedded, enrichment_results


# In[33]:


adata_embedded, enrichment_results = complete_de_enrichment_workflow(
    adata_embedded,
    groupby='leiden',
    n_de_genes=100,
    n_heatmap_genes=10,
    n_enrichment_genes=100,
    output_dir='enrichment_results'
)


# Evaluating individual cluster results.

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Load enrichment results for a specific cluster
cluster_id = '8'  
enrichment_file = f'enrichment_results/enrichment_results/cluster_{cluster_id}_reactome.csv'

df_enr = pd.read_csv(enrichment_file)

# Filter significant results and get top 15
df_top = df_enr[df_enr['Adjusted P-value'] < 0.05].nsmallest(15, 'Adjusted P-value')

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate -log10(p-value) for better visualization
df_top['neg_log10_p'] = -np.log10(df_top['Adjusted P-value'])

# Create horizontal bar plot
bars = ax.barh(
    range(len(df_top)), 
    df_top['neg_log10_p'],
    #color=plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, len(df_top)))
)

# Customize
ax.set_yticks(range(len(df_top)))
ax.set_yticklabels([term[:60] for term in df_top['Term']], fontsize=9)  # Truncate long names
ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12, fontweight='bold')
ax.set_title(f'Enriched Reactome Pathways - Cluster {cluster_id}', 
             fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()  # Most significant at top
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'cluster_{cluster_id}_enrichment_barplot.png', dpi=300, bbox_inches='tight')
plt.show()


# In[26]:


# Load data
df_enr = pd.read_csv(enrichment_file)
df_top = df_enr[df_enr['Adjusted P-value'] < 0.05].nsmallest(20, 'Adjusted P-value')

# Parse overlap ratio (format: "5/100")
df_top['genes_in_pathway'] = df_top['Overlap'].str.split('/').str[0].astype(int)
df_top['pathway_size'] = df_top['Overlap'].str.split('/').str[1].astype(int)
df_top['gene_ratio'] = df_top['genes_in_pathway'] / df_top['pathway_size']
df_top['neg_log10_p'] = -np.log10(df_top['Adjusted P-value'])

# Create dot plot
fig, ax = plt.subplots(figsize=(12, 10))

scatter = ax.scatter(
    df_top['gene_ratio'],
    range(len(df_top)),
    s=df_top['genes_in_pathway'] * 30,  # Size = number of genes
    c=df_top['neg_log10_p'],  # Color = significance
    cmap='YlOrRd',
    edgecolors='black',
    linewidth=0.5,
    alpha=0.8
)

# Customize
ax.set_yticks(range(len(df_top)))
ax.set_yticklabels([term[:70] for term in df_top['Term']], fontsize=9)
ax.set_xlabel('Gene Ratio (Genes in pathway / Pathway size)', 
              fontsize=12, fontweight='bold')
ax.set_title(f'Enrichment Analysis - Cluster {cluster_id}\n(Dot size = number of genes)', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add colorbar for p-value
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('-log10(Adjusted P-value)', fontsize=11, fontweight='bold')

# Add legend for dot sizes
for size in [5, 10, 20]:
    ax.scatter([], [], s=size*30, c='gray', alpha=0.6, 
               edgecolors='black', linewidth=0.5,
               label=f'{size} genes')
ax.legend(scatterpoints=1, frameon=True, labelspacing=1.5, 
          title='Gene Count', loc='lower right')

plt.tight_layout()
plt.show()

