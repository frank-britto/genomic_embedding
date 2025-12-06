#!/usr/bin/env python
# coding: utf-8

# # Fibroblast dataset exploration using scanpy

# Here we explore the dataset Hs27-CRISPRa-TFs (https://zenodo.org/records/15200179) following the scanpy tutorial (https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html)

# In[1]:


# Core scverse libraries (from https://virtualcellmodels.cziscience.com/quickstart/scgpt-quickstart)
from __future__ import annotations

# Data retrieval
import anndata as ad
import scanpy as sc

# Data processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Checking what's inside the file.

# In[2]:


# Set plotting parameters for Jupyter
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', figsize=(6, 4))
get_ipython().run_line_magic('matplotlib', 'inline')

# Load your data
adata = sc.read_h5ad('fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad')


# In[3]:


print(adata)


# In[4]:


adata.layers


# In[5]:


adata.obs


# ## TF annotation

# For solving a multi-class classification, we will need to group the TF into "families" based on functional annotations, since there are not enough samples for us to solve the major classification problem (i.e., determine the TF that was perturbed). We start by getting the annotations from some datasets.

# In[18]:


import pandas as pd
import gseapy as gp

import gseapy as gp
import pandas as pd
from collections import defaultdict

def get_tf_annotations_from_gene_sets(tf_list, verbose=True):
    """
    Gather comprehensive TF annotations from multiple Enrichr libraries.

    Parameters:
    -----------
    tf_list : list
        List of gene symbols (transcription factors)
    verbose : bool
        Print progress and statistics

    Returns:
    --------
    pd.DataFrame with columns: gene, n_terms, terms
    """

    # Comprehensive library selection
    libraries = {
        # Gene Ontology (multiple aspects)
        'GO_Biological_Process_2023': 'GO:BP',
        'GO_Molecular_Function_2023': 'GO:MF',
        'GO_Cellular_Component_2023': 'GO:CC',

        # Pathways
        'KEGG_2021_Human': 'KEGG',
        'Reactome_2022': 'Reactome',
        'WikiPathway_2023_Human': 'WikiPathways',
        'BioPlanet_2019': 'BioPlanet',

        # Disease associations
        'GWAS_Catalog_2023': 'GWAS',
        'ClinVar_2019': 'ClinVar',
        'DisGeNET': 'DisGeNET',
        'OMIM_Disease': 'OMIM',

        # Tissue/Cell type expression
        'Human_Gene_Atlas': 'Tissue',
        'GTEx_Tissue_Sample_Gene_Expression_Profiles_up': 'GTEx',
        'ARCHS4_Tissues': 'ARCHS4_Tissue',
        'Descartes_Cell_Types_and_Tissue_2021': 'Descartes',
        'Tabula_Sapiens': 'Tabula_Sapiens',

        # Transcription factor targets (circular but useful)
        'ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X': 'ENCODE_ChIP',
        'TRRUST_Transcription_Factors_2019': 'TRRUST',
        'ChEA_2022': 'ChEA',

        # Protein interactions and complexes
        'BioPlex_2017': 'BioPlex',
        'CORUM': 'CORUM',
        'Protein_Protein_Interactions': 'PPI',

        # Drug/Chemical perturbations
        'DrugMatrix': 'DrugMatrix',
        'LINCS_L1000_Chem_Pert_up': 'LINCS_Chem',

        # Phenotypes
        'MGI_Mammalian_Phenotype_Level_4_2021': 'MGI_Pheno',
        'Human_Phenotype_Ontology': 'HPO',

        # Evolutionary/Comparative
        'HomoloGene': 'HomoloGene',

        # Additional functional annotations
        'Jensen_DISEASES': 'Jensen_Disease',
        'Azimuth_Cell_Types_2021': 'Azimuth',
        'PanglaoDB_Augmented_2021': 'PanglaoDB',
    }

    tf_annotations = defaultdict(set)  # Use set to auto-deduplicate
    library_stats = {}

    if verbose:
        print(f"Fetching annotations for {len(tf_list)} TFs from {len(libraries)} libraries...")
        print("=" * 70)

    for lib, category in libraries.items():
        try:
            if verbose:
                print(f"Loading {lib}...", end=" ")

            gene_sets = gp.get_library(name=lib, organism='human')

            n_terms_added = 0
            n_tfs_found = 0

            for term, genes in gene_sets.items():
                # Add category prefix to term for better organization
                annotated_term = f"[{category}] {term}"

                for tf in tf_list:
                    if tf in genes:
                        tf_annotations[tf].add(annotated_term)
                        n_terms_added += 1

            n_tfs_found = sum(1 for tf in tf_list if any(
                term.startswith(f"[{category}]") for term in tf_annotations[tf]
            ))

            library_stats[lib] = {
                'category': category,
                'n_terms_added': n_terms_added,
                'n_tfs_annotated': n_tfs_found
            }

            if verbose:
                print(f"✓ {n_tfs_found} TFs, {n_terms_added} annotations")

        except Exception as e:
            if verbose:
                print(f"✗ Failed: {str(e)[:50]}")
            library_stats[lib] = {'category': category, 'error': str(e)}
            continue

    # Convert to DataFrame
    annotations = []
    for tf in tf_list:
        terms_list = sorted(list(tf_annotations[tf]))
        annotations.append({
            'gene': tf,
            'n_terms': len(terms_list),
            'terms': terms_list
        })

    df = pd.DataFrame(annotations)

    # Print summary statistics
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total TFs processed: {len(tf_list)}")
        print(f"TFs with annotations: {(df['n_terms'] > 0).sum()}")
        print(f"TFs without annotations: {(df['n_terms'] == 0).sum()}")
        print(f"\nAnnotation statistics:")
        print(f"  Mean terms per TF: {df['n_terms'].mean():.1f}")
        print(f"  Median terms per TF: {df['n_terms'].median():.0f}")
        print(f"  Max terms for one TF: {df['n_terms'].max()}")
        print(f"  Total unique terms: {df['terms'].apply(len).sum()}")

        # Category breakdown
        print(f"\nSuccessfully loaded libraries by category:")
        category_counts = defaultdict(int)
        for lib, stats in library_stats.items():
            if 'error' not in stats and stats.get('n_tfs_annotated', 0) > 0:
                category_counts[stats['category']] += 1

        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count} libraries")

    return df


def analyze_annotation_coverage(annotations_df, tf_list):
    """
    Analyze annotation coverage and identify poorly annotated TFs.

    Parameters:
    -----------
    annotations_df : pd.DataFrame
        Output from get_tf_annotations_from_gene_sets
    tf_list : list
        Original list of TFs

    Returns:
    --------
    dict with coverage statistics
    """

    print("\n" + "=" * 70)
    print("ANNOTATION COVERAGE ANALYSIS")
    print("=" * 70)

    # Basic coverage
    n_annotated = (annotations_df['n_terms'] > 0).sum()
    n_unannotated = (annotations_df['n_terms'] == 0).sum()

    print(f"\nCoverage: {n_annotated}/{len(tf_list)} TFs ({100*n_annotated/len(tf_list):.1f}%)")

    # Unannotated TFs
    if n_unannotated > 0:
        unannotated = annotations_df[annotations_df['n_terms'] == 0]['gene'].tolist()
        print(f"\n⚠ {n_unannotated} TFs without annotations:")
        print(f"  {', '.join(unannotated[:10])}")
        if len(unannotated) > 10:
            print(f"  ... and {len(unannotated) - 10} more")

    # Poorly annotated (< 10 terms)
    poorly_annotated = annotations_df[
        (annotations_df['n_terms'] > 0) & (annotations_df['n_terms'] < 10)
    ]

    if len(poorly_annotated) > 0:
        print(f"\n⚠ {len(poorly_annotated)} TFs with < 10 annotations:")
        for _, row in poorly_annotated.head(5).iterrows():
            print(f"  {row['gene']}: {row['n_terms']} terms")

    # Well annotated (> 50 terms)
    well_annotated = annotations_df[annotations_df['n_terms'] > 50]

    if len(well_annotated) > 0:
        print(f"\n✓ {len(well_annotated)} well-annotated TFs (> 50 terms):")
        for _, row in well_annotated.head(5).iterrows():
            print(f"  {row['gene']}: {row['n_terms']} terms")

    # Category distribution
    print("\nAnnotation category distribution:")
    all_terms = []
    for terms in annotations_df['terms']:
        all_terms.extend(terms)

    categories = defaultdict(int)
    for term in all_terms:
        if term.startswith('['):
            cat = term.split(']')[0][1:]
            categories[cat] += 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count} annotations")

    return {
        'n_annotated': n_annotated,
        'n_unannotated': n_unannotated,
        'coverage_pct': 100 * n_annotated / len(tf_list),
        'mean_terms': annotations_df['n_terms'].mean(),
        'median_terms': annotations_df['n_terms'].median(),
    }

# Compiling matrixes
tf_list = adata.obs[~adata.obs['control']]['guide_target'].unique().tolist()
tf_list = [tf for tf in tf_list if tf != 'non']

# Get annotations
annotations = get_tf_annotations_from_gene_sets(tf_list, verbose=True)

# Analyze coverage
stats = analyze_annotation_coverage(annotations, tf_list)

# Show sample annotations for one TF
print("\n" + "=" * 70)
print(f"Sample annotations for {tf_list[0]}:")
print("=" * 70)
sample_terms = annotations[annotations['gene'] == tf_list[0]]['terms'].iloc[0]
for term in sample_terms[:20]:
    print(f"  {term}")
if len(sample_terms) > 20:
    print(f"  ... and {len(sample_terms) - 20} more terms")


# Due to the sparse nature of the annotations we did before, we rely on the dimensionality reduction for determining if functional groups are possible.

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import umap  # pip install umap-learn

# Extract binary feature matrix
X = feature_matrix.values

# ========================================
# APPROACH 1: Dimensionality Reduction (See if structure exists)
# ========================================
print("=" * 60)
print("APPROACH 1: Visualize Data Structure")
print("=" * 60)

# Try multiple methods
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. PCA (linear)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)
axes[0, 0].set_title(f'PCA (explains {pca.explained_variance_ratio_.sum():.2%} variance)')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')

# 2. Truncated SVD (better for sparse data)
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)
axes[0, 1].scatter(X_svd[:, 0], X_svd[:, 1], alpha=0.6, s=30)
axes[0, 1].set_title(f'Truncated SVD (explains {svd.explained_variance_ratio_.sum():.2%} variance)')
axes[0, 1].set_xlabel('Component 1')
axes[0, 1].set_ylabel('Component 2')

# 3. t-SNE (nonlinear, good for visualization)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric='jaccard')
X_tsne = tsne.fit_transform(X)
axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=30)
axes[1, 0].set_title('t-SNE (Jaccard distance)')
axes[1, 0].set_xlabel('t-SNE 1')
axes[1, 0].set_ylabel('t-SNE 2')

# 4. UMAP (often better than t-SNE for sparse data)
umap_model = umap.UMAP(n_components=2, random_state=42, metric='jaccard', n_neighbors=15)
X_umap = umap_model.fit_transform(X)
axes[1, 1].scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6, s=30)
axes[1, 1].set_title('UMAP (Jaccard distance)')
axes[1, 1].set_xlabel('UMAP 1')
axes[1, 1].set_ylabel('UMAP 2')

plt.tight_layout()
plt.show()

print("\n📊 Interpretation:")
print("- If you see clear separated groups → clustering might work")
print("- If you see a continuous cloud → your data doesn't have discrete clusters")
print("- If you see a horseshoe/arch → your data lies on a continuum\n")

# ========================================
# APPROACH 2: Co-occurrence Network Analysis
# ========================================
print("=" * 60)
print("APPROACH 2: Co-occurrence Analysis")
print("=" * 60)

# Compute Jaccard similarity for all pairs
from sklearn.metrics import pairwise_distances

similarity_matrix = 1 - pairwise_distances(X, metric='jaccard')
np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarity

# Find most similar TF pairs
n_top = 20
triu_indices = np.triu_indices_from(similarity_matrix, k=1)
similarities = similarity_matrix[triu_indices]
top_indices = np.argsort(similarities)[-n_top:][::-1]

print(f"\nTop {n_top} most similar TF pairs:")
print("-" * 60)
for idx in top_indices:
    i, j = triu_indices[0][idx], triu_indices[1][idx]
    tf1 = feature_matrix.index[i]
    tf2 = feature_matrix.index[j]
    sim = similarities[idx]

    # Find shared annotations
    shared = feature_matrix.columns[(X[i] == 1) & (X[j] == 1)].tolist()
    n_shared = len(shared)

    print(f"{tf1} ↔ {tf2}: {sim:.3f} similarity ({n_shared} shared terms)")

# ========================================
# APPROACH 3: Hierarchical Feature Selection
# ========================================
print("\n" + "=" * 60)
print("APPROACH 3: Reduce Feature Sparsity")
print("=" * 60)

# Check sparsity
sparsity = 1 - (X.sum() / X.size)
print(f"\nCurrent sparsity: {sparsity:.2%}")
print(f"Matrix shape: {X.shape}")

# Strategy: Keep only features with intermediate frequency
feature_counts = X.sum(axis=0)
min_freq = max(2, int(0.05 * X.shape[0]))  # At least 5% of TFs
max_freq = int(0.95 * X.shape[0])  # At most 95% of TFs

selected_features = (feature_counts >= min_freq) & (feature_counts <= max_freq)
X_filtered = X[:, selected_features]
selected_terms = feature_matrix.columns[selected_features]

print(f"\nFiltered to features appearing in {min_freq}-{max_freq} TFs")
print(f"New matrix shape: {X_filtered.shape}")
print(f"New sparsity: {1 - (X_filtered.sum() / X_filtered.size):.2%}")

# Try clustering on filtered data
if X_filtered.shape[1] > 0:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    sil_scores = []
    cluster_range = range(2, min(11, X_filtered.shape[0]))

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_filtered)
        score = silhouette_score(X_filtered, labels, metric='jaccard')
        sil_scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(cluster_range, sil_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Clustering on filtered features (intermediate frequency)')
    plt.grid(alpha=0.3)
    plt.show()

    print(f"\nBest k on filtered data: {cluster_range[np.argmax(sil_scores)]}")

# ========================================
# APPROACH 4: Community Detection (Graph-based)
# ========================================
print("\n" + "=" * 60)
print("APPROACH 4: Network Community Detection")
print("=" * 60)

try:
    import networkx as nx
    from networkx.algorithms import community

    # Build similarity graph
    threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 75)  # Top 25%

    G = nx.Graph()
    G.add_nodes_from(range(len(feature_matrix)))

    for i in range(len(feature_matrix)):
        for j in range(i+1, len(feature_matrix)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Louvain community detection
    communities = community.greedy_modularity_communities(G, weight='weight')

    print(f"Found {len(communities)} communities")
    for i, comm in enumerate(communities):
        if len(comm) > 1:
            tfs = [feature_matrix.index[idx] for idx in list(comm)[:5]]
            print(f"  Community {i+1}: {len(comm)} TFs, e.g., {', '.join(tfs)}...")

    # Create community labels
    community_labels = np.zeros(len(feature_matrix), dtype=int)
    for i, comm in enumerate(communities):
        for node in comm:
            community_labels[node] = i

    print("\n✓ Community labels stored in 'community_labels'")

except ImportError:
    print("NetworkX not installed. Run: pip install networkx")
    community_labels = None


# ## Understanding the dataset for classification

# In[7]:


# Exploratory Data Analysis for CRISPRa Dataset
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create binary labels: 0 = control, 1 = perturbed
binary_labels = (~adata.obs['control']).astype(int)
class_counts = binary_labels.value_counts().sort_index()
class_percentages = binary_labels.value_counts(normalize=True).sort_index() * 100

print("\nClass Distribution:")
print(f"  Class 0 (Unperturbed/Control): {class_counts[0]:,} cells ({class_percentages[0]:.2f}%)")
print(f"  Class 1 (Perturbed):           {class_counts[1]:,} cells ({class_percentages[1]:.2f}%)")
print(f"\nClass Imbalance Ratio: {class_counts[1] / class_counts[0]:.3f}")

# Plot binary class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
axes[0].bar(['Control (0)', 'Perturbed (1)'], class_counts.values, 
            color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Number of Cells', fontsize=12)
axes[0].set_title('Binary Classification: Class Distribution', fontsize=14, fontweight='bold')
axes[0].ticklabel_format(style='plain', axis='y')
for i, (count, pct) in enumerate(zip(class_counts.values, class_percentages.values)):
    axes[0].text(i, count + 1000, f'{count:,}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Pie chart
colors = ['#3498db', '#e74c3c']
axes[1].pie(class_counts.values, labels=['Control', 'Perturbed'], 
           autopct='%1.1f%%', colors=colors, startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Binary Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# In[47]:


# Compute number of cells per target
cells_per_target = adata.obs["guide_target"].value_counts()

# Exclude control or non-targeting guides (common keywords)
cells_per_target_nc = cells_per_target[~cells_per_target.index.str.contains("non|NTC|ctrl|control", case=False)]

plt.figure(figsize=(6,3.5))
plt.hist(cells_per_target_nc.values, bins=50, color='teal', edgecolor='black', alpha=0.7)
#plt.axvline(cells_per_target_nc.median(), color='red', linestyle='--', label=f"Median: {cells_per_target_nc.median():.0f}")
#plt.xscale('log')
plt.xlabel("Cells per Target Gene")
plt.ylabel("Number of Target Genes")
plt.title("Perturbation coverage")
#plt.legend()
plt.tight_layout()
plt.show()


# ## Understanding the metadata

# By design, a CRISPR activation assay assigns ~6 unique guides to the same gene. The redundancy comes from the need for replicates and to make sure that the gene is been perturbed, but it introduces a "perturbation level" effect, meaning that some genes might be more activated than others.

# In[39]:


print(f"Total cells: {adata.n_obs:,}")
print(f"Total unique guides: {adata.obs['guide_identity'].nunique():,}")
print(f"Total unique target genes: {adata.obs['guide_target'].nunique():,}")
print(f"Control cells: {adata.obs['control'].sum():,} ({100*adata.obs['control'].sum()/adata.n_obs:.1f}%)")
print(f"Perturbed cells: {(~adata.obs['control']).sum():,} ({100*(~adata.obs['control']).sum()/adata.n_obs:.1f}%)")


# In[40]:


# How many guides per target?
guides_per_target = adata.obs.groupby('guide_target')['guide_identity'].nunique().sort_values(ascending=False)
print(f"Average guides per target: {guides_per_target.mean():.2f}")
print(f"Median guides per target: {guides_per_target.median():.0f}")
print(f"Max guides per target: {guides_per_target.max()}")

print("\nTop 10 targets with most guides:")
print(guides_per_target.head(10))


# Each unique single guide RNA (sgRNA) in the CRISPR activation screen is represented in approximately 16 single cells on average. Since each target gene is typically assigned around six distinct sgRNAs, this results in roughly 100 perturbed cells per gene when aggregating across all its sgRNAs. In other words, each sgRNA serves as a technical replicate for the same gene perturbation, and together they provide a measure of both biological and technical variability in the activation strength of that target. The observed medians thus reflect the experimental coverage of perturbations across the dataset, where “cells per guide” indicates the replication depth per individual sgRNA, and “cells per target gene” captures the overall number of cells in which each gene was successfully perturbed.

# In[42]:


cells_per_guide = adata.obs.groupby("guide_identity").size().sort_values(ascending=False)
cells_per_target = adata.obs.groupby("guide_target").size().sort_values(ascending=False)

# Remove controls ("non", "NTC", etc.)
cells_per_guide_noctrl = cells_per_guide[~cells_per_guide.index.str.contains("non", case=False)]
cells_per_target_noctrl = cells_per_target[~cells_per_target.index.str.contains("non", case=False)]

# Figure layout
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.4)

# Cells per guide histogram
axes[0, 0].hist(cells_per_guide.values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
#axes[0, 0].axvline(cells_per_guide.median(), color='red', linestyle='--', label=f"Median: {cells_per_guide.median():.0f}")
axes[0, 0].set_xlabel("Cells per Guide")
axes[0, 0].set_ylabel("Number of Guides")
axes[0, 0].set_title("Distribution of Cells per Guide")
axes[0, 0].set_yscale("log")
#axes[0, 0].legend()

# Cells per target gene histogram (might not be useful!)
axes[0, 1].hist(cells_per_target.values, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].axvline(cells_per_target.median(), color='red', linestyle='--', label=f"Median: {cells_per_target.median():.0f}")
axes[0, 1].set_xlabel("Cells per Target Gene")
axes[0, 1].set_ylabel("Number of Targets")
axes[0, 1].set_title("Distribution of Cells per Target Gene")
axes[0, 1].set_yscale("log")
axes[0, 1].legend()

# Top 20 guide sgRNAs
top_guides = cells_per_guide.head(20)
axes[1, 0].barh(range(len(top_guides)), top_guides.values, color='steelblue')
axes[1, 0].set_yticks(range(len(top_guides)))
axes[1, 0].set_yticklabels(top_guides.index, fontsize=7)
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel("Number of Cells")
axes[1, 0].set_title("Top 20 Guides by Cell Count (All)")

# Top 20 target genes
top_targets = cells_per_target.head(20)
axes[1, 1].barh(range(len(top_targets)), top_targets.values, color='coral')
axes[1, 1].set_yticks(range(len(top_targets)))
axes[1, 1].set_yticklabels(top_targets.index, fontsize=7)
axes[1, 1].invert_yaxis()
axes[1, 1].set_xlabel("Number of Cells")
axes[1, 1].set_title("Top 20 Targets by Cell Count (All)")

# Top 20 guide sgRNAS (excluding controls)
top_guides_nc = cells_per_guide_noctrl.head(20)
axes[2, 0].barh(range(len(top_guides_nc)), top_guides_nc.values, color='slateblue')
axes[2, 0].set_yticks(range(len(top_guides_nc)))
axes[2, 0].set_yticklabels(top_guides_nc.index, fontsize=7)
axes[2, 0].invert_yaxis()
axes[2, 0].set_xlabel("Number of Cells")
axes[2, 0].set_title("Top 20 Guides by Cell Count (No Controls)")

# Top 20 target genes (exluding controls)
top_targets_nc = cells_per_target_noctrl.head(20)
axes[2, 1].barh(range(len(top_targets_nc)), top_targets_nc.values, color='lightsalmon')
axes[2, 1].set_yticks(range(len(top_targets_nc)))
axes[2, 1].set_yticklabels(top_targets_nc.index, fontsize=7)
axes[2, 1].invert_yaxis()
axes[2, 1].set_xlabel("Number of Cells")
axes[2, 1].set_title("Top 20 Targets by Cell Count (No Controls)")

plt.tight_layout()
plt.show()


# Recall that `guide_identity` is the sgRNA, while `guide_target` is the gene that is been perturbed by the sgRNA. This is a many-to-one relationship as ~6 sgRNAs will target the same gene.

# In[9]:


adata.obs.groupby('guide_identity').size()


# In[10]:


# Coverage thresholds
thresholds = [10, 20, 50, 100, 200]
print("Guides with at least N cells:")
for thresh in thresholds:
    n_guides = (cells_per_guide >= thresh).sum()
    print(f"  ≥{thresh:3d} cells: {n_guides:5,} guides ({100*n_guides/len(cells_per_guide):.1f}%)")

print("\nTargets with at least N cells:")
for thresh in thresholds:
    n_targets = (cells_per_target >= thresh).sum()
    print(f"  ≥{thresh:3d} cells: {n_targets:5,} targets ({100*n_targets/len(cells_per_target):.1f}%)")


# In[11]:


# Get non-control targets
perturbed_cells = adata[~adata.obs['control']]
unique_targets = perturbed_cells.obs['guide_target'].unique()
print(f"Total unique genes targeted: {len(unique_targets)}")

# Most and least covered targets
print(f"\nTop 20 most covered targets (by cell count):")
for i, (target, count) in enumerate(cells_per_target.head(20).items(), 1):
    n_guides = guides_per_target[target]
    print(f"{i:2d}. {target:20s}: {count:5,} cells, {n_guides:2d} guides")

print(f"\nBottom 20 least covered targets (by cell count):")
for i, (target, count) in enumerate(cells_per_target.tail(20).items(), 1):
    n_guides = guides_per_target[target]
    print(f"{i:2d}. {target:20s}: {count:5,} cells, {n_guides:2d} guides")


# In[12]:


control_cells = adata[adata.obs['control']]
perturbed_cells = adata[~adata.obs['control']]

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics = ['UMI_count', 'mt_frac', 'guide_umi_count']
for i, metric in enumerate(metrics):
    data_to_plot = [control_cells.obs[metric], perturbed_cells.obs[metric]]
    axes[i].violinplot(data_to_plot, positions=[0, 1], showmeans=True)
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['Control', 'Perturbed'])
    axes[i].set_ylabel(metric)
    axes[i].set_title(f'{metric} Distribution')
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[13]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Scatter plot: Guide UMI vs Total UMI
axes[0].scatter(control_cells.obs['guide_umi_count'], control_cells.obs['UMI_count'],
                s=1, alpha=0.3, label='Control', color='gray')
axes[0].scatter(perturbed_cells.obs['guide_umi_count'], perturbed_cells.obs['UMI_count'],
                s=1, alpha=0.3, label='Perturbed', color='blue')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel('Guide UMI Count')
axes[0].set_ylabel('Total UMI Count')
axes[0].set_title('Guide UMI vs Total UMI')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Histogram: Guide UMI distribution
axes[1].hist([control_cells.obs['guide_umi_count'], perturbed_cells.obs['guide_umi_count']],
             bins=50, color=['gray','blue'], alpha=0.7, label=['Control','Perturbed'])
axes[1].set_yscale('log')
axes[1].set_xlabel('Guide UMI Count')
axes[1].set_ylabel('Number of Cells')
axes[1].set_title('Guide UMI Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

