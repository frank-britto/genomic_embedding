# %%
import scanpy as sc
import numpy as np

# %%
adata = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/fibroblast_scFoundation_pertubed_training_leiden.h5ad")
adata

# %%
perturbed_genes = list(adata.obs["guide_target"].unique())

# %%
import gseapy as gp

libs = gp.get_library_name() 

# %%
[b for b in libs if 'Reactome' in b]

# %%
sc.tl.rank_genes_groups(
    adata,
    groupby="leiden",
    method="wilcoxon",
    n_genes=200
)

# %%
cluster_labels = adata.obs["leiden"].cat.categories
cluster_top_genes = {}
top_n_markers = 100 

for cl in cluster_labels:
    # DE table for this cluster vs rest
    df = sc.get.rank_genes_groups_df(adata, group=cl)
    
    # sort by log fold-change (or "scores")
    df = df.sort_values("logfoldchanges", ascending=False)
    
    genes = df["names"].head(top_n_markers).tolist()
    cluster_top_genes[cl] = genes


# %%
gene_sets = [
    # "GO_Biological_Process_2025",
    # "GO_Molecular_Function_2025",
    "Reactome_Pathways_2024", 
]

# %%
enrich_results = {}

for cl, genes in cluster_top_genes.items():
    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_sets,
        organism="Human",
        cutoff=1.0, 
    )
    enrich_results[cl] = enr.results

# %%
import matplotlib.pyplot as plt

def plot_enrich_bar(res, top_n=10, title=None, subset_gene_set=None):
    """
    res: enrichment dataframe from gseapy.enrichr().results
    top_n: number of pathways to plot
    subset_gene_set: if not None, filter to only that library
                     (e.g. 'Reactome_Pathways_2024' or 'GO_Biological_Process_2025')
    """
    if res is None or res.empty:
        print("No enrichment results.")
        return
    
    df = res.copy()
    
    if subset_gene_set is not None:
        df = df[df["Gene_set"] == subset_gene_set].copy()
        if df.empty:
            print(f"No results for gene set: {subset_gene_set}")
            return
    
    # choose top_n by adjusted p-value
    df = df.sort_values("Adjusted P-value").head(top_n).copy()
    
    # x-axis = -log10(adj p)
    df["neg_log10_p"] = -np.log10(df["Adjusted P-value"].replace(0, 1e-300))
    
    # reverse so most significant is on top
    df = df.iloc[::-1]
    
    if title is None:
        title = "Pathway Enrichment"
    
    plt.figure(figsize=(8, 3 + 0.3 * len(df)))
    plt.barh(df["Term"], df["neg_log10_p"])
    plt.xlabel("-log10(adj p-value)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# %%
res0 = enrich_results["1"]
plot_enrich_bar(
    res0,
    top_n=10,
    title="Cluster 1 – Reactome pathways",
    subset_gene_set="Reactome_Pathways_2024",      # or comment out to mix all libraries
)


# %%



