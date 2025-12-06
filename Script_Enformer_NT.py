import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gzip
import pyfaidx
import kipoiseq
from kipoiseq import Interval
SEQUENCE_LENGTH = 196608

class Enformer:

    def __init__(self, tfhub_url):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence, target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]
        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = tf.reduce_sum(target_mask[tf.newaxis] * self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass
        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)

class EnformerScoreVariantsRaw:

    def __init__(self, tfhub_url, organism='human'):
        self._model = Enformer(tfhub_url)
        self._organism = organism

    def predict_on_batch(self, inputs):
        ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
        alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]
        return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)

class EnformerScoreVariantsNormalized:

    def __init__(self, tfhub_url, transform_pkl_path, organism='human'):
        assert organism == 'human', 'Transforms only compatible with organism=human'
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            transform_pipeline = joblib.load(f)
        self._transform = transform_pipeline.steps[0][1]

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)

class EnformerScoreVariantsPCANormalized:

    def __init__(self, tfhub_url, transform_pkl_path, organism='human', num_top_features=500):
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            self._transform = joblib.load(f)
        self._num_top_features = num_top_features

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)[:, :self._num_top_features]

class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval, **kwargs):
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom, max(interval.start, 0), min(interval.end, chromosome_length))
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom, trimmed_interval.start + 1, trimmed_interval.stop).seq).upper()
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

def variant_generator(vcf_file, gzipped=False):
    """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""

    def _open(file):
        return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)
    with _open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            chrom, pos, id, ref, alt_list = line.split('\t')[:5]
            for alt in alt_list.split(','):
                yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos, ref=ref, alt=alt, id=id)

def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

def variant_centered_sequences(vcf_file, sequence_length, gzipped=False, chr_prefix=''):
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=FastaStringExtractor(fasta_file))
    for variant in variant_generator(vcf_file, gzipped=gzipped):
        interval = Interval(chr_prefix + variant.chrom, variant.pos, variant.pos)
        interval = interval.resize(sequence_length)
        center = interval.center() - interval.start
        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)
        yield {'inputs': {'ref': one_hot_encode(reference), 'alt': one_hot_encode(alternate)}, 'metadata': {'chrom': chr_prefix + variant.chrom, 'pos': variant.pos, 'id': variant.id, 'ref': variant.ref, 'alt': variant.alt}}

def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()

def pad_sequence(seq, target_len=196608):
    if len(seq) >= target_len:
        return seq[:target_len]
    pad_len = target_len - len(seq)
    left_pad = pad_len // 2
    right_pad = pad_len - left_pad
    return 'N' * left_pad + seq + 'N' * right_pad
import torch
from enformer_pytorch import Enformer
model = Enformer.from_hparams(dim=1536, depth=11, heads=8, output_heads=dict(human=5313, mouse=1643), target_length=896)
model = Enformer.from_hparams(dim=1536, depth=11, heads=8, output_heads=dict(human=5313, mouse=1643), target_length=896)

def extract_trunk_embeddings(target_interval, pred=False):
    fasta_extractor = FastaStringExtractor('GRCh38.primary_assembly.genome.fa')
    dna_seq = fasta_extractor.extract(target_interval.resize(196608))
    dna_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq_int = torch.tensor([[dna_map[base] for base in dna_seq]])
    sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
    if pred:
        with torch.no_grad():
            output, embeddings = model(seq_int, return_embeddings=True)
            in_embedlyr_seq = torch.tensor(sequence_one_hot).reshape(1, 4, 196608)
            x_final = model.final_pointwise(model.crop_final(model.transformer(model.conv_tower(model.stem(in_embedlyr_seq)))))
        return (output['human'], output['mouse'], x_final)
    else:
        with torch.no_grad():
            in_embedlyr_seq = torch.tensor(sequence_one_hot).reshape(1, 4, 196608)
            x_final = model.final_pointwise(model.crop_final(model.transformer(model.conv_tower(model.stem(in_embedlyr_seq)))))
        return x_final
fasta_extractor = FastaStringExtractor('GRCh38.primary_assembly.genome.fa')
target_interval = kipoiseq.Interval('chr11', 35082742, 35197430)
dna_seq = fasta_extractor.extract(target_interval.resize(196608))
dna_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
seq_int = torch.tensor([[dna_map[base] for base in dna_seq]])
sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
sequence_one_hot
in_embedlyr_seq = torch.tensor(sequence_one_hot).reshape(1, 4, 196608)
x_final = model.final_pointwise(model.crop_final(model.transformer(model.conv_tower(model.stem(in_embedlyr_seq)))))
embeddings_list = []
embeddings = extract_trunk_embeddings(kipoiseq.Interval('chr11', 35082742, 35197430))
embeddings_list.append(embeddings)
embeddings_list[0].shape
import pandas as pd
df_genes = pd.read_table('gencode.v49.primary_assembly.basic.annotation.gtf', header=None)
df_genes
filtered_prot_gene = df_genes[df_genes[8].str.contains('prot') & df_genes[2].str.contains('gene')].copy()
import scanpy as sc
adata_sc = sc.read_h5ad('fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad')
adata_sc.var_names_make_unique()
adata_sc.var.index = adata_sc.var['gene_name']
sc.pl.highest_expr_genes(adata_sc, n_top=20)
sc.pp.filter_cells(adata_sc, min_genes=200)
sc.pp.filter_genes(adata_sc, min_cells=3)
adata_sc
adata_sc.var
adata_sc.var_names
adata_sc.var['MT'] = adata_sc.var_names.str.contains('MT-')
sc.pp.calculate_qc_metrics(adata_sc, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adata_sc.obs
sc.pl.violin(adata_sc, ['n_genes', 'n_genes_by_counts', 'total_counts', 'total_counts_MT', 'pct_counts_MT'], jitter=0.4, multi_panel=True)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
sc.pl.scatter(adata_sc, x='total_counts', y='pct_counts_MT', show=False, ax=axs[0])
sc.pl.scatter(adata_sc, x='total_counts', y='n_genes_by_counts', show=False, ax=axs[1])
adata_sc = adata_sc[(adata_sc.obs.n_genes_by_counts < 4500) & (adata_sc.obs.n_genes_by_counts > 200) & (adata_sc.obs.pct_counts_MT < 5), :].copy()
adata_sc.layers['counts'] = adata_sc.X.copy()
import re
genes = adata_sc.var['gene_name'].astype(str)
mask = ~genes.str.match('^(MT|RPS|RPL)', flags=re.IGNORECASE)
adata_filtered = adata_sc[:, mask]
adata_filtered
sc.pp.highly_variable_genes(adata_filtered, layer='counts', n_top_genes=2000, min_mean=0.0125, max_mean=3, min_disp=0.5, flavor='seurat_v3')
adata_filtered
adata_hvg = adata_filtered[:, adata_filtered.var.highly_variable].copy()
adata_hvg
gene_names = [x.split(';')[2].split('"')[1] for x in filtered_prot_gene[8].to_list()]
filtered_prot_gene['gene_names'] = gene_names
filtered_prot_gene_hvg = filtered_prot_gene[filtered_prot_gene['gene_names'].isin(adata_hvg.var_names.to_list())].copy()
filtered_prot_gene_hvg['gene_names']
adata_sc = adata_hvg[:, adata_hvg.var_names.isin(filtered_prot_gene_hvg['gene_names'])].copy()
adata_sc.write_h5ad('TF_Pert_HVG.h5ad')
from tqdm import tqdm
embeddings_list = []
for _, gene in tqdm(filtered_prot_gene_hvg.iterrows(), total=filtered_prot_gene_hvg.shape[0]):
    chrom = gene[0]
    start = gene[3]
    end = gene[4]
    embeddings = extract_trunk_embeddings(kipoiseq.Interval(chrom, start, end))
    embeddings_list.append(embeddings)
len(embeddings_list)
embeddings_list[0].sum(dim=1)
flatten = [x.sum(dim=1)[0] for x in embeddings_list]
np.vstack(flatten).shape
np.vstack(flatten)
del adata
import anndata as ad
X = np.vstack(flatten)
X = X.T
obs_names = [f'cell_{i}' for i in range(X.shape[0])]
var_names = [f'gene_{j}' for j in range(X.shape[1])]
adata = ad.AnnData(X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
print(adata)
from scipy import sparse
adata = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
adata = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
adata.layers['scaled'] = adata.X.toarray()
sc.pp.scale(adata, max_value=1, layer='scaled')
sc.pp.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
adata.write_h5ad('Embeddings.h5ad')
adata = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
adata.layers['scaled'] = adata.X.toarray()
sc.pp.scale(adata, max_value=1, layer='scaled')
sc.pp.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
adata = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
adata.layers['scaled'] = adata.X.toarray()
sc.pp.scale(adata, max_value=1, layer='scaled')
sc.pp.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
sc.tl.umap(adata, negative_sample_rate=1, min_dist=0.01, spread=1, maxiter=1)
sc.pl.umap(adata)
adata = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
adata.layers['scaled'] = adata.X.toarray()
sc.pp.scale(adata, max_value=1, layer='scaled')
sc.pp.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
sc.tl.umap(adata, negative_sample_rate=1, min_dist=0.01, spread=1, maxiter=1)
sc.pl.umap(adata)
sc.tl.leiden(adata)
adata = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
adata.layers['scaled'] = adata.X.toarray()
sc.pp.scale(adata, max_value=1, layer='scaled')
sc.pp.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
sc.tl.umap(adata, negative_sample_rate=1, min_dist=0.01, spread=1, maxiter=1)
sc.pl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color='leiden')
adata.write_h5ad('Embeddings_Enformer.h5ad')
adata.var.index = filtered_prot_gene_hvg['gene_names']
pd.DataFrame(adata.var_names).to_csv('HVG.csv', header=None)
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv('DEGs.csv', header=None)
from transformers import AutoTokenizer, AutoModelForMaskedLM
model_name = 'InstaDeepAI/nucleotide-transformer-500m-human-ref'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
seqs = ['ATTCCGATTCCGATTCCG', 'ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT']
filtered_prot_gene_hvg
kipoiseq.Interval('chr1', 1001138, 1014540)
fasta_extractor = FastaStringExtractor('GRCh38.primary_assembly.genome.fa')
target_interval = kipoiseq.Interval('chr11', 35082742, 35197430)
dna_seq = fasta_extractor.extract(target_interval.resize(196608))
dna_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
seq_int = torch.tensor([[dna_map[base] for base in dna_seq]])
sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
dna_seq
embeddings_list = []
for _, gene in tqdm(filtered_prot_gene_hvg.iterrows(), total=filtered_prot_gene_hvg.shape[0]):
    chrom = gene[0]
    start = gene[3]
    end = gene[4]
    fasta_extractor = FastaStringExtractor('GRCh38.primary_assembly.genome.fa')
    target_interval = kipoiseq.Interval(chrom, start, end)
    dna_seq = fasta_extractor.extract(target_interval)
    max_length = tokenizer.model_max_length
    enc = tokenizer(dna_seq, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    token_emb = out.hidden_states[-1]
    mask = attention_mask.unsqueeze(-1)
    seq_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    embeddings_list.append(seq_emb)
np.array(embeddings_list).shape
np.vstack(embeddings_list)
X = np.vstack(embeddings_list)
X = X.T
obs_names = [f'cell_{i}' for i in range(X.shape[0])]
var_names = [f'gene_{j}' for j in range(X.shape[1])]
adata = ad.AnnData(X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
print(adata)
adata = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
adata.layers['scaled'] = adata.X.toarray()
sc.pp.scale(adata, max_value=1, layer='scaled')
sc.pp.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
sc.tl.umap(adata, negative_sample_rate=1, min_dist=0.01, spread=1, maxiter=1)
sc.pl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color='leiden')
adata.write_h5ad('Embeddings_NT.h5ad')
adata.var.index = filtered_prot_gene_hvg['gene_names']
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv('DEGs_NT.csv', header=None)
adata_sc.var
pd.DataFrame(pd.DataFrame.sparse.from_spmatrix(adata_sc[adata_sc.obs['guide_target'] == 'MYB'].X).mean(), columns=['MYB'])
df = pd.DataFrame.sparse.from_spmatrix(adata_sc[adata_sc.obs['guide_target'] == 'MYB'].X, columns=adata_sc.var_names)
adata_sc.obs['guide_target'].value_counts()
df
adata_NT = sc.read_h5ad('Embeddings_NT.h5ad')
adata_Enformer = sc.read_h5ad('Embeddings_Enformer.h5ad')
adata_NT = adata_NT[:, ~adata_NT.var_names.duplicated()].copy()
adata_Enformer = adata_Enformer[:, ~adata_Enformer.var_names.duplicated()].copy()
NT_embed = pd.DataFrame.sparse.from_spmatrix(adata_NT.X, columns=adata_sc.var_names)
Enformer_embed = pd.DataFrame.sparse.from_spmatrix(adata_Enformer.X, columns=adata_sc.var_names)
Enformer_embed.T
df.T
expr_cell_by_gene = df.T.T
tf_pert_df = []
for gene in adata_sc.obs['guide_target'].value_counts().index.to_list()[:]:
    print(gene)
    X = pd.DataFrame.sparse.from_spmatrix(adata_sc[adata_sc.obs['guide_target'] == gene].X).values
    pb_log1p = np.log1p(np.expm1(X).mean(axis=0))
    tf_pert_df.append(pb_log1p)
adata_sc[adata_sc.obs['guide_target'] == 'non']
pert_expr = pd.DataFrame(np.array(tf_pert_df), index=adata_sc.obs['guide_target'].value_counts().index.to_list(), columns=adata_sc.var_names)
pert_expr
pert_expr
control = 'non'
ctrl = pert_expr.loc[control]
delta_loge = pert_expr.sub(ctrl, axis=1)
log2fc = delta_loge / np.log(2)
log2fc = log2fc.drop(index=control)
Enformer_embed = Enformer_embed.T
NT_embed = NT_embed.T
NT_embed
Enformer_embed
log2fc
common_genes = NT_embed.index.intersection(log2fc.columns)
X = NT_embed.loc[common_genes]
Y = log2fc.loc[:, common_genes]
print(X.shape, Y.shape)
X
common_genes = NT_embed.index.intersection(log2fc.columns)
X = NT_embed.loc[common_genes].to_numpy()
Y = log2fc.loc[:, common_genes].T.to_numpy()
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import train_test_split
model = Pipeline([('scaler', StandardScaler()), ('mt_enet', MultiTaskElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-4, 1, 40), cv=5, max_iter=5000, verbose=1, n_jobs=-1))], verbose=True)
model.fit(Xtr, Ytr)
Ypred = model.predict(Xte)

def pearson(a, b):
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    return (a * b).sum(axis=0) / (np.sqrt((a * a).sum(axis=0)) * np.sqrt((b * b).sum(axis=0)) + 1e-12)
per_pert_corr = pearson(Ypred, Yte)
print('Mean Pearson across perturbations:', np.nanmean(per_pert_corr))
print('X', X.shape, 'Y', Y.shape)
print('Y finite fraction:', np.isfinite(Y).mean())
print('Mean std across perturbations:', Y.std(axis=0).mean())
print('Mean std across genes:', Y.std(axis=1).mean())
std_per_pert = Y.std(axis=0)
print('min/median/mean/max std:', std_per_pert.min(), np.median(std_per_pert), std_per_pert.mean(), std_per_pert.max())
thr = 0.02
print(f'Perturbations with std < {thr}:', (std_per_pert < thr).sum(), 'out of', Y.shape[1])
import torch.nn as nn
seed = 0
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=seed)
xscaler = StandardScaler()
Xtr_s = xscaler.fit_transform(Xtr)
Xte_s = xscaler.transform(Xte)
y_mean = Ytr.mean(axis=0, keepdims=True)
y_std = Ytr.std(axis=0, keepdims=True) + 1e-06
Ytr_s = (Ytr - y_mean) / y_std
Yte_s = (Yte - y_mean) / y_std
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
Ytr_t = torch.tensor(Ytr_s, dtype=torch.float32, device=device)
Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=device)
Yte_t = torch.tensor(Yte_s, dtype=torch.float32, device=device)
D = X.shape[1]
P = Y.shape[1]

class MultiTaskMLP(nn.Module):

    def __init__(self, D, P, hidden=512, bottleneck=64, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, bottleneck), nn.GELU())
        self.out = nn.Linear(bottleneck, P)

    def forward(self, x):
        h = self.net(x)
        return self.out(h)
model = MultiTaskMLP(D, P, hidden=512, bottleneck=64, dropout=0.25).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
loss_fn = nn.MSELoss()

def mean_perturbation_pearson(yhat, y):
    yhat = yhat - yhat.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    num = (yhat * y).sum(dim=0)
    den = torch.sqrt((yhat ** 2).sum(dim=0) * (y ** 2).sum(dim=0) + 1e-12)
    r = num / den
    return (torch.nanmean(r).item(), r.detach().cpu().numpy())
best = -1000000000.0
for epoch in range(1, 1001):
    model.train()
    opt.zero_grad(set_to_none=True)
    pred = model(Xtr_t)
    loss = loss_fn(pred, Ytr_t)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if epoch % 25 == 0:
        model.eval()
        with torch.no_grad():
            pred_te = model(Xte_t)
            te_loss = loss_fn(pred_te, Yte_t).item()
            mean_r, _ = mean_perturbation_pearson(pred_te, Yte_t)
        print(f'epoch {epoch:4d}  train_mse {loss.item():.4f}  test_mse {te_loss:.4f}  meanPearson(per-pert) {mean_r:.4f}')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
seed = 0
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=seed)
xscaler = StandardScaler()
Xtr_s = xscaler.fit_transform(Xtr)
Xte_s = xscaler.transform(Xte)
y_mean = Ytr.mean(axis=0, keepdims=True)
y_std = Ytr.std(axis=0, keepdims=True) + 1e-06
Ytr_s = (Ytr - y_mean) / y_std
Yte_s = (Yte - y_mean) / y_std
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
Ytr_t = torch.tensor(Ytr_s, dtype=torch.float32, device=device)
Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=device)
Yte_t = torch.tensor(Yte_s, dtype=torch.float32, device=device)
D = X.shape[1]
P = Y.shape[1]

class MultiTaskMLP(nn.Module):

    def __init__(self, D, P, hidden=512, bottleneck=64, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, bottleneck), nn.GELU())
        self.out = nn.Linear(bottleneck, P)

    def forward(self, x):
        h = self.net(x)
        return self.out(h)
model = MultiTaskMLP(D, P, hidden=512, bottleneck=64, dropout=0.25).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
loss_fn = nn.MSELoss()

def mean_perturbation_pearson(yhat, y):
    yhat = yhat - yhat.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    num = (yhat * y).sum(dim=0)
    den = torch.sqrt((yhat ** 2).sum(dim=0) * (y ** 2).sum(dim=0) + 1e-12)
    r = num / den
    return (torch.nanmean(r).item(), r.detach().cpu().numpy())

def mean_gene_pearson(yhat, y):
    yhat = yhat - yhat.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (yhat * y).sum(dim=1)
    den = torch.sqrt((yhat ** 2).sum(dim=1) * (y ** 2).sum(dim=1) + 1e-12)
    r = num / den
    return (torch.nanmean(r).item(), r.detach().cpu().numpy())
for epoch in range(1, 1001):
    model.train()
    opt.zero_grad(set_to_none=True)
    pred = model(Xtr_t)
    loss = loss_fn(pred, Ytr_t)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if epoch % 25 == 0:
        model.eval()
        with torch.no_grad():
            pred_te = model(Xte_t)
            te_loss = loss_fn(pred_te, Yte_t).item()
            mean_rp, _ = mean_perturbation_pearson(pred_te, Yte_t)
        print(f'epoch {epoch:4d}  train_mse {loss.item():.4f}  test_mse {te_loss:.4f}  meanPearson(per-pert) {mean_rp:.4f}')
model.eval()
with torch.no_grad():
    pred_te_s = model(Xte_t).detach().cpu().numpy()
pred_te = pred_te_s * y_std + y_mean
true_te = Yte
mse = mean_squared_error(true_te.ravel(), pred_te.ravel())
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_te.ravel(), pred_te.ravel())
r2 = r2_score(true_te.ravel(), pred_te.ravel())

def mean_pearson_np(Yhat, Y, axis):
    if axis == 0:
        A = Yhat - Yhat.mean(0, keepdims=True)
        B = Y - Y.mean(0, keepdims=True)
        num = (A * B).sum(0)
        den = np.sqrt((A * A).sum(0) * (B * B).sum(0) + 1e-12)
        r = num / den
        return (float(np.nanmean(r)), r)
    else:
        A = Yhat - Yhat.mean(1, keepdims=True)
        B = Y - Y.mean(1, keepdims=True)
        num = (A * B).sum(1)
        den = np.sqrt((A * A).sum(1) * (B * B).sum(1) + 1e-12)
        r = num / den
        return (float(np.nanmean(r)), r)
mean_r_pert, r_pert = mean_pearson_np(pred_te, true_te, axis=0)
mean_r_gene, r_gene = mean_pearson_np(pred_te, true_te, axis=1)
r2_per_pert = np.array([r2_score(true_te[:, j], pred_te[:, j]) for j in range(true_te.shape[1])])
r2_macro = float(np.nanmean(r2_per_pert))
print('\n=== Final regression metrics on TEST genes (original logFC units) ===')
print(f'MSE  : {mse:.6f}')
print(f'RMSE : {rmse:.6f}')
print(f'MAE  : {mae:.6f}')
print(f'R^2  : {r2:.6f}   (micro across all entries)')
print(f'R^2  : {r2_macro:.6f}   (macro avg over perturbations)')
print(f'Mean Pearson per perturbation (across genes): {mean_r_pert:.6f}')
print(f'Mean Pearson per gene (across perturbations): {mean_r_gene:.6f}')
topk = 10
best_pert = np.argsort(r_pert)[-topk:][::-1]
worst_pert = np.argsort(r_pert)[:topk]
print('\nTop perturbations by Pearson (indices):', best_pert)
print('Worst perturbations by Pearson (indices):', worst_pert)
