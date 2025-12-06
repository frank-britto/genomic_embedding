# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import scanpy as sc
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

# %%
adata = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/fibroblast_scFoundation_pertubed_only_leiden.h5ad")
adata

# %%
tf_df = pd.read_csv("/net/dali/home/mscbio/yul700/ML_project/tf_pathway_clusters.csv")
tf_df.columns = ["tf", "tf_path_cluster"]

tf_to_cluster = dict(zip(tf_df["tf"], tf_df["tf_path_cluster"]))
len(tf_to_cluster), list(tf_to_cluster.items())[:5]

# %%
adata.obs["tf_path_cluster"] = adata.obs["guide_target"].map(tf_to_cluster)
mask_labeled = adata.obs["tf_path_cluster"].notna()
print("Cells with TF-pathway label:", mask_labeled.sum(), "of", adata.n_obs)

# %%
adata_tf = adata[mask_labeled].copy()

# make labels integer
adata_tf.obs["tf_path_cluster"] = adata_tf.obs["tf_path_cluster"].astype(int)

# %%
adata_tf

# %%
out_fname = "/net/dali/home/mscbio/yul700/ML_project/adata_tf_classes.h5ad"
adata_tf.write_h5ad(out_fname, compression="lzf")

# %%
adata = sc.read_h5ad("/net/dali/home/mscbio/yul700/ML_project/adata_tf_classes.h5ad")
adata

# %%
idx = np.arange(adata.n_obs)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=0)
adata_train = adata[train_idx].copy()
adata_test  = adata[test_idx].copy()

# %%
X_train = adata_train.obsm["X_scFoundation"] 
X_train = normalize(StandardScaler().fit_transform(X_train))

# %%
X_test = adata_test.obsm["X_scFoundation"] 
X_test = normalize(StandardScaler().fit_transform(X_test))

# %%
y_train = adata_train.obs['tf_path_cluster'].astype(int).to_numpy()
y_test = adata_test.obs['tf_path_cluster'].astype(int).to_numpy()

# %%
# train random forest (no scaling required for tree-based models)
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=0,
)
rf.fit(X_train, y_train)

# %%
# predictions & probabilities
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

# metrics
print(classification_report(y_test, y_pred))

# %%
classes = rf.classes_
y_test_binarized = label_binarize(y_test, classes=classes)

roc_auc = roc_auc_score(
    y_test_binarized,
    y_proba,
    average="macro",
    multi_class="ovr",
)

print("ROC AUC (macro, OVR):", roc_auc)

# %%
y_train_xgb = y_train - 1
y_test_xgb  = y_test - 1

# %%
from xgboost import XGBClassifier

# train an XGBoost multiclass classifier to predict leiden clusters

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=21,
    n_estimators=200,
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=0,
    tree_method="hist",
)

xgb.fit(X_train, y_train_xgb)

# %%
# predictions & probabilities
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)

# metrics
print(classification_report(y_test_xgb, y_pred_xgb))

# %%
classes = xgb.classes_
y_test_binarized = label_binarize(y_test, classes=classes)
roc_auc_xgb = roc_auc_score(
    y_test_binarized,
    y_proba_xgb,
    average="macro",
    multi_class="ovr",
)
print("XGBoost ROC AUC (macro, OVR):", roc_auc_xgb)

# %%



