#!/usr/bin/env python
# coding: utf-8

# # Solving pseudo-label classification 

# Importing libraries.

# In[26]:


import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score
)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import os
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelBinarizer


# Splitting dataset 80/20 for training and testing.

# In[3]:


# Importing the data
adata = sc.read_h5ad("scgpt_embeddings_leiden.h5ad")

# Extract embeddings and HVG expression for baselines
X_embeddings = adata.obsm['X_scGPT']
X_hvg = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

# Leiden clusters (labels)
y_leiden = LabelEncoder().fit_transform(adata.obs['leiden'])

print(f"\nEmbeddings shape: {X_embeddings.shape}")
print(f"HVG expression shape: {X_hvg.shape}")

# Split with stratification
X_emb_train, X_emb_test, y_leiden_train, y_leiden_test = train_test_split(
    X_embeddings, y_leiden, test_size=0.2, random_state=42, stratify=y_leiden
)

X_hvg_train, X_hvg_test, _, _ = train_test_split(
    X_hvg, y_leiden, test_size=0.2, random_state=42, stratify=y_leiden
)

print(f"\nTrain set: {X_emb_train.shape[0]} cells")
print(f"Test set: {X_emb_test.shape[0]} cells")


# ## Random classifier

# In[4]:


np.random.seed(42)
y_random_pred = np.random.choice(np.unique(y_leiden_train), size=len(y_leiden_test))

random_acc = accuracy_score(y_leiden_test, y_random_pred)
random_balanced_acc = balanced_accuracy_score(y_leiden_test, y_random_pred)

print(f"\nRandom Classifier Performance:")
print(f"  Accuracy: {random_acc:.4f}")
print(f"  Balanced Accuracy: {random_balanced_acc:.4f}")


# # Models trained with scGPT embeddings

# ## Random forest

# In[5]:


rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_model.fit(X_emb_train, y_leiden_train)


# In[6]:


# -----------------------------
# 1. Predictions
# -----------------------------
y_pred = rf_model.predict(X_emb_test)  # replace with the appropriate model

# -----------------------------
# 2. Weighted metrics
# -----------------------------
bal_acc = balanced_accuracy_score(y_leiden_test, y_pred)
precision_w = precision_score(y_leiden_test, y_pred, average='weighted')
recall_w = recall_score(y_leiden_test, y_pred, average='weighted')
f1_w = f1_score(y_leiden_test, y_pred, average='weighted')

# AUC (macro OVR)
try:
    y_score = rf_model.predict_proba(X_emb_test)  # for tree-based models
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_leiden_test)
    auc_macro = roc_auc_score(Yb, y_score, average='macro', multi_class='ovr')
except Exception:
    auc_macro = np.nan

# -----------------------------
# 3. Print results
# -----------------------------
print(f"\n✓ Model Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# ## XGBoost

# In[7]:


xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss'
)

xgb_model.fit(X_emb_train, y_leiden_train)


# In[8]:


# -----------------------------
# Predictions
# -----------------------------
y_pred = xgb_model.predict(X_emb_test)

# -----------------------------
# Metrics
# -----------------------------
bal_acc = balanced_accuracy_score(y_leiden_test, y_pred)
precision_w = precision_score(y_leiden_test, y_pred, average='weighted')
recall_w = recall_score(y_leiden_test, y_pred, average='weighted')
f1_w = f1_score(y_leiden_test, y_pred, average='weighted')

# AUC (macro OVR)
try:
    y_score = xgb_model.predict_proba(X_emb_test)
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_leiden_test)
    auc_macro = roc_auc_score(Yb, y_score, average='macro', multi_class='ovr')
except Exception:
    auc_macro = np.nan

# -----------------------------
# Print results
# -----------------------------
print(f"\n✓ XGBoost Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# ## MLP

# In[9]:


# Define MLP architecture
class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(SimpleMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Dataset wrapper
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# In[10]:


# Prepare data loaders
train_dataset = EmbeddingDataset(X_emb_train, y_leiden_train)
test_dataset = EmbeddingDataset(X_emb_test, y_leiden_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

input_dim = X_embeddings.shape[1]
hidden_dim = 256
num_classes = len(np.unique(y_leiden_train))

mlp_model = SimpleMLPClassifier(input_dim, hidden_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
print("\nTraining MLP...")
n_epochs = 50
best_val_acc = 0

for epoch in range(n_epochs):
    mlp_model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = mlp_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {train_loss/len(train_loader):.4f}")


# In[13]:


# -----------------------------
# Make sure model is on the right device
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp_model.to(device)
mlp_model.eval()

y_pred_list = []
y_score_list = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = mlp_model(X_batch)

        # Predicted classes
        preds = outputs.argmax(dim=1)
        y_pred_list.extend(preds.cpu().numpy())

        # Probabilities for AUC
        probs = F.softmax(outputs, dim=1)
        y_score_list.extend(probs.cpu().numpy())

y_pred = np.array(y_pred_list)
y_score = np.array(y_score_list)

# -----------------------------
# Metrics
# -----------------------------
bal_acc = balanced_accuracy_score(y_leiden_test, y_pred)
precision_w = precision_score(y_leiden_test, y_pred, average='weighted')
recall_w = recall_score(y_leiden_test, y_pred, average='weighted')
f1_w = f1_score(y_leiden_test, y_pred, average='weighted')

# AUC (macro OVR)
try:
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_leiden_test)
    auc_macro = roc_auc_score(Yb, y_score, average='macro', multi_class='ovr')
except Exception:
    auc_macro = np.nan

# -----------------------------
# Print results
# -----------------------------
print(f"\n✓ MLP Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# # Models trained using raw (HVGs) data

# ## Random Forest

# In[14]:


rf_hvg_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

rf_hvg_model.fit(X_hvg_train, y_leiden_train)


# In[15]:


# -----------------------------
# Predictions and evaluation
# -----------------------------
y_rf_hvg_pred = rf_hvg_model.predict(X_hvg_test)

# Metrics
bal_acc = balanced_accuracy_score(y_leiden_test, y_rf_hvg_pred)
precision_w = precision_score(y_leiden_test, y_rf_hvg_pred, average='weighted')
recall_w = recall_score(y_leiden_test, y_rf_hvg_pred, average='weighted')
f1_w = f1_score(y_leiden_test, y_rf_hvg_pred, average='weighted')

# AUC (macro OVR)
try:
    y_score = rf_hvg_model.predict_proba(X_hvg_test)
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_leiden_test)
    auc_macro = roc_auc_score(Yb, y_score, average='macro', multi_class='ovr')
except Exception:
    auc_macro = np.nan

# -----------------------------
# Print results
# -----------------------------
print(f"\n✓ Random Forest (HVG) Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# ## XGBoost

# In[16]:


xgb_hvg_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss'
)

xgb_hvg_model.fit(X_hvg_train, y_leiden_train)


# In[17]:


# -----------------------------
# Predictions and evaluation
# -----------------------------
y_xgb_hvg_pred = xgb_hvg_model.predict(X_hvg_test)

# Metrics
bal_acc = balanced_accuracy_score(y_leiden_test, y_xgb_hvg_pred)
precision_w = precision_score(y_leiden_test, y_xgb_hvg_pred, average='weighted')
recall_w = recall_score(y_leiden_test, y_xgb_hvg_pred, average='weighted')
f1_w = f1_score(y_leiden_test, y_xgb_hvg_pred, average='weighted')

# AUC (macro OVR)
try:
    y_score = xgb_hvg_model.predict_proba(X_hvg_test)
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_leiden_test)
    auc_macro = roc_auc_score(Yb, y_score, average='macro', multi_class='ovr')
except Exception:
    auc_macro = np.nan

# -----------------------------
# Print results
# -----------------------------
print(f"\n✓ XGBoost (HVG) Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# ## MLP

# In[18]:


# Prepare data loaders for HVG
train_dataset_hvg = EmbeddingDataset(X_hvg_train, y_leiden_train)
test_dataset_hvg = EmbeddingDataset(X_hvg_test, y_leiden_test)

train_loader_hvg = DataLoader(train_dataset_hvg, batch_size=256, shuffle=True)
test_loader_hvg = DataLoader(test_dataset_hvg, batch_size=256, shuffle=False)

# Initialize model with same architecture
input_dim_hvg = X_hvg.shape[1]
hidden_dim = 256
num_classes = len(np.unique(y_leiden_train))

mlp_hvg_model = SimpleMLPClassifier(input_dim_hvg, hidden_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer_hvg = optim.Adam(mlp_hvg_model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
print(f"\nTraining MLP on HVG expression (input dim: {input_dim_hvg})...")
n_epochs = 50

for epoch in range(n_epochs):
    mlp_hvg_model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader_hvg:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer_hvg.zero_grad()
        outputs = mlp_hvg_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer_hvg.step()

        train_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {train_loss/len(train_loader_hvg):.4f}")


# In[19]:


# -----------------------------
# Make sure model is on the right device
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp_hvg_model.to(device)
mlp_hvg_model.eval()

y_pred_list = []
y_score_list = []

with torch.no_grad():
    for X_batch, _ in test_loader_hvg:
        X_batch = X_batch.to(device)
        outputs = mlp_hvg_model(X_batch)

        # Predicted classes
        preds = outputs.argmax(dim=1)
        y_pred_list.extend(preds.cpu().numpy())

        # Probabilities for AUC
        probs = F.softmax(outputs, dim=1)
        y_score_list.extend(probs.cpu().numpy())

y_pred = np.array(y_pred_list)
y_score = np.array(y_score_list)

# -----------------------------
# Metrics
# -----------------------------
bal_acc = balanced_accuracy_score(y_leiden_test, y_pred)
precision_w = precision_score(y_leiden_test, y_pred, average='weighted')
recall_w = recall_score(y_leiden_test, y_pred, average='weighted')
f1_w = f1_score(y_leiden_test, y_pred, average='weighted')

# AUC (macro OVR)
try:
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_leiden_test)
    auc_macro = roc_auc_score(Yb, y_score, average='macro', multi_class='ovr')
except Exception:
    auc_macro = np.nan

# -----------------------------
# Print results
# -----------------------------
print(f"\n✓ MLP (HVG) Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# Saving all models.
# 

# In[20]:


# Create 'models' folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Dictionary of models to save
models_to_save = {
    "rf_hvg_model": rf_hvg_model,
    "rf_model": rf_model,
    "xgb_hvg_model": xgb_hvg_model,
    "xgb_model": xgb_model,
    "mlp_hvg_model": mlp_hvg_model,
    "mlp_model": mlp_model
}

# Save each model
for name, model in models_to_save.items():
    filepath = os.path.join("models", f"{name}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {name} -> {filepath}")


# # Solving classification with regulon-based clustering

# Splitting data into train and test.

# In[21]:


# Load your embeddings + regulon cluster labels
adata = sc.read_h5ad("scgpt_embeddings_hvg_regulon.h5ad")

# Extract embeddings and HVG expression for baselines
X_embeddings = adata.obsm['X_scGPT']
X_hvg = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

# Regulon-based clusters
y_regulon = LabelEncoder().fit_transform(adata.obs['tf_path_cluster'])

print(f"\nEmbeddings shape: {X_embeddings.shape}")
print(f"HVG expression shape: {X_hvg.shape}")
print(f"Number of clusters: {len(set(y_regulon))}")

# Stratified train-test split based on regulon clusters
X_emb_train, X_emb_test, y_train, y_test = train_test_split(
    X_embeddings, y_regulon,
    test_size=0.2,
    random_state=42,
    stratify=y_regulon
)

X_hvg_train, X_hvg_test, _, _ = train_test_split(
    X_hvg, y_regulon,
    test_size=0.2,
    random_state=42,
    stratify=y_regulon
)

print(f"\nTrain set: {X_emb_train.shape[0]} cells")
print(f"Test set: {X_emb_test.shape[0]} cells")


# ## Raw data

# ## Random Forest

# In[22]:


# -----------------------------
# Initialize and train Random Forest
# -----------------------------
rf_regulon = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_regulon.fit(X_hvg_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_rf_pred = rf_regulon.predict(X_hvg_test)

# -----------------------------
# Metrics
# -----------------------------
bal_acc = balanced_accuracy_score(y_test, y_rf_pred)
precision_w = precision_score(y_test, y_rf_pred, average='weighted')
recall_w = recall_score(y_test, y_rf_pred, average='weighted')
f1_w = f1_score(y_test, y_rf_pred, average='weighted')

# AUC (macro OVR)
try:
    y_score = rf_regulon.predict_proba(X_hvg_test)
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_test)
    auc_macro = roc_auc_score(Yb, y_score, average='macro', multi_class='ovr')
except Exception:
    auc_macro = np.nan

# -----------------------------
# Print results
# -----------------------------
print(f"\n✓ Random Forest (HVG, regulon clusters) Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# ## XGBoost

# In[24]:


# -----------------------------
# Initialize and train XGBoost
# -----------------------------
xgb_regulon = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_regulon.fit(X_hvg_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_xgb_pred = xgb_regulon.predict(X_hvg_test)

# -----------------------------
# Metrics
# -----------------------------
bal_acc = balanced_accuracy_score(y_test, y_xgb_pred)
precision_w = precision_score(y_test, y_xgb_pred, average='weighted')
recall_w = recall_score(y_test, y_xgb_pred, average='weighted')
f1_w = f1_score(y_test, y_xgb_pred, average='weighted')

# AUC (macro OVR)
try:
    y_score = xgb_regulon.predict_proba(X_hvg_test)
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_test)
    auc_macro = roc_auc_score(Yb, y_score, average='macro', multi_class='ovr')
except Exception:
    auc_macro = np.nan

# -----------------------------
# Print results
# -----------------------------
print(f"\n✓ XGBoost (HVG, regulon clusters) Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# ## MLP

# In[27]:


# -----------------------------
# 1. Encode labels to integers
# -----------------------------
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# -----------------------------
# 2. Convert to tensors
# -----------------------------
X_hvg_train_tensor = torch.tensor(X_hvg_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
X_hvg_test_tensor = torch.tensor(X_hvg_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# -----------------------------
# 3. DataLoaders
# -----------------------------
train_dataset_hvg = TensorDataset(X_hvg_train_tensor, y_train_tensor)
test_dataset_hvg = TensorDataset(X_hvg_test_tensor, y_test_tensor)

train_loader_hvg = DataLoader(train_dataset_hvg, batch_size=256, shuffle=True)
test_loader_hvg = DataLoader(test_dataset_hvg, batch_size=256, shuffle=False)

# -----------------------------
# 4. Define MLP
# -----------------------------
class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------
# 5. Initialize model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim_hvg = X_hvg_train_tensor.shape[1]
hidden_dim = 256
num_classes = len(np.unique(y_train_encoded))

mlp_regulon = SimpleMLPClassifier(input_dim_hvg, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_hvg = optim.Adam(mlp_regulon.parameters(), lr=0.001, weight_decay=1e-5)

# -----------------------------
# 6. Training loop
# -----------------------------
n_epochs = 50
for epoch in range(n_epochs):
    mlp_regulon.train()
    train_loss = 0
    for X_batch, y_batch in train_loader_hvg:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_hvg.zero_grad()
        outputs = mlp_regulon(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer_hvg.step()
        train_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        avg_loss = train_loss / len(train_loader_hvg)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

# -----------------------------
# 7. Evaluation
# -----------------------------
mlp_regulon.eval()
y_pred_list = []
y_score_list = []

with torch.no_grad():
    for X_batch, _ in test_loader_hvg:
        X_batch = X_batch.to(device)
        outputs = mlp_regulon(X_batch)
        preds = outputs.argmax(dim=1)
        y_pred_list.extend(preds.cpu().numpy())
        probs = F.softmax(outputs, dim=1)
        y_score_list.extend(probs.cpu().numpy())

y_pred = np.array(y_pred_list)
y_score = np.array(y_score_list)
y_pred_orig = le.inverse_transform(y_pred)

# -----------------------------
# 8. Metrics
# -----------------------------
bal_acc = balanced_accuracy_score(y_test, y_pred_orig)
precision_w = precision_score(y_test, y_pred_orig, average='weighted')
recall_w = recall_score(y_test, y_pred_orig, average='weighted')
f1_w = f1_score(y_test, y_pred_orig, average='weighted')

try:
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_test)
    auc_macro = roc_auc_score(Yb, y_score, average='macro', multi_class='ovr')
except Exception:
    auc_macro = np.nan

# -----------------------------
# 9. Print results
# -----------------------------
print(f"\n✓ MLP (HVG, regulon clusters) Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# ## scGPT

# ### Random Forest

# In[28]:


# -----------------------------
# 1. Remove unannotated cells
# -----------------------------
mask_train = y_train != -1
mask_test = y_test != -1

X_emb_train_clean = X_emb_train[mask_train]
y_train_clean = y_train[mask_train]

X_emb_test_clean = X_emb_test[mask_test]
y_test_clean = y_test[mask_test]

# -----------------------------
# 2. Random Forest on embeddings
# -----------------------------
rf_emb = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_emb.fit(X_emb_train_clean, y_train_clean)

# -----------------------------
# 3. Predictions
# -----------------------------
y_rf_emb_pred = rf_emb.predict(X_emb_test_clean)
y_rf_emb_proba = rf_emb.predict_proba(X_emb_test_clean)

# -----------------------------
# 4. Metrics
# -----------------------------
bal_acc = balanced_accuracy_score(y_test_clean, y_rf_emb_pred)
precision_w = precision_score(y_test_clean, y_rf_emb_pred, average='weighted')
recall_w = recall_score(y_test_clean, y_rf_emb_pred, average='weighted')
f1_w = f1_score(y_test_clean, y_rf_emb_pred, average='weighted')

try:
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_test_clean)
    auc_macro = roc_auc_score(Yb, y_rf_emb_proba, average='macro', multi_class='ovr')
except Exception:
    auc_macro = float('nan')

# -----------------------------
# 5. Print results
# -----------------------------
print(f"\n✓ Random Forest (Embeddings, regulon clusters) Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# ### XGBoost

# In[29]:


# -----------------------------
# 1. XGBoost on embeddings
# -----------------------------
xgb_emb = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_emb.fit(X_emb_train_clean, y_train_clean)

# -----------------------------
# 2. Predictions
# -----------------------------
y_xgb_emb_pred = xgb_emb.predict(X_emb_test_clean)
y_xgb_emb_proba = xgb_emb.predict_proba(X_emb_test_clean)

# -----------------------------
# 3. Metrics
# -----------------------------
bal_acc = balanced_accuracy_score(y_test_clean, y_xgb_emb_pred)
precision_w = precision_score(y_test_clean, y_xgb_emb_pred, average='weighted')
recall_w = recall_score(y_test_clean, y_xgb_emb_pred, average='weighted')
f1_w = f1_score(y_test_clean, y_xgb_emb_pred, average='weighted')

try:
    lb = LabelBinarizer()
    Yb = lb.fit_transform(y_test_clean)
    auc_macro = roc_auc_score(Yb, y_xgb_emb_proba, average='macro', multi_class='ovr')
except Exception:
    auc_macro = float('nan')

# -----------------------------
# 4. Print results
# -----------------------------
print(f"\n✓ XGBoost (Embeddings, regulon clusters) Performance:")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Weighted Precision: {precision_w:.4f}")
print(f"  Weighted Recall: {recall_w:.4f}")
print(f"  Weighted F1: {f1_w:.4f}")
print(f"  AUC (macro OVR): {auc_macro:.4f}")


# ### MLP

# In[30]:


# -----------------------------
# 1. Convert embeddings to PyTorch tensors
# -----------------------------
X_emb_train_tensor = torch.tensor(X_emb_train_clean, dtype=torch.float32)
X_emb_test_tensor = torch.tensor(X_emb_test_clean, dtype=torch.float32)

train_dataset_emb = TensorDataset(X_emb_train_tensor, y_train_tensor)
test_dataset_emb = TensorDataset(X_emb_test_tensor, y_test_tensor)

train_loader_emb = DataLoader(train_dataset_emb, batch_size=256, shuffle=True)
test_loader_emb = DataLoader(test_dataset_emb, batch_size=256, shuffle=False)

# -----------------------------
# 2. Define MLP input dimension for embeddings
# -----------------------------
input_dim_emb = X_emb_train_tensor.shape[1]

mlp_emb = SimpleMLPClassifier(input_dim_emb, hidden_dim, num_classes).to(device)
optimizer_emb = optim.Adam(mlp_emb.parameters(), lr=0.001, weight_decay=1e-5)

# -----------------------------
# 3. Training loop
# -----------------------------
n_epochs = 50
for epoch in range(n_epochs):
    mlp_emb.train()
    train_loss = 0

    for X_batch, y_batch in train_loader_emb:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer_emb.zero_grad()
        outputs = mlp_emb(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer_emb.step()

        train_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = train_loss / len(train_loader_emb)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

# -----------------------------
# 4. Evaluation
# -----------------------------
mlp_emb.eval()
y_mlp_emb_pred = []
y_mlp_emb_proba = []

with torch.no_grad():
    for X_batch, _ in test_loader_emb:
        X_batch = X_batch.to(device)
        outputs = mlp_emb(X_batch)

        # Predictions
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        y_mlp_emb_pred.extend(predicted.cpu().numpy())
        y_mlp_emb_proba.extend(probs.cpu().numpy())

y_mlp_emb_pred = np.array(y_mlp_emb_pred)
y_mlp_emb_proba = np.array(y_mlp_emb_proba)
y_mlp_emb_pred_orig = le.inverse_transform(y_mlp_emb_pred)

# -----------------------------
# 5. Compute metrics
# -----------------------------
mlp_emb_acc = accuracy_score(y_test_clean, y_mlp_emb_pred_orig)
mlp_emb_bal_acc = balanced_accuracy_score(y_test_clean, y_mlp_emb_pred_orig)
mlp_emb_precision_w = precision_score(y_test_clean, y_mlp_emb_pred_orig, average='weighted')
mlp_emb_recall_w = recall_score(y_test_clean, y_mlp_emb_pred_orig, average='weighted')
mlp_emb_f1_w = f1_score(y_test_clean, y_mlp_emb_pred_orig, average='weighted')

# AUC (macro OVR)
try:
    mlp_emb_auc = roc_auc_score(
        y_test_clean,
        y_mlp_emb_proba,
        average='macro',
        multi_class='ovr'
    )
except ValueError:
    mlp_emb_auc = float('nan')

print("\nMLP (Embeddings, regulon clusters) Performance:")
print(f"  Balanced Accuracy: {mlp_emb_bal_acc:.4f}")
print(f"  Weighted Precision: {mlp_emb_precision_w:.4f}")
print(f"  Weighted Recall: {mlp_emb_recall_w:.4f}")
print(f"  Weighted F1: {mlp_emb_f1_w:.4f}")
print(f"  AUC (macro OVR): {mlp_emb_auc:.4f}")


# In[ ]:


# Dictionary of models to save
models_to_save = {
    "rf_hvg_regulon": rf_regulon,
    "rf_regulon": rf_emb,
    "xgb_hvg_regulon": xgb_regulon,
    "xgb_regulon": xgb_emb,
    "mlp_hvg_regulon": mlp_regulon,
    "mlp_regulon": mlp_emb
}

# Save each model
for name, model in models_to_save.items():
    filepath = os.path.join("models", f"{name}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {name} -> {filepath}")

