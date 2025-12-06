# %%
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from sklearn.preprocessing import normalize

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

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
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

# %%
y_train = adata_train.obs['tf_path_cluster'].astype(int).to_numpy()
y_test = adata_test.obs['tf_path_cluster'].astype(int).to_numpy()

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# %%
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

# %%
input_dim  = X_train.shape[1]
n_classes  = len(np.unique(y_train))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout=0.2):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim, hidden_dims=(256, 128), n_classes=n_classes, dropout=0.3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


# %%
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += y_batch.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += y_batch.size(0)

    return running_loss / total, correct / total


# %%
n_epochs = 30

for epoch in tqdm(range(1, n_epochs + 1), desc="Training epochs"):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc     = eval_epoch(model, test_loader,  criterion, device)

    tqdm.write(
        f"Epoch {epoch:02d} | "
        f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
        f"test loss {val_loss:.4f}, acc {val_acc:.3f}"
    )


# %%
from sklearn.metrics import classification_report

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_true = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y_batch.numpy())
    return np.concatenate(all_true), np.concatenate(all_preds)

y_true, y_pred = get_predictions(model, test_loader, device)
print(classification_report(y_true, y_pred, digits=3))


# %%
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

@torch.no_grad()
def get_proba(model, loader, device):
    model.eval()
    all_proba = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        proba = logits.softmax(dim=1).cpu().numpy()
        all_proba.append(proba)
    return np.concatenate(all_proba)

y_proba = get_proba(model, test_loader, device)

classes = np.unique(y_train)
y_test_bin = label_binarize(y_true, classes=classes)

roc_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
print("MLP ROC AUC (macro, OVR):", roc_auc)


# %%



