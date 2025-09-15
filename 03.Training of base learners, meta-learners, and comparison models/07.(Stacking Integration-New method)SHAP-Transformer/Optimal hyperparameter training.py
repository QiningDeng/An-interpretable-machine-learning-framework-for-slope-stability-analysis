"""
This script implements a Transformer-based regressor for tabular data regression tasks.
It loads training and test datasets from Excel files, uses 5-fold cross-validation
to evaluate performance, generates out-of-fold (OOF) predictions, retrains on the full
training set, and evaluates on the test set. Metrics include MAE, RMSE, and R².
The final model and evaluation results are saved to disk.
"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tkinter import Tk
from tkinter.filedialog import askopenfilename
import warnings, os, random, time

warnings.filterwarnings("ignore")

# ===================== 0. Basic Configuration (fixed training hyperparameters & environment) =====================
SEED = 1
FIXED_BATCH_SIZE   = 16
FIXED_LR           = 1e-3
FIXED_WEIGHT_DECAY = 1e-4
FIXED_EPOCHS       = 50
EARLYSTOP_PATIENCE = 10

def set_seed(seed=SEED):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"[Info] Using device: {device}")

# ===================== 1. Select and Load Training & Test Set =====================
def select_file(title="Select Excel File"):
    root = Tk()
    root.withdraw()
    return askopenfilename(title=title, filetypes=[("Excel Files","*.xlsx;*.xls")])

print("\n[Info] Please select training set Excel file...")
train_path = select_file("Select training set Excel file")
if not train_path:
    print("No training set file selected"); exit()

print("\n[Info] Please select test set Excel file...")
test_path = select_file("Select test set Excel file")
if not test_path:
    print("No test set file selected"); exit()

train_raw = pd.read_excel(train_path)
test_raw  = pd.read_excel(test_path)

# Convention: last column is target, remaining are features
X_train_np = train_raw.iloc[:, :-1].values.astype(np.float32)
y_train_np = train_raw.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)
idx_train_glb = np.arange(len(y_train_np))

X_test_np = test_raw.iloc[:, :-1].values.astype(np.float32)
y_test_np = test_raw.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)
idx_test_glb = np.arange(len(y_test_np))

n_features = X_train_np.shape[1]
print(f"[Info] Train size: {len(y_train_np)}, Test size: {len(y_test_np)}, Features: {n_features}")

# ===================== 2. Dataset and Dataloader =====================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def make_loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=FIXED_BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )

# ===================== 3. Model Definition (Transformer Regressor) =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position*div_term)[:, :pe[:,1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerRegressor(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_layers, dim_ff, dropout, use_posenc=True):
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} must be divisible by nhead={nhead}"
        self.n_features = n_features
        self.value_proj = nn.Linear(1, d_model)
        self.feat_emb   = nn.Embedding(n_features, d_model)
        self.use_posenc = use_posenc
        if use_posenc:
            self.pos_enc = PositionalEncoding(d_model, max_len=max(32, n_features+10))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.register_buffer("feat_ids", torch.arange(n_features).long())
    def forward(self, x):
        B, F = x.shape
        v = self.value_proj(x.unsqueeze(-1))
        ids = self.feat_ids.unsqueeze(0).expand(B, F)
        e = self.feat_emb(ids)
        h = v + e
        if self.use_posenc: h = self.pos_enc(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        y = self.head(h)
        return y

# ===================== 4. Evaluation Functions =====================
def metric_report(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae  = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    r2   = metrics.r2_score(y_true, y_pred)
    return mae, rmse, r2

def print_fold_metrics(fold, phase, mae, rmse, r2, n):
    print(f"[Fold {fold:>2}] {phase:<10} | n={n:<5} | MAE={mae:.6f} | RMSE={rmse:.6f} | R²={r2:.6f}")

def predict_numpy(model, X_np):
    dummy_y = np.zeros((len(X_np), 1), dtype=np.float32)
    ds = TabularDataset(X_np.astype(np.float32), dummy_y)
    dl = DataLoader(ds, batch_size=FIXED_BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)
            yb = model(xb)
            preds.append(yb.detach().cpu().numpy())
    return np.vstack(preds)

# ===================== 5. Single Model Training Function =====================
BEST_D_MODEL   = 64
BEST_NHEAD     = 4
BEST_NUM_LAYER = 3
BEST_DIM_FF    = 25
BEST_DROPOUT   = 0.0802

def train_one_model_with_metrics(
    X_tr_np, y_tr_np, X_val_np, y_val_np,
    d_model=BEST_D_MODEL, nhead=BEST_NHEAD, num_layers=BEST_NUM_LAYER,
    dim_ff=BEST_DIM_FF, dropout=BEST_DROPOUT,
    max_epochs=FIXED_EPOCHS, patience=EARLYSTOP_PATIENCE, verbose=False
):
    """Train one Transformer model with given hyperparameters and return it."""
    train_ds = TabularDataset(X_tr_np.astype(np.float32), y_tr_np.astype(np.float32).reshape(-1,1))
    val_ds   = TabularDataset(X_val_np.astype(np.float32), y_val_np.astype(np.float32).reshape(-1,1))
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,  shuffle=False)

    model = TransformerRegressor(
        n_features=n_features,
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dim_ff=int(dim_ff),
        dropout=float(dropout),
        use_posenc=True
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FIXED_LR, weight_decay=FIXED_WEIGHT_DECAY)

    best_val = float('inf'); best_state = None; wait = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # Validate
        model.eval()
        preds_val, gts_val = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pv = model(xb)
                preds_val.append(pv.detach().cpu().numpy())
                gts_val.append(yb.detach().cpu().numpy())
        preds_val = np.vstack(preds_val); gts_val = np.vstack(gts_val)
        val_mae = metrics.mean_absolute_error(gts_val, preds_val)
        if verbose:
            print(f"  [CV-Train] Epoch {epoch+1:03d} | val_MAE={val_mae:.6f}")
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
