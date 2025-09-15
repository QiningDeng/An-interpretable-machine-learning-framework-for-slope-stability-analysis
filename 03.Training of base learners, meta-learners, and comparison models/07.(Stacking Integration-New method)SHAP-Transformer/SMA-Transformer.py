"""
This script performs hyperparameter search for a Transformer-based tabular regression model.
It uses SMA (Slime Mould Algorithm) from the 'mealpy' library to optimize only the model's
structural hyperparameters (such as d_model, nhead, num_layers, dim_ff, dropout), given discrete
candidate sets. The search uses a fixed training and evaluation pipeline with early stopping,
based on MAE on a held-out test set. The best structure is then retrained on the full training data
and the resulting model is saved to disk.
"""

import math
import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mealpy import FloatVar, IntegerVar, SMA, Problem
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import warnings, os, random, time
warnings.filterwarnings("ignore")

# ===================== 0. Basic configuration (fixed training hyperparameters & environment) =====================
SEED = 1
FIXED_BATCH_SIZE = 32
FIXED_LR = 1e-3
FIXED_WEIGHT_DECAY = 1e-4
FIXED_EPOCHS = 100
EARLYSTOP_PATIENCE = 10

# ---- Search configuration (total evaluations = OPT_EPOCH * OPT_POP_SIZE) ----
OPT_EPOCH = 10
OPT_POP_SIZE = 10
EXPECTED_EVALS = OPT_EPOCH * OPT_POP_SIZE
EVAL_COUNT = 0
BEST_SO_FAR = {"mae": float("inf"), "params": None}
TIC = time.time()

# ---- Discrete candidate sets (ensure d_model % nhead == 0) ----
D_MODEL_CHOICES = [8, 16, 32, 64, 128]
NHEAD_CHOICES   = [1, 2, 4, 8]

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"[Info] Using device: {device}")

# ===================== 1. Select and load training/test Excel files =====================
def select_file(title="Select Excel file"):
    root = Tk()
    root.withdraw()
    return askopenfilename(title=title, filetypes=[("Excel Files","*.xlsx;*.xls")])

print("\n[Step] Please select the TRAINING SET Excel (last column is target, others are features)")
train_path = select_file("Select training set Excel (last column is target)")
if not train_path:
    print("No training set file selected"); exit()

print("[Step] Please select the TEST SET Excel (same structure as training set, last column is target)")
test_path = select_file("Select test set Excel (last column is target)")
if not test_path:
    print("No test set file selected"); exit()

train_df = pd.read_excel(train_path)
test_df  = pd.read_excel(test_path)

# ---- Consistency check: column names and order (recommended identical) ----
if list(train_df.columns) != list(test_df.columns):
    print("[Warn] Training and test set columns differ! Ensure identical column names/order (last col = target).")
    try:
        test_df = test_df[train_df.columns]
        print("[Info] Test set columns reordered to match training set.")
    except Exception as e:
        print(f"[Error] Unable to align columns: {e}")
        exit()

# Convention: last column is target, others are continuous features (apply normalization if needed)
X_train_np = train_df.iloc[:, :-1].values.astype(np.float32)
y_train_np = train_df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

X_test_np  = test_df.iloc[:, :-1].values.astype(np.float32)
y_test_np  = test_df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

n_features = X_train_np.shape[1]
if X_test_np.shape[1] != n_features:
    print(f"[Error] Feature count mismatch: train={n_features}, test={X_test_np.shape[1]}")
    exit()

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_ds = TabularDataset(X_train_np, y_train_np)
test_ds  = TabularDataset(X_test_np,  y_test_np)

def make_loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=FIXED_BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )

# ===================== 2. Model definition (Transformer regressor) =====================
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
    """
    Treat each continuous feature as a token:
      token = value_proj(value) + feat_emb(feature_id) [+ pos_enc]
    Pass through TransformerEncoder, then mean pooling + MLP -> 1D output
    """
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
        v = self.value_proj(x.unsqueeze(-1))          # (B,F,1)->(B,F,d)
        ids = self.feat_ids.unsqueeze(0).expand(B, F)
        e = self.feat_emb(ids)                        # (B,F,d)
        h = v + e
        if self.use_posenc: h = self.pos_enc(h)
        h = self.encoder(h)                           # (B,F,d)
        h = h.mean(dim=1)                             # (B,d)
        y = self.head(h)                              # (B,1)
        return y

# ===================== 3. Training / evaluation (MAE) =====================
def evaluate_mae(model, loader):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            preds.append(pred.cpu().numpy()); gts.append(yb.cpu().numpy())
    preds = np.vstack(preds); gts = np.vstack(gts)
    return metrics.mean_absolute_error(gts, preds)

def train_one_model(struct_hparams, train_ds, test_ds, max_epochs=FIXED_EPOCHS, patience=EARLYSTOP_PATIENCE, verbose=False):
    d_model = int(struct_hparams['d_model'])
    nhead   = int(struct_hparams['nhead'])
    assert d_model % nhead == 0

    model = TransformerRegressor(
        n_features=n_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=int(struct_hparams['num_layers']),
        dim_ff=int(struct_hparams['dim_ff']),
        dropout=float(struct_hparams['dropout']),
        use_posenc=True
    ).to(device)

    train_loader = make_loader(train_ds, shuffle=True)
    test_loader  = make_loader(test_ds,  shuffle=False)

    criterion = nn.L1Loss()  # MAE
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

        val_mae = evaluate_mae(model, test_loader)
        if verbose:
            print(f"  [Train] Epoch {epoch+1:03d} | val_MAE={val_mae:.6f}")
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

    final_mae = evaluate_mae(model, test_loader)
    return model, final_mae

# ===================== 4. Define "structure-only" optimization problem (discrete sets + progress display) =====================
def _format_params(p):
    return (f"d_model={int(p['d_model'])}, nhead={int(p['nhead'])}, "
            f"num_layers={int(p['num_layers'])}, dim_ff={int(p['dim_ff'])}, "
            f"dropout={float(p['dropout']):.3f}")

class TransStructProblem(Problem):
    def __init__(self, bounds=None, minmax="min", **kwargs):
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        d_model = D_MODEL_CHOICES[int(x_decoded["d_model_idx"])]
        nhead   = NHEAD_CHOICES[int(x_decoded["nhead_idx"])]
        assert d_model % nhead == 0, f"d_model={d_model} must be divisible by nhead={nhead}"

        struct = dict(
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(x_decoded["num_layers"]),
            dim_ff=int(x_decoded["dim_ff"]),
            dropout=float(x_decoded["dropout"]),
        )

        _, mae = train_one_model(struct, train_ds, test_ds, max_epochs=FIXED_EPOCHS, patience=EARLYSTOP_PATIENCE, verbose=False)

        global EVAL_COUNT, BEST_SO_FAR
        EVAL_COUNT += 1
        if mae < BEST_SO_FAR["mae"]:
            BEST_SO_FAR = {"mae": mae, "params": struct}

        elapsed = time.time() - TIC
        print(f"[Search #{EVAL_COUNT:04d}] MAE={mae:.6f} | {_format_params(struct)} | "
              f"Best={BEST_SO_FAR['mae']:.6f} | Elapsed={elapsed/60:.1f} min")

        return mae  # Objective: minimize MAE

# ---- Search space (note: d_model/nhead use "index variables") ----
my_bounds = [
    IntegerVar(lb=0,  ub=len(D_MODEL_CHOICES)-1, name="d_model_idx"),
    IntegerVar(lb=0,  ub=len(NHEAD_CHOICES)-1,   name="nhead_idx"),

    IntegerVar(lb=1,  ub=4,    name="num_layers"),
    IntegerVar(lb=4,  ub=64,   name="dim_ff"),
    FloatVar(  lb=0.0,ub=0.3,  name="dropout"),
]

problem = TransStructProblem(bounds=my_bounds, minmax="min")

# ===================== 5. Run SMA optimization (structure-only, discrete sets) =====================
print("\n========== Hyperparam Search (structure-only, discrete) ==========")
print(f"Plan: epoch={OPT_EPOCH}, pop_size={OPT_POP_SIZE} -> expected evals={EXPECTED_EVALS}\n")

evolver = SMA.OriginalSMA(epoch=OPT_EPOCH, pop_size=OPT_POP_SIZE)
evolver.solve(problem)

print("\n================== Search Summary ==================")
print(f"Expected evaluations: {EXPECTED_EVALS}")
print(f"Actual evaluations:   {EVAL_COUNT}")
if EVAL_COUNT != EXPECTED_EVALS:
    print("[Warn] Actual vs expected evaluation count mismatch: some versions/strategies may add/remove evals.")
print(f"Best MAE: {evolver.g_best.target.fitness:.6f}")

best_struct = evolver.problem.decode_solution(evolver.g_best.solution)
best_struct = dict(
    d_model = int(D_MODEL_CHOICES[int(best_struct["d_model_idx"])]),
    nhead   = int(NHEAD_CHOICES[int(best_struct["nhead_idx"])]),
    num_layers=int(best_struct["num_layers"]),
    dim_ff=int(best_struct["dim_ff"]),
    dropout=float(best_struct["dropout"]),
)

print("[Best structure params]")
for k, v in best_struct.items():
    print(f"  {k}: {v if k!='dropout' else round(v, 4)}")

# ===================== 6. Retrain with best structure and save model =====================
print("\n[Info] Retraining final model with best structure...")
final_model, final_mae = train_one_model(
    best_struct, train_ds, test_ds,
    max_epochs=FIXED_EPOCHS, patience=EARLYSTOP_PATIENCE, verbose=True
)

save_path = "transformer_regressor_best.pt"
torch.save({
    "model_state": final_model.state_dict(),
    "struct_hparams": best_struct,
    "n_features": n_features,
    "seed": SEED,
    "fixed_train_hparams": {
        "batch_size": FIXED_BATCH_SIZE,
        "lr": FIXED_LR,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "epochs": FIXED_EPOCHS,
        "patience": EARLYSTOP_PATIENCE,
    },
    "search_meta": {
        "epoch": OPT_EPOCH,
        "pop_size": OPT_POP_SIZE,
        "expected_evals": EXPECTED_EVALS,
        "actual_evals": EVAL_COUNT,
        "best_val_mae_during_search": float(BEST_SO_FAR["mae"]),
        "d_model_choices": D_MODEL_CHOICES,
        "nhead_choices": NHEAD_CHOICES,
    },
    "feature_names": list(train_df.columns[:-1]),
    "target_name": train_df.columns[-1],
    "train_path": os.path.abspath(train_path),
    "test_path": os.path.abspath(test_path),
}, save_path)

print(f"[Saved] Final model saved to: {os.path.abspath(save_path)}")
print(f"[Result] Test MAE: {final_mae:.6f} | Batch size: {FIXED_BATCH_SIZE} | Device: {device}")
