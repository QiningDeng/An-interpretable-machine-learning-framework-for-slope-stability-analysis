import torch
import pandas as pd
import numpy as np
import shap
import time
import math
from pathlib import Path
import matplotlib.pyplot as plt
from torch import nn
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ===================== 1. Transformer Definition =====================
class PositionalEncoding(nn.Module):
    """Positional encoding module"""
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term) if d_model % 2 == 0 else torch.cos(position*div_term)[:, :pe[:,1::2].shape[1]]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerRegressor(nn.Module):
    """
    Treat each continuous feature as a token:
      token = value_proj(value) + feat_emb(feature_id) [+ pos_enc]
    Pass through TransformerEncoder, mean-pooling + MLP outputs 1-D continuous value
    """
    def __init__(self, n_features, d_model, nhead, num_layers, dim_ff, dropout, use_posenc=True):
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} must be divisible by nhead={nhead}"
        self.n_features = n_features
        self.value_proj = nn.Linear(1, d_model)
        self.feat_emb = nn.Embedding(n_features, d_model)
        self.use_posenc = use_posenc
        if use_posenc:
            self.pos_enc = PositionalEncoding(d_model, max_len=max(32, n_features+10))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Head MLP for final regression output
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        # Register feature indices for embedding
        self.register_buffer("feat_ids", torch.arange(n_features).long())

    def forward(self, x):
        B, F = x.shape
        v = self.value_proj(x.unsqueeze(-1))  # (B,F,1)->(B,F,d)
        ids = self.feat_ids.unsqueeze(0).expand(B, F)
        e = self.feat_emb(ids)  # (B,F,d)
        h = v + e
        if self.use_posenc: h = self.pos_enc(h)
        h = self.encoder(h)  # (B,F,d)
        h = h.mean(dim=1)  # (B,d)
        return self.head(h)  # (B,1)

# ===================== 2. Load Model =====================
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    # Rebuild model architecture
    model = TransformerRegressor(
        n_features=checkpoint['n_features'],
        d_model=checkpoint['struct_hparams']['d_model'],
        nhead=checkpoint['struct_hparams']['nhead'],
        num_layers=int(checkpoint['struct_hparams']['num_layers']),
        dim_ff=int(checkpoint['struct_hparams']['dim_ff']),
        dropout=float(checkpoint['struct_hparams']['dropout']),
        use_posenc=True
    )
    # Load model weights (strict=False for partial match)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()  # Set to evaluation mode
    return model

# ===================== 3. File Selection =====================
def select_file(title="Select Excel file"):
    root = Tk()
    root.withdraw()
    path = askopenfilename(title=title, filetypes=[("Excel or PyTorch Files","*.xlsx;*.xls;*.pt")])
    root.destroy()
    return path

# Use CPU device
device = torch.device("cpu")
model_path = select_file("Select Transformer model file (.pt)")
model = load_model(model_path, device)

excel_file = select_file("Select dataset file (.xlsx)")
df = pd.read_excel(excel_file, header=0)
feature_cols = df.columns[:5]  # Assume first 5 columns are features
X = df[feature_cols].to_numpy(dtype=float, copy=True)

# Compute predictions
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_pred = model(X_tensor).cpu().numpy()

# ===================== 4. Select Background and Explanation Samples =====================
bg_size = 100  # Background dataset size
exp_size = 1000  # Explanation dataset size
X_bg = X[:bg_size]
X_exp = X[:exp_size]

# ===================== 5. Create SHAP Explainer =====================
def get_explainer(model, X_bg):
    # Wrap model predict function for SHAP
    def model_predict(X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_tensor).cpu().detach().numpy()
    explainer = shap.KernelExplainer(model_predict, X_bg)
    return explainer

explainer = get_explainer(model, X_bg)
shap_values_raw = explainer.shap_values(X_exp)
shap_matrix = np.asarray(shap_values_raw).reshape(-1, len(feature_cols))

# ===================== 6. Plot and Save Figures =====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

timestamp = time.strftime("%Y%m%d-%H%M%S")
out_dir = Path(f"shap_outputs_{timestamp}")
out_dir.mkdir(parents=True, exist_ok=True)

# Global contribution bar plot
plt.figure()
shap.summary_plot(shap_matrix, X_exp, feature_names=feature_cols, plot_type="bar", max_display=len(feature_cols), show=False)
ax = plt.gca()
for i, patch in enumerate(ax.patches):
    # Annotate feature importance value
    ax.text(patch.get_width()+0.01, patch.get_y()+patch.get_height()/2, f'{patch.get_width():.4f}', ha='left', va='center', fontsize=8, family='Times New Roman')
plt.tight_layout()
plt.savefig(out_dir / "global_contribution_bar.png", bbox_inches="tight", dpi=1000)
plt.close()

# Decision plot
plt.figure()
shap.decision_plot(explainer.expected_value, shap_matrix[:1000], X_exp[:1000], feature_names=feature_cols.tolist(), show=False)
plt.tight_layout()
plt.savefig(out_dir / "decision_plot.png", bbox_inches="tight", dpi=1000)
plt.close()

# Summary beeswarm plot
plt.figure()
shap.summary_plot(shap_matrix, X_exp, feature_names=feature_cols, max_display=len(feature_cols), show=False)
plt.tight_layout()
plt.savefig(out_dir / "summary_beeswarm.png", bbox_inches="tight", dpi=1000)
plt.close()

# Single sample waterfall plot
plt.figure()
shap.waterfall_plot(shap.Explanation(values=shap_matrix[13], base_values=explainer.expected_value, data=X_exp[0], feature_names=feature_cols), show=False)
plt.tight_layout()
plt.savefig(out_dir / "sample_0_waterfall.png", bbox_inches="tight", dpi=1000)
plt.close()

print(f"\nSHAP analysis complete. Results saved at: {out_dir.resolve()}")
