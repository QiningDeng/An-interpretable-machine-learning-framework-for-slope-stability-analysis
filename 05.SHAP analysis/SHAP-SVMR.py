import joblib
import pandas as pd
import numpy as np
import shap
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ===================== 1. Load trained SVMR model =====================
def load_svmr_model(model_path):
    """Directly load the pre-trained SVMR model (.pkl)."""
    return joblib.load(model_path)

# ===================== 2. Select files =====================
def select_file(title="Select File"):
    root = Tk()
    root.withdraw()
    path = askopenfilename(title=title, filetypes=[("Excel or Pickle Files", "*.xlsx;*.xls;*.pkl")])
    root.destroy()
    return path

# Select and load SVMR model
model_path = select_file("Select SVMR model file (.pkl)")
svmr_model = load_svmr_model(model_path)

# Select and load dataset
excel_file = select_file("Select dataset file (.xlsx)")
df = pd.read_excel(excel_file, header=0)

# Assume the first 5 columns are features
feature_cols = df.columns[:5]
X = df[feature_cols].to_numpy(dtype=float, copy=True)

# Compute predictions
y_pred = svmr_model.predict(X)

# ===================== 3. Select background set and explanation samples =====================
bg_size = 100    # Background set size
exp_size = 1000  # Explanation sample size

X_bg = X[:bg_size]
y_pred_bg = y_pred[:bg_size]
X_exp = X[:exp_size]
y_pred_exp = y_pred[:exp_size]

# ===================== 4. Choose explainer & compute SHAP values =====================
def get_explainer(model, X_bg):
    X_bg = np.array(X_bg)
    def model_predict(X):
        return model.predict(X)
    explainer = shap.KernelExplainer(model_predict, X_bg)
    return explainer

explainer = get_explainer(svmr_model, X_bg)
shap_values_raw = explainer.shap_values(X_exp)
shap_matrix = np.asarray(shap_values_raw).reshape(-1, len(feature_cols))

# ===================== 5. Plot and save figures =====================
# Configure font and DPI
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

# Create output directory with timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
out_dir = Path(f"shap_outputs_{timestamp}")
out_dir.mkdir(parents=True, exist_ok=True)

# 5.1 Global contribution bar chart
plt.figure()
shap.summary_plot(
    shap_matrix, X_exp,
    feature_names=feature_cols,
    plot_type="bar",
    max_display=len(feature_cols),
    show=False
)

# Annotate bars with numeric values
ax = plt.gca()
for i, patch in enumerate(ax.patches):
    value = patch.get_width()
    ax.text(value + 0.01, patch.get_y() + patch.get_height() / 2, f'{value:.4f}',
            ha='left', va='center', fontsize=8, family='Times New Roman')

plt.tight_layout()
plt.savefig(out_dir / "global_contribution_bar.png", bbox_inches="tight", dpi=1000)
plt.close()

# 5.2 Decision plot (use first 1000 samples)
plt.figure()
shap.decision_plot(
    explainer.expected_value, shap_matrix[:1000], X_exp[:1000],
    feature_names=feature_cols.tolist(),
    show=False
)
plt.tight_layout()
plt.savefig(out_dir / "decision_plot.png", bbox_inches="tight", dpi=1000)
plt.close()

# 5.3 Summary beeswarm plot
plt.figure()
shap.summary_plot(
    shap_matrix, X_exp,
    feature_names=feature_cols,
    max_display=len(feature_cols),
    show=False
)
plt.tight_layout()
plt.savefig(out_dir / "summary_beeswarm.png", bbox_inches="tight", dpi=1000)
plt.close()

# 5.4 Single-sample waterfall plot (example: sample index 13)
plt.figure()
shap.waterfall_plot(
    shap.Explanation(values=shap_matrix[13],
                     base_values=explainer.expected_value,
                     data=X_exp[0],
                     feature_names=feature_cols),
    show=False
)
plt.tight_layout()
plt.savefig(out_dir / "sample_0_waterfall.png", bbox_inches="tight", dpi=1000)
plt.close()

print(f"\nSHAP analysis completed. Results saved to: {out_dir.resolve()}")
