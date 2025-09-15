import json
import time
import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

# ==========================
# 0. Global configuration (reproducible)
# ==========================
SEED = 42
rng = np.random.default_rng(SEED)  # Use same random seed
plt.rcParams.update({'figure.dpi': 1000})

# ==========================
# 1. Popup window to select model files
# ==========================
def select_model(title):
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title=title, filetypes=[("Pickle files", "*.pkl")])
    root.destroy()
    return joblib.load(file_path)

# Load base learner models
models = {}
models['ENR']  = select_model("Select ENR model file (.pkl)")   # Linear model (ElasticNet/Linear regression family)
models['SVR']  = select_model("Select SVR model file (.pkl)")   # Black-box
models['DTR']  = select_model("Select DTR model file (.pkl)")   # Tree-based model
models['KNNR'] = select_model("Select KNNR model file (.pkl)")  # Black-box
models['MLPR'] = select_model("Select MLPR model file (.pkl)")  # Black-box (MLP regressor)

# ==========================
# 2. Select and read multiple Excel feature files (one per model)
# ==========================
excel_files = {}
for model_name in models.keys():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title=f"Select Excel input features for {model_name}", filetypes=[("Excel files", "*.xlsx;*.xls")])
    root.destroy()
    excel_files[model_name] = file_path

# ==========================
# 3. Read each model's features and predictions
# ==========================
data = {}
for model_name, file_path in excel_files.items():
    df = pd.read_excel(file_path, header=0)
    feature_cols = df.columns[:5]      # Assume first 5 columns are input features
    X = df[feature_cols].to_numpy(dtype=float, copy=True)
    y_pred = df['Fs'].to_numpy(dtype=float, copy=True)  # Use 'Fs' column as predictions
    data[model_name] = {'X': X, 'y_pred': y_pred, 'feature_cols': feature_cols}

    print(f"{model_name} total samples: {X.shape[0]}")

# ==========================
# 4. Randomly select background set and explanation samples
# ==========================
bg_size = 100  # Background set size
exp_size = 300  # Number of explanation samples

for model_name in models.keys():
    X = data[model_name]['X']
    n_total = X.shape[0]

    # Randomly select background and explanation indices
    bg_idx = rng.choice(np.arange(n_total), size=bg_size, replace=False)
    exp_idx = rng.choice(np.arange(n_total), size=exp_size, replace=False)

    # Extract background and explanation samples
    data[model_name]['X_bg'] = X[bg_idx]
    data[model_name]['X_exp'] = X[exp_idx]
    data[model_name]['y_pred_bg'] = y_pred[bg_idx]  # Predictions for background
    data[model_name]['y_pred_exp'] = y_pred[exp_idx]  # Predictions for explanation samples

    print(f"{model_name} background size: {bg_size} | explanation sample size: {exp_size}")

# ==========================
# 5. Output directory (timestamped)
# ==========================
timestamp = time.strftime("%Y%m%d-%H%M%S")
out_dir = Path(f"shap_outputs_{timestamp}")
out_dir.mkdir(parents=True, exist_ok=True)

# ==========================
# 6. Select explainer & compute SHAP
# ==========================
def get_explainer(model_name, model, X_bg):
    if model_name == "DTR":
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent", 
            model_output="raw",
        )
        return explainer, "TreeExplainer"
    elif model_name == "ENR":
        masker = shap.maskers.Independent(X_bg)
        explainer = shap.LinearExplainer(model, masker)
        return explainer, "LinearExplainer"
    else:
        explainer = shap.KernelExplainer(model.predict, X_bg)
        return explainer, "KernelExplainer"

# ==========================
# 7. Perform SHAP analysis for each model
# ==========================
for name, model in models.items():
    print(f"\n=== Analyzing {name} ===")
    model_dir = out_dir / f"{name}"
    model_dir.mkdir(exist_ok=True)

    # Retrieve features and predictions
    X = data[name]['X']
    y_pred = data[name]['y_pred']
    feature_cols = data[name]['feature_cols']
    X_bg = data[name]['X_bg']
    X_exp = data[name]['X_exp']
    y_pred_bg = data[name]['y_pred_bg']
    y_pred_exp = data[name]['y_pred_exp']

    # Create explainer
    explainer, explainer_type = get_explainer(name, model, X_bg)
    with open(model_dir / "explainer_type.txt", "w", encoding="utf-8") as f:
        f.write(explainer_type)

    # Compute SHAP values
    if explainer_type == "KernelExplainer":
        shap_values_raw = explainer.shap_values(X_exp)
    elif explainer_type == "TreeExplainer":
        shap_values_raw = explainer.shap_values(X_exp, check_additivity=False)
    else:
        shap_values_raw = explainer.shap_values(X_exp)

    shap_matrix = np.asarray(shap_values_raw).reshape(-1, len(feature_cols))  # (n_exp, n_features)

    # Save SHAP values together with predictions
    shap_matrix_df = pd.DataFrame(shap_matrix, columns=feature_cols)
    shap_matrix_df['Prediction'] = y_pred_exp
    shap_matrix_df.to_csv(model_dir / "shap_values_with_predictions.csv", index=False, encoding="utf-8-sig")

    # ========== Plotting and saving ==========
    # 1) Global contribution bar chart
    plt.figure()
    shap.summary_plot(
        shap_matrix, X_exp,
        feature_names=feature_cols,
        plot_type="bar",
        max_display=feature_cols.size,
        show=False
    )
    plt.tight_layout()
    plt.savefig(model_dir / "global_contribution_bar.png", bbox_inches="tight")
    plt.close()

    # 2) Decision plot (subsample 300)
    plt.figure()
    shap.decision_plot(
        explainer.expected_value, shap_matrix[:300], X_exp[:300],
        feature_names=feature_cols.tolist(),
        show=False
    )
    plt.tight_layout()
    plt.savefig(model_dir / "decision_plot.png", bbox_inches="tight")
    plt.close()

    # 3) Summary beeswarm plot
    plt.figure()
    shap.summary_plot(
        shap_matrix, X_exp,
        feature_names=feature_cols,
        max_display=feature_cols.size,
        show=False
    )
    plt.tight_layout()
    plt.savefig(model_dir / "summary_beeswarm.png", bbox_inches="tight")
    plt.close()

    # 4) Single sample waterfall plot (example: first sample)
    plt.figure()
    shap.waterfall_plot(shap.Explanation(values=shap_matrix[0], base_values=explainer.expected_value, data=X_exp[0], feature_names=feature_cols), show=False)
    plt.tight_layout()
    plt.savefig(model_dir / "sample_0_waterfall.png", bbox_inches="tight")
    plt.close()

print(f"\nAll done. Results saved to: {out_dir.resolve()}")
