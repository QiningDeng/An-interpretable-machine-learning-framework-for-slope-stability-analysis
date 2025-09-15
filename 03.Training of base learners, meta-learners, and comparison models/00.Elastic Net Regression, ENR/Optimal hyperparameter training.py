"""
This script trains and evaluates an ElasticNet regression model using 
predefined hyperparameters. It performs 5-fold cross-validation on the 
training set (reporting per-fold metrics and generating OOF predictions), 
fits a final model on the full training data, evaluates on the test set, 
and exports metrics, predictions, and summary reports as CSV files.
"""

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import joblib
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ========== 0. Utilities ==========
def metric_report(y_true, y_pred):
    mae  = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    r2   = metrics.r2_score(y_true, y_pred)
    return mae, rmse, r2

def print_fold_metrics(fold, phase, mae, rmse, r2, n):
    print(f"[Fold {fold:>2}] {phase:<10} | n={n:<5} | MAE={mae:.6f} | RMSE={rmse:.6f} | R²={r2:.6f}")

# ========== 1. File selection ==========
def select_file():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title="Select Excel file", filetypes=[("Excel Files", "*.xlsx;*.xls")])
    return file_path

file_path = select_file()
if not file_path:
    print("No file selected")
    exit()

# ========== 2. Load data and split ==========
data_raw = pd.read_excel(file_path)
X = data_raw.iloc[:, 0:5].values   # Feature columns (adjust as needed)
y = data_raw.iloc[:, 5].values     # Target column (adjust as needed)
idx_all = np.arange(len(y))        # Keep original sample indices

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, idx_all, test_size=0.3, random_state=1
)

# ========== 3. Predefined hyperparameters ==========
best_alpha = 0.0483
best_l1_ratio = 0.0001
best_max_iter = 4

# ========== 4. 5-fold CV on training set ==========
kf = KFold(n_splits=5, shuffle=True, random_state=1)

rows = []  
val_mae_list, val_rmse_list, val_r2_list = [], [], []
oof_pred = np.empty_like(y_train, dtype=float)
oof_pred[:] = np.nan
oof_rows = []

print("\n=== 5-fold Cross-validation on training set (per-fold details + OOF generation) ===")
for fold_id, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    model_cv = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                          max_iter=best_max_iter, random_state=1)
    model_cv.fit(X_tr, y_tr)

    # Training subset metrics
    y_pred_tr = model_cv.predict(X_tr)
    tr_mae, tr_rmse, tr_r2 = metric_report(y_tr, y_pred_tr)
    print_fold_metrics(fold_id, "Train", tr_mae, tr_rmse, tr_r2, len(y_tr))
    rows.append({"Fold": fold_id, "Phase": "Train", "n": len(y_tr),
                 "MAE": tr_mae, "RMSE": tr_rmse, "R2": tr_r2})

    # Validation subset metrics
    y_pred_val = model_cv.predict(X_val)
    val_mae, val_rmse, val_r2 = metric_report(y_val, y_pred_val)
    print_fold_metrics(fold_id, "Val", val_mae, val_rmse, val_r2, len(y_val))
    rows.append({"Fold": fold_id, "Phase": "Val", "n": len(y_val),
                 "MAE": val_mae, "RMSE": val_rmse, "R2": val_r2})

    # Fill OOF predictions
    oof_pred[val_idx] = y_pred_val

    for i_local, i_global in enumerate(idx_train[val_idx]):
        oof_rows.append({
            "global_index": int(i_global),
            "fold": int(fold_id),
            "y_true": float(y_val[i_local]),
            "y_pred_oof": float(y_pred_val[i_local])
        })

    val_mae_list.append(val_mae)
    val_rmse_list.append(val_rmse)
    val_r2_list.append(val_r2)

# Aggregate validation statistics
val_mae_mean,  val_mae_std  = np.mean(val_mae_list),  np.std(val_mae_list)
val_rmse_mean, val_rmse_std = np.mean(val_rmse_list), np.std(val_rmse_list)
val_r2_mean,   val_r2_std   = np.mean(val_r2_list),   np.std(val_r2_list)

print("\n--- Validation metrics (per-fold aggregated) ---")
print(f"MAE  : {val_mae_mean:.6f} ± {val_mae_std:.6f}")
print(f"RMSE : {val_rmse_mean:.6f} ± {val_rmse_std:.6f}")
print(f"R²   : {val_r2_mean:.6f} ± {val_r2_std:.6f}")

cv_metrics_df = pd.DataFrame(rows)
cv_metrics_df.to_csv("cv_metrics_per_fold.csv", index=False, encoding="utf-8-sig")
print("Per-fold CV metrics saved as 'cv_metrics_per_fold.csv'")

# ========== 5. Export OOF predictions ==========
oof_df = pd.DataFrame({
    "global_index": idx_train,
    "y_true": y_train,
    "y_pred_oof_enr": oof_pred
})
feat_names = [f"feat_{i}" for i in range(X_train.shape[1])]
oof_features = pd.DataFrame(X_train, columns=feat_names)
oof_df = pd.concat([oof_df, oof_features], axis=1)

oof_df.sort_values("global_index", inplace=True)
oof_df.to_csv("oof_predictions_train_enr.csv", index=False, encoding="utf-8-sig")
print("OOF predictions for training saved as 'oof_predictions_train_enr.csv'")

# ========== 6. Final model training + test evaluation ==========
final_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                         max_iter=best_max_iter, random_state=1)
final_model.fit(X_train, y_train)
joblib.dump(final_model, 'final_enr_model_refit_on_full_train.pkl')
print("\nFinal model saved as 'final_enr_model_refit_on_full_train.pkl'")

# Training performance
y_pred_train_full = final_model.predict(X_train)
train_mae, train_rmse, train_r2 = metric_report(y_train, y_pred_train_full)
print("\n=== Training set (full) ===")
print(f"MAE={train_mae:.6f} | RMSE={train_rmse:.6f} | R²={train_r2:.6f}")

# Test set evaluation
y_pred_test_final = final_model.predict(X_test)
test_mae, test_rmse, test_r2 = metric_report(y_test, y_pred_test_final)
print("\n=== Test set (final evaluation) ===")
print(f"MAE={test_mae:.6f} | RMSE={test_rmse:.6f} | R²={test_r2:.6f}")

# Export test predictions
test_df = pd.DataFrame({
    "global_index": idx_test,
    "y_true": y_test,
    "y_pred_test_enr": y_pred_test_final
})
feat_names = [f"feat_{i}" for i in range(X_test.shape[1])]
test_features = pd.DataFrame(X_test, columns=feat_names)
test_df = pd.concat([test_df, test_features], axis=1)

test_df.sort_values("global_index", inplace=True)
test_df.to_csv("predictions_test_enr.csv", index=False, encoding="utf-8-sig")
print("Test set predictions saved as 'predictions_test_enr.csv'")

# Summary report
summary = pd.DataFrame([
    ["Validation CV (mean±std)", val_mae_mean, val_rmse_mean, val_r2_mean,
     f"MAE±{val_mae_std:.6f}; RMSE±{val_rmse_std:.6f}; R2±{val_r2_std:.6f}"],
    ["Train (full, hold-in)",   train_mae,     train_rmse,     train_r2,     ""],
    ["Test (hold-out once)",    test_mae,      test_rmse,      test_r2,      ""],
], columns=["Phase", "MAE", "RMSE", "R2", "Dispersion"])
summary.to_csv("metrics_summary.csv", index=False, encoding="utf-8-sig")
print("\nMetrics summary saved as 'metrics_summary.csv'")
