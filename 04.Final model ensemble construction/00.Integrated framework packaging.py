"""
- Sequentially select 5 base learners in a fixed order: ENR, SVR, DTR, KNNR, MLPR (.pkl/.joblib)
- Select a Transformer meta-learner (.pt/.pth, containing a dict with model_state/struct_hparams/n_features)
- Package all into a single file (.pkl) containing only "bytes + metadata". 
  Prediction script will be responsible for restoring the models.
"""

import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

BASE_NAMES = ["ENR", "SVR", "DTR", "KNNR", "MLPR"]
TORCH_EXTS = {".pt", ".pth"}

# Read a file in binary mode
def read_binary(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# Open a file picker dialog
def pick_file(title: str, patterns):
    return filedialog.askopenfilename(
        title=title,
        filetypes=patterns
    ) or ""

# Open a save file dialog
def pick_save_path() -> str:
    return filedialog.asksaveasfilename(
        title="Save ensemble bundle (.pkl)",
        defaultextension=".pkl",
        filetypes=[("Pickle", "*.pkl"), ("All Files", "*.*")]
    ) or ""

def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    try:
        # 1) Sequentially select 5 base learners
        base_files = []
        for name in BASE_NAMES:
            messagebox.showinfo("Select Base Learner", f"Please select {name} model file (.pkl/.joblib)")
            p = pick_file(f"Select {name} model file",
                          [("Pickle/Joblib", "*.pkl *.joblib"), ("All Files", "*.*")])
            if not p:
                messagebox.showwarning("Aborted", f"{name} not selected.")
                return
            base_files.append(p)

        # 2) Select Transformer meta-learner (.pt/.pth)
        messagebox.showinfo("Select Meta-Learner", "Please select Transformer meta-learner (.pt/.pth)")
        meta_path = pick_file("Select Transformer meta-learner",
                              [("PyTorch", "*.pt *.pth"), ("All Files", "*.*")])
        if not meta_path:
            messagebox.showwarning("Aborted", "Meta-learner not selected.")
            return

        meta_ext = os.path.splitext(meta_path)[1].lower()
        if meta_ext not in TORCH_EXTS:
            messagebox.showwarning("Invalid Format", "Please provide .pt/.pth (PyTorch dict containing model_state etc.)")
            return

        # 3) Choose save location
        save_path = pick_save_path()
        if not save_path:
            messagebox.showwarning("Aborted", "No save path selected.")
            return

        # 4) Package as "bytes + metadata"
        bundle = {
            "format": "ensemble_bundle.v3",
            "base_names": BASE_NAMES,
            "base_files": [os.path.basename(p) for p in base_files],
            "base_blobs": [read_binary(p) for p in base_files],   # pickle/joblib bytes
            "meta_file": os.path.basename(meta_path),
            "meta_blob": read_binary(meta_path),                  # PyTorch .pt/.pth bytes
            "meta_kind": "torch_state_v1",                        # explicit save format label
        }

        # Save bundle
        with open(save_path, "wb") as f:
            pickle.dump(bundle, f)

        messagebox.showinfo(
            "Done",
            f"Ensemble bundle saved:\n{save_path}\n\n"
            f"Base learners order: {BASE_NAMES}\n"
            f"Meta-learner: {bundle['meta_file']} ({bundle['meta_kind']})"
        )

    except Exception as e:
        messagebox.showerror("Error", f"{e}")
        raise

if __name__ == "__main__":
    main()
