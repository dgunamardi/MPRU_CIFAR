import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, List

# ==============
# === Loader ===
# ==============

@dataclass
class ModelResults: 
    confidences: Dict[int, np.ndarray]      # per-class confidence arrays (N_cls, C)
    preds: Dict[int, np.ndarray]            # per-class prediction arrays (N_cls,)
    accuracy: np.ndarray                    # per-class accuracy (C,)
    targets: Dict[int, np.ndarray] = None   # per-class target arrays (N_cls,)

def model_results_from_npz(filepath: str, num_classes: int) -> ModelResults: 
    """
        Get model results from npz file, receives filepath.
        Outputs:
            Confidences (N, C)
            Preds (N, )
            Accuracy (C, )
    """
    npz_data = np.load(filepath)
    preds = npz_data['preds'].astype(int)
    targets = npz_data['targets'].astype(int)
    confs = npz_data['confs']  # shape (N, C)

    # Preallocate dicts
    class_confidences = {cls: [] for cls in range(num_classes)}
    class_preds = {cls: [] for cls in range(num_classes)}
    class_targets = {cls: [] for cls in range(num_classes)}
    class_correct = np.zeros(num_classes, dtype=int)
    class_total = np.zeros(num_classes, dtype=int)

    # Group by target
    for p, t, c in zip(preds, targets, confs):
        class_confidences[t].append(c)
        class_preds[t].append(p)
        class_targets[t].append(t)
        class_total[t] += 1
        class_correct[t] += int(p == t)

    # Convert lists → arrays
    for cls in class_confidences:
        class_confidences[cls] = np.stack(class_confidences[cls]) # (N, C)
        class_preds[cls] = np.array(class_preds[cls], dtype=int) # (N,)
        class_targets[cls] = np.array(class_targets[cls], dtype=int) # (N,)


    # --- Compute per-class accuracy ---
    class_accuracy = class_correct / class_total  # per-class accuracy

    return ModelResults(
        confidences=class_confidences, 
        preds=class_preds, 
        accuracy=class_accuracy,
        targets=class_targets,
    )

# ===============
# === Numbers ===
# ===============

def normalize_l1(arr: np.ndarray, axis=None) -> np.ndarray:
    """
    Perform L1 normalization (divide by sum of absolute values).
    Works for 1D and 2D arrays.

    Parameters
    ----------
    arr : np.ndarray
        Input array (1D or 2D).
    axis : int or None, optional
        Axis along which to normalize. 
        - None → normalize whole array (1D only).
        - 0 → normalize column-wise.
        - 1 → normalize row-wise.

    Returns
    -------
    np.ndarray
        L1-normalized array.
    """
    arr = np.asarray(arr, dtype=float)
    denom = np.sum(np.abs(arr), axis=axis, keepdims=True)
    denom[denom == 0] = 1.0  # avoid divide-by-zero
    return arr / denom

# ===============
# === Compare ===
# ===============

@dataclass
class ResultVersion:
    model: ModelResults
    name: str

def compare_results(
    classes: List[str],
    results: List[ResultVersion], 
    removed_idx: int, 
    seed: str = "42",
    mode: str = "forget", # forget, retain, all
    out_csv: str = None,
    out_png: str = None,
) -> pd.DataFrame:
    """
    Compare predictions across multiple model versions.

    Args:
        results: list of ResultVersion objects (name + model)
        removed_idx: index of the removed class
        mode: "forget" -> only removed class
              "retain" -> all classes except removed
              "all" -> all classes
        out_csv: optional CSV path to save results
        out_png: optional PNG path to save plot
    Returns:
        pd.DataFrame with counts per class per version
    """
    # --- Font sizes ---
    title_fs = 18
    label_fs = 16
    tick_fs = 14
    annot_fs = 12
    legend_fs = 14

    if mode not in {"forget", "retain", "all"}:
        raise ValueError(f"Invalid mode {mode}. Choose 'forget', 'retain', or 'all'.")

    # Index
    num_classes = len(classes)
    df_dict = {
        "Class Index": np.arange(num_classes),
        "Class Name": classes 
    }

    # Boolean mask: True means include this class
    keep_mask = np.zeros(num_classes, dtype=bool)
    if mode == "forget":
        keep_mask[removed_idx] = True
    elif mode == "retain":
        keep_mask[:] = True
        keep_mask[removed_idx] = False
    else:  # mode == "all"
        keep_mask[:] = True

    # Collect counts for each version
    for version in results:
        # Gather all predictions for the selected classes
        selected_preds = [
            version.model.preds[cls] for cls, keep in enumerate(keep_mask) if keep
        ]
        if selected_preds:
            all_preds = np.concatenate(selected_preds)
        else:
            all_preds = np.array([], dtype=int)

        counts = np.bincount(all_preds, minlength=num_classes)
        df_dict[f"{version.name} Count"] = counts

    df = pd.DataFrame(df_dict)

    # --- Plot ---
    plt.figure(figsize=(12, 7))
    for version in results:
        counts = df[f"{version.name} Count"].values
        plt.plot(classes, counts, marker="o", label=version.name)
        # Annotate each point with count
        for x, y in zip(classes, counts):
            plt.text(x, y + 1, str(y), ha='center', va='bottom', fontsize=annot_fs)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)

    plt.xticks(rotation=45, fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.ylabel("Prediction %", fontsize=label_fs)
    plt.xlabel("Classes", fontsize=label_fs)
    #plt.title(f"{mode.capitalize()} Set Distribution (Removed = {classes[removed_idx]}) | Model Seed = {seed}", fontsize=title_fs)
    plt.title(f"{mode.capitalize()} Set Distribution (Removed = {classes[removed_idx]})", fontsize=title_fs)
    plt.legend(fontsize=legend_fs)
    plt.tight_layout()
    
    # Save CSV and PNG if specified
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)

    plt.show()
    
    return df

def compare_results_cifar100(
    classes: List[str],
    results: List[ResultVersion], 
    removed_idx: int, 
    selected_classes: List[int] = None,
    seed: str = "42",
    mode: str = "forget",  # forget, retain, all
    out_csv: str = None,
    out_png: str = None,
) -> pd.DataFrame:
    """
    Compare predictions for CIFAR-100 with option to select a subset of classes.
    
    Args:
        classes: list of all 100 class names
        results: list of ResultVersion objects (name + model)
        removed_idx: index of the removed class
        selected_classes: list of class indices to show in the chart (subset)
        mode: "forget" -> only removed class
              "retain" -> all classes except removed
              "all" -> all classes
        out_csv: optional CSV path to save results
        out_png: optional PNG path to save plot
    Returns:
        pd.DataFrame with counts per class per version
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import pandas as pd

    # --- Font sizes ---
    title_fs = 20
    label_fs = 16
    tick_fs = 14
    annot_fs = 10
    legend_fs = 14

    if mode not in {"forget", "retain", "all"}:
        raise ValueError(f"Invalid mode {mode}. Choose 'forget', 'retain', or 'all'.")

    num_classes = len(classes)
    df_dict = {
        "Class Index": np.arange(num_classes),
        "Class Name": classes 
    }

    # Boolean mask
    keep_mask = np.zeros(num_classes, dtype=bool)
    if mode == "forget":
        keep_mask[removed_idx] = True
    elif mode == "retain":
        keep_mask[:] = True
        keep_mask[removed_idx] = False
    else:  # "all"
        keep_mask[:] = True

    # Collect counts
    for version in results:
        selected_preds = [
            version.model.preds[cls] for cls, keep in enumerate(keep_mask) if keep
        ]
        if selected_preds:
            all_preds = np.concatenate(selected_preds)
        else:
            all_preds = np.array([], dtype=int)
        counts = np.bincount(all_preds, minlength=num_classes)
        df_dict[f"{version.name} Count"] = counts

    df = pd.DataFrame(df_dict)

    # Subset selection
    if selected_classes is not None:
        df_plot = df.loc[selected_classes].copy()
    else:
        df_plot = df.copy()

    # --- Plot ---
    plt.figure(figsize=(14, 7))
    for version in results:
        counts = df_plot[f"{version.name} Count"].values
        class_labels = df_plot["Class Name"].values
        plt.plot(class_labels, counts, marker="o", label=version.name)
        # Annotate
        for x, y in zip(class_labels, counts):
            plt.text(x, y + 1, str(y), ha='center', va='bottom', fontsize=annot_fs)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_plot.to_csv(out_csv, index=False)

    plt.xticks(rotation=90, fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.ylabel("Prediction %", fontsize=label_fs)
    plt.xlabel("Classes", fontsize=label_fs)
    plt.title(f"{mode.capitalize()} Set Distribution (Removed = {classes[removed_idx]})", fontsize=title_fs)
    plt.legend(fontsize=legend_fs)
    plt.tight_layout()
    
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)

    plt.show()
    
    return df_plot