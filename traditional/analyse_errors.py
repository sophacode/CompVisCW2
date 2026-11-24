"""
traditional/analyze_errors.py

Reads saved prediction CSVs from joe.py and generates four high-resolution
error-analysis figures in traditional/outputs/plots/:
  1. top_confusions_svc.png        - Top 15 most frequent (True -> Pred) error pairs
  2. failure_grid_svc.png          - 4x4 visual contact sheet of misclassified signs
  3. worst_classes_comparison.png  - Side-by-side error rate on the 10 hardest classes
  4. model_disagreement.png        - Breakdown of where LinearSVC & RandomForest disagree
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================================================================
# CONFIGURATION & PATHS
# ===========================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "clean_dataset"
CLASS_CSV = ROOT_DIR / "class.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SVC_CSV = OUTPUT_DIR / "linear_svc_hog_baseline.csv"
RF_CSV = OUTPUT_DIR / "random_forest_hog_baseline.csv"


def load_class_names(csv_path: Path) -> dict[int, str]:
    """Map integer ClassId to human-readable SignName from class.csv."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["ClassId"], df["SignName"]))


def short_label(class_id: int, class_names: dict[int, str], max_len: int = 24) -> str:
    """Return a concise 'ID: SignName' string that fits cleanly on plots."""
    name = class_names.get(class_id, f"Class {class_id}")
    if len(name) > max_len:
        name = name[: max_len - 3] + "..."
    return f"{class_id}: {name}"


# ===========================================================================
# 1. TOP CONFUSION PAIRS (HORIZONTAL BAR CHART)
# ===========================================================================
def plot_top_confusions(
    df: pd.DataFrame,
    class_names: dict[int, str],
    model_label: str,
    save_path: Path,
    top_n: int = 15,
) -> None:
    errors = df[df["true_label"] != df["pred_label"]]
    pair_counts = (
        errors.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    labels = [
        f"{short_label(r.true_label, class_names, 24)}  →  {short_label(r.pred_label, class_names, 24)}"
        for r in pair_counts.itertuples(index=False)
    ]
    counts = pair_counts["count"].values

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(labels[::-1], counts[::-1], color="#d9534f", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Number of Misclassifications", fontsize=11)
    ax.set_title(f"Top {top_n} Confusion Pairs — {model_label}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ===========================================================================
# 2. VISUAL FAILURE GRID (CONTACT SHEET)
# ===========================================================================
def plot_failure_grid(
    df: pd.DataFrame,
    class_names: dict[int, str],
    model_label: str,
    save_path: Path,
    n_samples: int = 16,
    seed: int = 42,
) -> None:
    errors = df[df["true_label"] != df["pred_label"]]
    sample = errors.sample(n=min(n_samples, len(errors)), random_state=seed)

    cols = 4
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, row in zip(axes, sample.itertuples(index=False)):
        img_path = DATA_DIR / row.Path
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        ax.set_title(
            f"True: {short_label(row.true_label, class_names, 24)}\n"
            f"Pred: {short_label(row.pred_label, class_names, 24)}",
            fontsize=8.5,
            color="#222222",
        )
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(f"Sample Misclassified Signs — {model_label}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ===========================================================================
# 3. WORST-PERFORMING CLASSES (SIDE-BY-SIDE COMPARISON)
# ===========================================================================
def plot_worst_classes(
    df_svc: pd.DataFrame,
    df_rf: pd.DataFrame,
    class_names: dict[int, str],
    save_path: Path,
    top_n: int = 10,
) -> None:
    def class_error_rates(df: pd.DataFrame) -> pd.Series:
        is_err = (df["true_label"] != df["pred_label"]).astype(float)
        return df.assign(err=is_err).groupby("true_label")["err"].mean() * 100.0

    err_svc = class_error_rates(df_svc)
    err_rf = class_error_rates(df_rf)

    comp = pd.DataFrame({"LinearSVC": err_svc, "RandomForest": err_rf}).fillna(0.0)
    # Sort by highest average error rate across both models
    comp["avg_err"] = comp.mean(axis=1)
    worst = comp.sort_values("avg_err", ascending=False).head(top_n)

    labels = [short_label(cid, class_names, 26) for cid in worst.index]
    y = np.arange(len(labels))
    height = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y + height / 2, worst["LinearSVC"].values[::-1], height, label="LinearSVC", color="#337ab7")
    ax.barh(y - height / 2, worst["RandomForest"].values[::-1], height, label="RandomForest", color="#f0ad4e")

    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1], fontsize=9.5)
    ax.set_xlabel("Error Rate (%)", fontsize=11)
    ax.set_title(f"Top {top_n} Hardest Classes — LinearSVC vs. RandomForest (Raw HOG)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ===========================================================================
# 4. CLASSIFIER DISAGREEMENT BREAKDOWN
# ===========================================================================
def plot_model_disagreement(
    df_svc: pd.DataFrame,
    df_rf: pd.DataFrame,
    save_path: Path,
) -> None:
    merged = pd.merge(
        df_svc.rename(columns={"pred_label": "pred_svc"}),
        df_rf[["Path", "pred_label"]].rename(columns={"pred_label": "pred_rf"}),
        on="Path",
    )

    svc_ok = merged["true_label"] == merged["pred_svc"]
    rf_ok = merged["true_label"] == merged["pred_rf"]

    both_ok = int((svc_ok & rf_ok).sum())
    both_wrong = int((~svc_ok & ~rf_ok).sum())
    only_svc_ok = int((svc_ok & ~rf_ok).sum())
    only_rf_ok = int((~svc_ok & rf_ok).sum())
    total = len(merged)

    print("\n--- Classifier Agreement Summary ---")
    print(f"Total test images:            {total}")
    print(f"Both correct:                 {both_ok} ({both_ok / total:.2%})")
    print(f"Both wrong (hardest cases):   {both_wrong} ({both_wrong / total:.2%})")
    print(f"Only LinearSVC correct:       {only_svc_ok} ({only_svc_ok / total:.2%})")
    print(f"Only RandomForest correct:    {only_rf_ok} ({only_rf_ok / total:.2%})")

    # Plot the 3 error/disagreement buckets so differences are clearly visible
    categories = [
        "Both Wrong\n(HOG Ceiling)",
        "LinearSVC Correct\nRandomForest Wrong",
        "RandomForest Correct\nLinearSVC Wrong",
    ]
    counts = [both_wrong, only_svc_ok, only_rf_ok]
    colors = ["#d9534f", "#337ab7", "#f0ad4e"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(categories, counts, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax.set_ylabel("Number of Test Images", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.20)
    ax.set_title(
        f"Error & Disagreement Breakdown (Both Correct: {both_ok}/{total} = {both_ok / total:.2%})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{count}\n({count / total:.2%})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    class_names = load_class_names(CLASS_CSV)
    df_svc = pd.read_csv(SVC_CSV)
    df_rf = pd.read_csv(RF_CSV)

    plot_top_confusions(
        df_svc,
        class_names,
        model_label="LinearSVC (Raw HOG)",
        save_path=PLOTS_DIR / "top_confusions_svc.png",
    )
    plot_failure_grid(
        df_svc,
        class_names,
        model_label="LinearSVC (Raw HOG)",
        save_path=PLOTS_DIR / "failure_grid_svc.png",
    )
    plot_worst_classes(
        df_svc,
        df_rf,
        class_names,
        save_path=PLOTS_DIR / "worst_classes_comparison.png",
    )
    plot_model_disagreement(
        df_svc,
        df_rf,
        save_path=PLOTS_DIR / "model_disagreement.png",
    )


if __name__ == "__main__":
    main()