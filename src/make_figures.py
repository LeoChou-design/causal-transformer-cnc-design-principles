"""Figures for the ablation studies and the final configuration check."""

import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def fig1_causal_mask_ablation():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ablation1_causal_mask.csv"))
    means = df.groupby("config")[["auroc_TWF", "auroc_OSF", "wear_r2"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["auroc_TWF", "auroc_OSF", "wear_r2"]):
        means[col].plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.set_title(col)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Ablation 1: Causal Mask vs. Bidirectional Attention (GAP pooling)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_causal_mask_ablation.png"), dpi=150)
    plt.close()


def fig2_mtl_ablation():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ablation2_mtl.csv"))
    means = df.groupby("config")["auroc_TWF"].mean().reindex(
        ["joint_fixed", "single_task_fault", "joint_dynamic"]
    )
    plt.figure(figsize=(7, 5))
    means.plot(kind="bar", color="#55A868")
    plt.title("Ablation 2: TWF AUROC by MTL Configuration")
    plt.ylabel("AUROC")
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_mtl_ablation.png"), dpi=150)
    plt.close()


def fig3_window_length_sweep():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "window_length_sweep.csv"))
    pivot = df.pivot(index="window_length", columns="config", values="wear_r2")
    plt.figure(figsize=(7, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)
    plt.xlabel("Window length L")
    plt.ylabel("Wear regression R2")
    plt.title("Window-Length Sweep: Causal+GAP Degrades, Bidirectional+GAP Stays Flat")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_window_length_sweep.png"), dpi=150)
    plt.close()


def fig4_pooling_ablation():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "pooling_ablation.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["auroc_TWF", "auroc_OSF", "wear_r2"]):
        ax.bar(df["pooling"], df[col], color=["#C44E52", "#4C72B0"])
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Pooling Ablation at L=50 (causal mask fixed)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_pooling_ablation.png"), dpi=150)
    plt.close()


def fig5_final_configuration():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "final_significance_fold_avg.csv"))
    order = ["causal+gap (original baseline)", "causal+last (final configuration)", "bidirectional+gap"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, ["auroc_TWF", "auroc_OSF", "wear_r2"]):
        means = df.groupby("config")[col].mean().reindex(order)
        stds = df.groupby("config")[col].std().reindex(order)
        ax.bar(range(len(means)), means.values, yerr=stds.values, capsize=5,
               color=["#8C8C8C", "#55A868", "#4C72B0"])
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(order, rotation=20, ha="right")
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Final Configuration vs. Baseline vs. Bidirectional (mean +/- std across folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_final_configuration.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig1_causal_mask_ablation()
    fig2_mtl_ablation()
    fig3_window_length_sweep()
    fig4_pooling_ablation()
    fig5_final_configuration()
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
