from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_labels(ax, bars, digits=4):
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.{digits}f}",
            va="center",
            fontsize=9,
        )


def human_standard_results():
    models = [
        "Shared heads",
        "Auxiliary dual branch",
        "3-seed MLP ensemble",
        "TissuePMHC",
    ]
    metrics = {
        "Mean task AUROC": [0.7972, 0.8239, 0.8345, 0.8448],
        "Mean task AUPRC": [0.7829, 0.8117, 0.8233, 0.8348],
        "Worst-10 AUROC": [0.6299, 0.6501, 0.6666, 0.6834],
    }
    colors = ["#7f8c8d", "#7f8c8d", "#d7821f", "#167d91"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    y = np.arange(len(models))
    for index, (title, values) in enumerate(metrics.items()):
        bars = axes[index].barh(y, values, color=colors)
        axes[index].set_title(title)
        axes[index].grid(axis="x", alpha=0.2)
        axes[index].set_axisbelow(True)
        axes[index].set_xlim(min(values) - 0.02, max(values) + 0.025)
        add_labels(axes[index], bars)
        if index == 0:
            axes[index].set_yticks(y, models)
        else:
            axes[index].set_yticks(y, [])
        axes[index].invert_yaxis()
    fig.suptitle("Human standard pair-disjoint benchmark", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "human_standard_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def mouse_confirmation_results():
    metrics = ["Mean task AUROC", "Mean task AUPRC", "Worst-6 AUROC"]
    oof = np.array([0.8392, 0.8316, 0.7101])
    fixed = np.array([0.8562, 0.8506, 0.7245])
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    bars_oof = ax.bar(x - width / 2, oof, width, label="Train-only OOF", color="#7f8c8d")
    bars_fixed = ax.bar(x + width / 2, fixed, width, label="Frozen fixed test", color="#167d91")
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Task-macro score")
    ax.set_ylim(0.66, 0.88)
    ax.set_title("Mouse standard pair-disjoint confirmation")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for bars in (bars_oof, bars_fixed):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mouse_standard_confirmation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    human_standard_results()
    mouse_confirmation_results()
