from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parents[1] / "results" / "tissuePMHC_model_principles_formulas"
FORMULAS = {
    "e0_linear": r"$s=b+\sum_{i=1}^{9}\sum_a w_{i,a}x_{i,a}$",
    "e2_shared": r"$z=f_\theta(p),\qquad \hat{y}_t=\sigma(w_t^\top z+b_t)$",
    "e8_fusion": r"$s_{\mathrm{E8}}=\frac{s_{\mathrm{global}}+s_{\mathrm{HLA}}}{2}$",
    "e15_rank": r"$r_{\mathrm{fusion}}=\frac{r_{\mathrm{global}}+r_{\mathrm{HLA}}}{2}$",
    "e17_seed": r"$\bar{p}(x)=\frac{1}{K}\sum_{k=1}^{K}p_k(x)$",
    "e29_gain": r"$\text{better local motif representation}+\text{independent seed averaging}$",
}


def render(name: str, formula: str) -> None:
    fig = plt.figure(figsize=(7.0, 0.72), dpi=220, facecolor="white")
    fig.text(0.5, 0.5, formula, ha="center", va="center", fontsize=18, color="black")
    plt.axis("off")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, formula in FORMULAS.items():
        render(name, formula)
    print(OUT)


if __name__ == "__main__":
    main()
