import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.calibration import calibration_curve

def plot_fusion_analysis(val_csv: str, train_csv: str):
    df_val = pd.read_csv(val_csv)
    df_train = pd.read_csv(train_csv)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Analiza fuzji: MegaDescriptor + LightGlue", fontsize=16, fontweight="bold")

    # ── 1. Bar chart metryk ──────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    methods = ["MegaDescriptor", "LightGlue", "Fuzja"]
    rank1  = [0.260, 0.172, 0.292]
    rank5  = [0.453, 0.266, 0.458]
    map_sc = [0.358, 0.239, 0.383]
    colors = ["#534AB7", "#1D9E75", "#D85A30"]

    x = np.arange(len(methods))
    w = 0.25
    ax1.bar(x - w, rank1,  w, label="Rank-1", color=[c + "CC" for c in colors] if False else colors, alpha=0.9)
    ax1.bar(x,     rank5,  w, label="Rank-5", color=colors, alpha=0.6)
    ax1.bar(x + w, map_sc, w, label="mAP",    color=colors, alpha=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel("Wartość metryki")
    ax1.set_title("Porównanie metryk (val subset)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── 2. Scatter: MegaDesc vs LightGlue, same vs different (val) ──────
    ax2 = fig.add_subplot(2, 3, 2)
    same = df_val[df_val["is_same_ground_truth"] == True]
    diff = df_val[df_val["is_same_ground_truth"] == False]
    ax2.scatter(diff["megadesc_score"], diff["lightglue_matches"],
                alpha=0.2, s=4, color="#534AB7", label="Different")
    ax2.scatter(same["megadesc_score"], same["lightglue_matches"],
                alpha=0.9, s=25, color="#1D9E75", label="Same", zorder=3)
    ax2.set_xlabel("MegaDescriptor score (cosine sim)")
    ax2.set_ylabel("LightGlue matches")
    ax2.set_title("Komplementarność sygnałów")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── 3. Histogram fusion_prob: same vs different ──────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(diff["fusion_prob"], bins=60, alpha=0.6, color="#534AB7",
             label="Different", density=True)
    ax3.hist(same["fusion_prob"], bins=60, alpha=0.8, color="#1D9E75",
             label="Same", density=True)
    ax3.set_xlabel("fusion_prob")
    ax3.set_ylabel("Density")
    ax3.set_title("Rozkład fusion_prob: same vs different")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ── 4. Reliability diagram ───────────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    try:
        prob_true, prob_pred = calibration_curve(
            df_val["is_same_ground_truth"],
            df_val["fusion_prob"],
            n_bins=10, strategy="quantile"
        )
        ax4.plot([0, 1], [0, 1], "--", color="gray", label="Idealna kalibracja")
        ax4.plot(prob_pred, prob_true, marker="o", linewidth=2,
                 color="#D85A30", label="Kalibrator")
    except ValueError as e:
        ax4.text(0.5, 0.5, f"Za mało danych:\n{e}", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=9)
    ax4.set_xlabel("Średnie przewidywane prawdopodobieństwo")
    ax4.set_ylabel("Frakcja pozytywów")
    ax4.set_title("Reliability Diagram")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # ── 5. Rozkład megadesc_score: same vs different (train) ─────────────
    ax5 = fig.add_subplot(2, 3, 5)
    same_tr = df_train[df_train["is_same_ground_truth"] == True]
    diff_tr = df_train[df_train["is_same_ground_truth"] == False]
    ax5.hist(diff_tr["megadesc_score"], bins=60, alpha=0.6, color="#534AB7",
             label="Different", density=True)
    ax5.hist(same_tr["megadesc_score"], bins=60, alpha=0.8, color="#1D9E75",
             label="Same", density=True)
    ax5.set_xlabel("MegaDescriptor score (cosine sim)")
    ax5.set_ylabel("Density")
    ax5.set_title("Rozkład MegaDesc score (train calibrator)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # ── 6. Rozkład lightglue_matches: same vs different (train) ──────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(diff_tr["lightglue_matches"], bins=60, alpha=0.6, color="#534AB7",
             label="Different", density=True)
    ax6.hist(same_tr["lightglue_matches"], bins=60, alpha=0.8, color="#1D9E75",
             label="Same", density=True)
    ax6.set_xlabel("LightGlue matches count")
    ax6.set_ylabel("Density")
    ax6.set_title("Rozkład LightGlue matches (train calibrator)")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("fusion_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Zapisano: fusion_analysis.png")


if __name__ == "__main__":
    plot_fusion_analysis(
        val_csv="val_calibrated_results_final.csv",
        train_csv="fusion_train_scores.csv",
    )