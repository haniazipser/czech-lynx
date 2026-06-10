import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from wildfusion.wildfusion import ScoresCalibrator
from sklearn.calibration import calibration_curve


def plot_reliability_diagram(csv_path):
    df = pd.read_csv(csv_path)

    print(f"Pozytywów: {df['is_same_ground_truth'].sum()}")
    print(f"Negatywów: {(~df['is_same_ground_truth']).sum()}")

    prob_true, prob_pred = calibration_curve(
        df['is_same_ground_truth'],
        df['fusion_prob'],
        n_bins=10,
        strategy='quantile'  # równa liczba próbek w każdym binie zamiast równe przedziały
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Idealna kalibracja')
    ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Kalibrator')
    ax1.set_xlabel('Średnie przewidywane prawdopodobieństwo')
    ax1.set_ylabel('Frakcja pozytywów')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.grid(True)

    # Histogram fusion_prob — same vs different
    ax2.hist(df[df['is_same_ground_truth'] == False]['fusion_prob'],
             bins=50, alpha=0.6, color='#534AB7', label='Different', density=True)
    ax2.hist(df[df['is_same_ground_truth'] == True]['fusion_prob'],
             bins=50, alpha=0.6, color='#1D9E75', label='Same', density=True)
    ax2.set_xlabel('fusion_prob')
    ax2.set_ylabel('Density')
    ax2.set_title('Rozkład fusion_prob: same vs different')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('reliability_diagram.png', dpi=150)
    plt.show()

def compute_metrics(df, prob_col):
    rank1_correct = 0
    rank5_correct = 0
    average_precisions = []

    queries = df['query_img'].unique()

    for q in queries:
        q_df = df[df['query_img'] == q].copy()
        q_df = q_df.sort_values(by=prob_col, ascending=False).reset_index(drop=True)

        # Rank-1
        if q_df.loc[0, 'is_same_ground_truth'] == True:
            rank1_correct += 1

        # Rank-5
        if q_df.head(5)['is_same_ground_truth'].any():
            rank5_correct += 1

        # AP
        correct = q_df['is_same_ground_truth'].astype(float).values
        n_relevant = correct.sum()
        if n_relevant == 0:
            continue

        cumsum = np.cumsum(correct)
        ranks = np.arange(1, len(correct) + 1)
        precision_at_k = cumsum / ranks
        ap = (precision_at_k * correct).sum() / n_relevant
        average_precisions.append(ap)

    rank1 = rank1_correct / len(queries)
    rank5 = rank5_correct / len(queries)
    map_score = np.mean(average_precisions)

    return rank1, rank5, map_score


def train_calibrator(train_csv, val_csv):
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    calibrator = ScoresCalibrator()
    calibrator.fit(
        global_scores=df_train["megadesc_score"],
        local_scores=df_train["lightglue_matches"],
        labels=df_train["is_same_ground_truth"],
    )

    df_val['fusion_prob'] = calibrator.predict_probability(
        df_val["megadesc_score"],
        df_val["lightglue_matches"]
    )

    print("\n--- RESULTS ON VAL SET ---")

    r1_m, r5_m, map_m = compute_metrics(df_val, 'megadesc_score')
    r1_l, r5_l, map_l = compute_metrics(df_val, 'lightglue_matches')
    r1_f, r5_f, map_f = compute_metrics(df_val, 'fusion_prob')

    print(f"MegaDescriptor: Rank-1={r1_m:.3f}, Rank-5={r5_m:.3f}, mAP={map_m:.3f}")
    print(f"LightGlue:      Rank-1={r1_l:.3f}, Rank-5={r5_l:.3f}, mAP={map_l:.3f}")
    print(f"Fusion:         Rank-1={r1_f:.3f}, Rank-5={r5_f:.3f}, mAP={map_f:.3f}")
    df_val.to_csv("val_calibrated_results_final.csv", index=False)
    print("\nSaved to: val_calibrated_results_final.csv")


def main():
    train_calibrator("fusion_train_scores.csv", "fusion_val_scores.csv")

if __name__ == "__main__":
    main()