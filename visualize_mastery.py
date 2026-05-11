"""
Visualize mastery curves từ kết quả zero-shot LLMKT.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12

PREDICTIONS_FILE = "results/comta_zeroshot_fold1.json"
KC_FILE = "results/comta_zeroshot_fold1_kcs.json"
OUTPUT_DIR = "results"


def load_data():
    with open(PREDICTIONS_FILE) as f:
        predictions = json.load(f)
    with open(KC_FILE) as f:
        kc_results = json.load(f)
    return predictions, kc_results


def plot1_pred_distribution(predictions):
    """Hình 1: Phân bố predicted probability vs actual labels."""
    correct = [p["pred_prob"] for p in predictions if p["label"] == 1]
    incorrect = [p["pred_prob"] for p in predictions if p["label"] == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct, bins=15, alpha=0.6, label=f"Correct (n={len(correct)})", color="#2196F3")
    ax.hist(incorrect, bins=15, alpha=0.6, label=f"Incorrect (n={len(incorrect)})", color="#F44336")
    ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold (0.5)")
    ax.set_xlabel("Predicted P(mastery)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted Mastery Probabilities")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_pred_distribution.png", dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fig_pred_distribution.png")


def plot2_mastery_curves(kc_results):
    """Hình 2: Mastery curves theo KC occurrence (như Figure 2 của paper)."""
    kc_to_probs = {}
    for dia_idx, turns in kc_results.items():
        dia_kc_probs = {}
        for turn_data in turns:
            for kc, prob in turn_data["kcs"].items():
                dia_kc_probs.setdefault(kc, []).append(prob)
        for kc, probs in dia_kc_probs.items():
            kc_to_probs.setdefault(kc, []).append(probs)

    # Sort by frequency
    common_kcs = sorted(kc_to_probs.items(), key=lambda x: -len(x[1]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for kc_idx, (kc, prob_lists) in enumerate(common_kcs[:6]):
        ax = axes[kc_idx]
        # Plot deltas
        delta_lists = [[p - probs[0] for p in probs] for probs in prob_lists]
        max_len = max(len(d) for d in delta_lists)

        means, stds = [], []
        for i in range(max_len):
            vals = [d[i] for d in delta_lists if i < len(d)]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        x = np.arange(1, len(means) + 1)
        ax.errorbar(x, means, yerr=stds, marker='o', linewidth=2, capsize=4, ecolor='gray')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("KC Occurrence")
        ax.set_ylabel("Δ Mastery")
        short_kc = kc[:45] + "..." if len(kc) > 45 else kc
        ax.set_title(f"KC: {short_kc}\n(n={len(prob_lists)} dialogues)", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Mastery Curves (Change from Initial) — Top 6 KCs", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_mastery_curves.png", dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fig_mastery_curves.png")


def plot3_case_study(predictions):
    """Hình 3: Case study — 1 dialogue có mastery thay đổi qua turns."""
    # Tìm dialogue có nhiều turns nhất
    dia_data = {}
    for p in predictions:
        dia_data.setdefault(p["dialogue_idx"], []).append(p)

    best_dia = max(dia_data.values(), key=len)

    fig, ax = plt.subplots(figsize=(10, 5))
    turns = [p["turn"] for p in best_dia]
    probs = [p["pred_prob"] for p in best_dia]
    labels = [p["label"] for p in best_dia]

    # Plot probability line
    ax.plot(turns, probs, 'b-o', linewidth=2, markersize=8, label="P(correct)")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Mark correct/incorrect
    for t, p, l in zip(turns, probs, labels):
        color = "#2196F3" if l == 1 else "#F44336"
        marker = "✓" if l == 1 else "✗"
        ax.annotate(marker, (t, p), textcoords="offset points",
                   xytext=(0, 12), fontsize=14, color=color, ha='center')

    ax.set_xlabel("Dialogue Turn")
    ax.set_ylabel("Predicted P(correct)")
    ax.set_title(f"Case Study: Mastery Trajectory (Dialogue {best_dia[0]['dialogue_idx']})")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_case_study.png", dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fig_case_study.png")


def print_summary_table(predictions):
    """In bảng so sánh với baselines cho luận văn."""
    labels = [p["label"] for p in predictions]
    probs = [p["pred_prob"] for p in predictions]
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    auc = roc_auc_score(labels, probs)
    preds = [1 if p > 0.5 else 0 for p in probs]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    print("\n" + "=" * 65)
    print("TABLE FOR CHAPTER 4 — CoMTA Results")
    print("=" * 65)
    print(f"{'Method':<20} {'Acc.':<12} {'AUC':<12} {'F1':<12}")
    print("-" * 56)
    print(f"{'BKT':<20} {'51.02':<12} {'52.50':<12} {'53.41':<12}")
    print(f"{'DKT':<20} {'52.59':<12} {'53.20':<12} {'49.56':<12}")
    print(f"{'DKVMN':<20} {'47.22':<12} {'46.81':<12} {'48.80':<12}")
    print(f"{'AKT':<20} {'49.95':<12} {'51.40':<12} {'50.78':<12}")
    print(f"{'SAINT':<20} {'49.17':<12} {'47.81':<12} {'50.38':<12}")
    print(f"{'simpleKT':<20} {'51.25':<12} {'51.25':<12} {'49.82':<12}")
    print(f"{'DKT-Sem':<20} {'56.83':<12} {'61.82':<12} {'63.27':<12}")
    print(f"{'LLMKT (fine-tune)':<20} {'58.01':<12} {'65.79':<12} {'60.72':<12}")
    print("-" * 56)
    print(f"{'Ours (zero-shot)':<20} {acc*100:<12.2f} {auc*100:<12.2f} {f1*100:<12.2f}")
    print("=" * 65)
    print("\n* Baselines quoted from Scarlatos et al. (2025)")
    print("* Ours: Llama-3.1-8B, 4-bit NF4, zero-shot, mean-ar aggregation")


def main():
    print("Generating visualizations for thesis...")
    predictions, kc_results = load_data()

    plot1_pred_distribution(predictions)
    plot2_mastery_curves(kc_results)
    plot3_case_study(predictions)
    print_summary_table(predictions)

    print("\n✓ All figures saved to results/")
    print("  → fig_pred_distribution.png (Chapter 4)")
    print("  → fig_mastery_curves.png (Chapter 4)")
    print("  → fig_case_study.png (Chapter 4)")


if __name__ == "__main__":
    main()