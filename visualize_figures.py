"""
python visualize_figures.py

Reads from:
  results/comta_zeroshot_fold1_metrics.json
  results/mathdial_zeroshot_metrics.json
  results/error_eval_metrics.json
  results/error_summary.json
  results/feedback_eval_metrics.json
  results/mini_finetune_test_eval.json
  results/mini_finetune_log.json

Outputs to results/fig_*.png
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load all metrics
comta_metrics = load_json(f"{RESULTS_DIR}/comta_zeroshot_fold1_metrics.json")
mathdial_metrics = load_json(f"{RESULTS_DIR}/mathdial_zeroshot_metrics.json")
error_eval = load_json(f"{RESULTS_DIR}/error_eval_metrics.json")
error_summary = load_json(f"{RESULTS_DIR}/error_summary.json")
feedback_eval = load_json(f"{RESULTS_DIR}/feedback_eval_metrics.json")
ft_test = load_json(f"{RESULTS_DIR}/mini_finetune_test_eval.json")
ft_log = load_json(f"{RESULTS_DIR}/mini_finetune_log.json")

print(f"Loaded results:")
print(f"  CoMTA AUC      = {comta_metrics['overall']['auc']:.4f}  (n={comta_metrics['overall']['n']})")
print(f"  MathDial AUC   = {mathdial_metrics['overall']['auc']:.4f}  (n={mathdial_metrics['overall']['n']})")
print(f"  Error agreement= {error_eval['agreement_rate']:.4f}  (n={error_eval['n_annotated']})")
print(f"  Feedback score = {feedback_eval['overall']['mean']:.2f}")
print(f"  FT AUC         = {ft_test['fine_tuned']['auc']:.4f} vs ZS {ft_test['zero_shot']['auc']:.4f}")
print()

# =============================================================
# Figure 1: AUC baseline comparison
# Baseline numbers from paper LLMKT (Scarlatos et al. 2025) — hardcoded
# Ours numbers from our JSON
# =============================================================
methods = ['BKT', 'DKT', 'DKVMN', 'AKT', 'SAINT', 'simpleKT', 'DKT-Sem',
           'LLMKT\nzero-shot\n(ours)', 'LLMKT\nfull FT\n(paper)']

# Paper baselines — fixed, không có trong JSON của ta
comta_baselines = [0.525, 0.532, 0.468, 0.514, 0.478, 0.513, 0.618]
mathdial_baselines = [0.642, 0.632, 0.604, 0.633, 0.601, 0.638, 0.662]

# Ours — đọc từ JSON
ours_comta = comta_metrics["overall"]["auc"]
ours_mathdial = mathdial_metrics["overall"]["auc"]

# Paper full fine-tuned — hardcoded (không có trong code base của ta)
paper_ft_comta = 0.658
paper_ft_mathdial = 0.767

comta_auc = comta_baselines + [ours_comta, paper_ft_comta]
mathdial_auc = mathdial_baselines + [ours_mathdial, paper_ft_mathdial]

x = np.arange(len(methods))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 5))
colors_c = ['#7F77DD'] * 9
colors_m = ['#1D9E75'] * 9
colors_c[7] = colors_m[7] = '#D85A30'  # highlight ours

b1 = ax.bar(x - width/2, comta_auc, width, color=colors_c)
b2 = ax.bar(x + width/2, mathdial_auc, width, color=colors_m)

ax.set_ylabel('AUC')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0.4, 0.8)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label='Random')

ax.legend(handles=[
    mpatches.Patch(color='#7F77DD', label='CoMTA'),
    mpatches.Patch(color='#1D9E75', label='MathDial'),
    mpatches.Patch(color='#D85A30', label='Ours (zero-shot)'),
], loc='upper left', frameon=False)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
                ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig_auc_baseline_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] fig_auc_baseline_comparison.png")


# =============================================================
# Figure 2: Error type distribution (donut)
# Đọc trực tiếp từ error_summary.json hoặc error_eval_metrics.json
# =============================================================
dist = error_summary["error_type_distribution"]
total = sum(dist.values())

# Sort by count descending
sorted_items = sorted(dist.items(), key=lambda x: -x[1])
labels = [k.capitalize() for k, _ in sorted_items]
sizes = [v / total * 100 for _, v in sorted_items]
counts = [v for _, v in sorted_items]

color_map = {
    'Conceptual': '#D85A30',
    'Procedural': '#BA7517',
    'Other': '#888780',
    'Careless': '#534AB7',
    'Calculation': '#0F6E56',
}
colors = [color_map.get(l, '#888780') for l in labels]

fig, ax = plt.subplots(figsize=(7, 5))
wedges, texts, autotexts = ax.pie(
    sizes, labels=[f"{l}\n(n={c})" for l, c in zip(labels, counts)],
    colors=colors, autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
    textprops={'fontsize': 11}
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax.set_title(f'Error type distribution (n={total} predicted, {error_summary["total_incorrect_turns"]} total turns)', pad=20)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig_error_type_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] fig_error_type_distribution.png")


# =============================================================
# Figure 3: Feedback evaluation scores (4 dimensions)
# Đọc trực tiếp từ feedback_eval_metrics.json
# =============================================================
dims = ['Correctness', 'Relevance', 'Clarity', 'Overall']
means = [
    feedback_eval['correctness_score']['mean'],
    feedback_eval['relevance_score']['mean'],
    feedback_eval['clarity_score']['mean'],
    feedback_eval['overall']['mean'],
]
stds = [
    feedback_eval['correctness_score']['std'],
    feedback_eval['relevance_score']['std'],
    feedback_eval['clarity_score']['std'],
    feedback_eval['overall']['std'],
]

fig, ax = plt.subplots(figsize=(8, 5))
colors_fb = ['#5DCAA5', '#5DCAA5', '#5DCAA5', '#1D9E75']
bars = ax.bar(dims, means, yerr=stds, capsize=6, color=colors_fb,
              error_kw={'ecolor': '#444441', 'linewidth': 1})
ax.set_ylabel('Score (out of 5)')
ax.set_ylim(0, 5)
ax.axhline(y=3, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.1,
            f'{m:.2f}±{s:.2f}', ha='center', va='bottom', fontsize=10)

ax.set_title('Feedback quality evaluation (LLM-as-judge, n=30 cases)')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig_feedback_quality.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] fig_feedback_quality.png")


# =============================================================
# Figure 4: Fine-tune improvement (grouped bars)
# Đọc trực tiếp từ mini_finetune_test_eval.json
# =============================================================
metrics_names = ['AUC', 'Accuracy', 'F1 macro']
zeroshot = [
    ft_test['zero_shot']['auc'],
    ft_test['zero_shot']['accuracy'],
    ft_test['zero_shot']['f1_macro'],
]
finetuned = [
    ft_test['fine_tuned']['auc'],
    ft_test['fine_tuned']['accuracy'],
    ft_test['fine_tuned']['f1_macro'],
]
improvements = [(f - z) / z * 100 for z, f in zip(zeroshot, finetuned)]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width/2, zeroshot, width, label='Zero-shot', color='#B4B2A9')
b2 = ax.bar(x + width/2, finetuned, width,
            label='Fine-tuned (200 samples, 2 epochs)', color='#1D9E75')

ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0.4, 0.6)
ax.legend(loc='upper left', frameon=False)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)

for i, imp in enumerate(improvements):
    ax.annotate(f'+{imp:.1f}%',
                xy=(i + width/2, finetuned[i] + 0.015),
                xytext=(i + width/2, finetuned[i] + 0.025),
                ha='center', fontsize=9, color='#0F6E56', fontweight='bold')

ax.set_title(f'Mastery estimator: zero-shot vs fine-tuned (n={ft_test["fine_tuned"]["n"]} per-KC predictions)')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig_finetune_improvement.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] fig_finetune_improvement.png")


# =============================================================
# Figure 5 (BONUS): Training loss curve
# Đọc từ mini_finetune_log.json
# =============================================================
losses = ft_log['train_loss']
val_metrics = ft_log['val_metrics']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Training loss
ax1.plot(range(1, len(losses) + 1), losses, color='#534AB7', linewidth=1.5, marker='o', markersize=4)
ax1.set_xlabel('Training step')
ax1.set_ylabel('Loss')
ax1.set_title(f'Training loss over {len(losses)} steps')
ax1.grid(alpha=0.3)

# Val AUC
val_steps = [v['step'] for v in val_metrics]
val_aucs = [v['auc'] for v in val_metrics]
val_accs = [v['acc'] for v in val_metrics]

ax2.plot(val_steps, val_aucs, color='#1D9E75', linewidth=2, marker='o', markersize=8, label='Val AUC')
ax2.plot(val_steps, val_accs, color='#D85A30', linewidth=2, marker='s', markersize=8, label='Val Accuracy')
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax2.set_xlabel('Training step')
ax2.set_ylabel('Score')
ax2.set_title(f'Validation metrics (best AUC = {ft_log["best_val_auc"]:.4f})')
ax2.legend(loc='lower right', frameon=False)
ax2.grid(alpha=0.3)
ax2.set_ylim(0.4, 0.6)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/fig_training_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] fig_training_curve.png")

print(f"\nAll figures saved to {RESULTS_DIR}/ at 300 DPI")