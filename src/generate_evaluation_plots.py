"""
Generate Publication-Ready Figures for Milestone 12 Evaluation Report.

Generates 5 clean, high-resolution figures saved in reports/figures/:
  1. fig1_baseline_comparison.png: 3-Way System Performance Comparison.
  2. fig2_intent_confusion_matrix.png: 10-Class Intent Confusion Heatmap.
  3. fig3_retrieval_performance.png: Historical Retrieval Hit@k and MRR.
  4. fig4_llm_judge_radar.png: 5-Dimension LLM-as-Judge Quality Radar Chart.
  5. fig5_escalation_confusion_matrix.png: 2x2 Routing Safety Confusion Matrix.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

HARNESS_JSON_PATH = "data/analysis/comprehensive_evaluation_harness.json"
FIGURES_DIR = "reports/figures"

# Set publication style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})

def generate_plots(harness_path: str = HARNESS_JSON_PATH, output_dir: str = FIGURES_DIR):
    if not os.path.exists(harness_path):
        raise FileNotFoundError(f"Evaluation harness data missing at {harness_path}. Run evaluate_harness.py first.")

    os.makedirs(output_dir, exist_ok=True)

    with open(harness_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------------------------------------------------------------
    # Figure 1: Baseline 1 vs Baseline 2 vs Final Hybrid Comparison
    # -------------------------------------------------------------
    print("Generating Figure 1: 3-Way Baseline Comparison...")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    systems = ["Baseline 1\n(Majority)", "Baseline 2\n(Classical ML)", "Final Hybrid\nSystem"]
    accuracies = [14.0, 90.5, data["dimension_1_intent_classification"]["accuracy"] * 100]
    macro_f1s = [2.46, 91.05, data["dimension_1_intent_classification"]["macro_f1"] * 100]
    weighted_f1s = [3.44, 90.43, data["dimension_1_intent_classification"]["weighted_f1"] * 100]

    x = np.arange(len(systems))
    width = 0.25

    rects1 = ax.bar(x - width, accuracies, width, label="Accuracy (%)", color="#4A90E2", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x, macro_f1s, width, label="Macro F1 (%)", color="#50E3C2", edgecolor="black", alpha=0.9)
    rects3 = ax.bar(x + width, weighted_f1s, width, label="Weighted F1 (%)", color="#B8E986", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("System Performance Comparison Across Milestones (Golden Set N=200)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", frameon=True)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_baseline_comparison.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"  -> Saved {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Normalized Intent Confusion Matrix
    # -------------------------------------------------------------
    print("Generating Figure 2: Intent Confusion Matrix Heatmap...")
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)

    cm = np.array(data["dimension_1_intent_classification"]["confusion_matrix"])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    intents = data["dimension_1_intent_classification"]["taxonomy_intents"]
    short_labels = [i.replace("_", " ").title() for i in intents]

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=short_labels, yticklabels=short_labels,
                cbar_kws={"label": "Normalized Proportion (Recall)"}, ax=ax)

    ax.set_xlabel("Predicted Intent", fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Intent", fontweight="bold", labelpad=10)
    ax.set_title("Intent Classification Normalized Confusion Matrix (N = 200)", fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_intent_confusion_matrix.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"  -> Saved {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Historical Retrieval Performance
    # -------------------------------------------------------------
    print("Generating Figure 3: Historical Retrieval Performance...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ret = data["dimension_2_retrieval_quality"]
    ret_labels = ["Hit@1", "Hit@3", "Hit@5", "MRR", "Intent Relevance"]
    ret_values = [
        ret["hit_at_1"] * 100,
        ret["hit_at_3"] * 100,
        ret["hit_at_5"] * 100,
        ret["mean_reciprocal_rank"] * 100,
        ret["top3_intent_relevance_rate"] * 100
    ]
    colors = ["#3498DB", "#2ECC71", "#1ABC9C", "#9B59B6", "#F39C12"]

    bars = ax.bar(ret_labels, ret_values, color=colors, edgecolor="black", width=0.55, alpha=0.9)
    ax.set_ylabel("Percentage (%) / MRR x 100", fontweight="bold")
    ax.set_title("Historical Precedent Retrieval Benchmarks (FAISS IndexFlatIP, N=200)", fontweight="bold", pad=15)
    ax.set_ylim(0, 105)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_retrieval_performance.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"  -> Saved {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: LLM-as-Judge Quality Radar Chart
    # -------------------------------------------------------------
    print("Generating Figure 4: LLM-as-Judge Radar Chart...")
    judge_scores = data["dimension_3_reply_quality_llm_judge"]["mean_scores"]

    categories = [
        "Correctness",
        "Historical\nGrounding",
        "Helpfulness",
        "Brand\nConsistency",
        "Safety &\nPolicy Integrity"
    ]
    raw_keys = [
        "correctness",
        "historical_grounding",
        "helpfulness",
        "brand_consistency",
        "safety_unsupported_claims"
    ]
    values = [judge_scores[k] for k in raw_keys]
    values += values[:1]  # Close radar loop

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)
    ax.fill(angles, values, color="#3498DB", alpha=0.35)
    ax.plot(angles, values, color="#2980B9", linewidth=2.5, marker="o", markersize=7)

    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1.0", "2.0", "3.0", "4.0", "5.0"], color="grey", size=9)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight="bold", size=10)
    ax.set_title(f"LLM-as-Judge Response Quality Scorecard\nOverall: {judge_scores['overall_score']:.2f} / 5.00",
                 fontweight="bold", pad=20)

    # Annotate points
    for angle, val in zip(angles[:-1], values[:-1]):
        ax.annotate(f"{val:.2f}", xy=(angle, val), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1A5276")

    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_llm_judge_radar.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"  -> Saved {fig4_path}")

    # -------------------------------------------------------------
    # Figure 5: Escalation Routing Confusion Matrix
    # -------------------------------------------------------------
    print("Generating Figure 5: Escalation Routing Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

    esc_cm = data["dimension_4_escalation_routing"]["confusion_matrix"]
    matrix_2x2 = np.array([
        [esc_cm["true_auto_handle"], esc_cm["false_escalate"]],
        [esc_cm["false_auto_handle"], esc_cm["true_escalate"]]
    ])

    group_names = [
        "True Auto-Handle\n(Safe Automation)",
        "False Escalate\n(Conservative Margin)",
        "False Auto-Handle\n(Critical Risk)",
        "True Escalate\n(Safety Guardrail)"
    ]
    group_counts = [f"{val}" for val in matrix_2x2.flatten()]
    group_percentages = [f"{val / matrix_2x2.sum() * 100:.1f}%" for val in matrix_2x2.flatten()]
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(matrix_2x2, annot=labels, fmt="", cmap="Purples", cbar=False, ax=ax,
                xticklabels=["Pred AUTO_HANDLE", "Pred ESCALATE"],
                yticklabels=["Actual AUTO_HANDLE", "Actual ESCALATE"])

    ax.set_title("Escalation Decision Policy Matrix (N = 200)", fontweight="bold", pad=15)
    ax.set_ylabel("Ground Truth Annotation", fontweight="bold")
    ax.set_xlabel("System Routing Decision", fontweight="bold")

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "fig5_escalation_confusion_matrix.png")
    plt.savefig(fig5_path)
    plt.close()
    print(f"  -> Saved {fig5_path}")

    print(f"\n[SUCCESS] All 5 publication-ready figures generated in {output_dir}")

def main():
    generate_plots()

if __name__ == "__main__":
    main()
