"""
Milestone 13: Publication-Grade Plots for Human vs. LLM Judge Validation.
Generates:
- fig6_human_vs_llm_correlation.png: Scatter plots with regression lines for all dimensions
- fig7_judge_bias_analysis.png: Mean bias deltas and agreement rate breakdowns
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = "data/analysis/human_vs_llm_validation_results.json"
FIG_DIR = "reports/figures"
BRAIN_DIR = r"C:\Users\prakh\.gemini\antigravity\brain\8930736f-8012-48b4-9963-c24696218ff6"

os.makedirs(FIG_DIR, exist_ok=True)

# Set global publication styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def load_data():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_figure_6(data):
    """Figure 6: Multi-panel scatter plot with regression and jitter."""
    pairs = data["matched_evaluation_pairs"]
    dims = [
        ("correctness", "Correctness", "#1f77b4"),
        ("historical_grounding", "Historical Grounding", "#ff7f0e"),
        ("helpfulness", "Helpfulness", "#2ca02c"),
        ("brand_consistency", "Brand Consistency", "#9467bd"),
        ("safety_unsupported_claims", "Safety & Unsupported Claims", "#d62728"),
        ("overall_score", "Overall Composite Score", "#333333")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    np.random.seed(42)

    for idx, (dim_key, dim_title, color) in enumerate(dims):
        ax = axes[idx]
        dim_metrics = data["results_by_dimension"][dim_key]

        x_hum = np.array([p["human_scorecard"][dim_key] for p in pairs])
        y_llm = np.array([p["llm_scorecard"][dim_key] for p in pairs])

        # Add slight jitter for visual clarity on integer scores
        jitter_x = x_hum + np.random.normal(0, 0.05, size=len(x_hum)) if dim_key != "overall_score" else x_hum
        jitter_y = y_llm + np.random.normal(0, 0.05, size=len(y_llm)) if dim_key != "overall_score" else y_llm

        # Scatter points
        ax.scatter(jitter_x, jitter_y, color=color, alpha=0.75, s=65, edgecolor="black", linewidth=0.7, zorder=3)

        # Regression line
        if np.std(x_hum) > 0 and np.std(y_llm) > 0:
            m, b = np.polyfit(x_hum, y_llm, 1)
            x_seq = np.linspace(0.8, 5.2, 100)
            ax.plot(x_seq, m * x_seq + b, color=color, linestyle="-", linewidth=2.2, label=f"Fit (slope={m:.2f})")

        # Identity line (y = x)
        ax.plot([0.5, 5.5], [0.5, 5.5], color="#888888", linestyle="--", linewidth=1.2, alpha=0.8, label="Ideal (y = x)")

        # Title & Annotations
        r_val = dim_metrics["pearson_r"]
        rho_val = dim_metrics["spearman_rho"]
        mae_val = dim_metrics["mae"]
        ax.set_title(f"{dim_title}\n(r = {r_val:.3f}, ρ = {rho_val:.3f}, MAE = {mae_val:.2f})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Human Expert Score (1–5)", fontsize=10)
        ax.set_ylabel("LLM Judge Score (1–5)", fontsize=10)
        ax.set_xlim(0.8, 5.3)
        ax.set_ylim(0.8, 5.3)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Figure 6: Human Expert vs. LLM-as-a-Judge Correlation Benchmark (N = 20)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig6_human_vs_llm_correlation.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 6 to {out_path}")

def generate_figure_7(data):
    """Figure 7: Systematic Judge Bias, Mean Deltas, and Agreement Breakdown."""
    dims = [
        ("correctness", "Correctness"),
        ("historical_grounding", "Historical Grounding"),
        ("helpfulness", "Helpfulness"),
        ("brand_consistency", "Brand Consistency"),
        ("safety_unsupported_claims", "Safety / Unsupported"),
        ("overall_score", "Overall Composite")
    ]

    names = [d[1] for d in dims]
    bias_deltas = [data["results_by_dimension"][d[0]]["mean_bias_delta"] for d in dims]
    exact_rates = [data["results_by_dimension"][d[0]]["exact_agreement_rate"] * 100 for d in dims]
    adjacent_rates = [data["results_by_dimension"][d[0]]["adjacent_agreement_rate"] * 100 for d in dims]
    qwk_scores = [data["results_by_dimension"][d[0]]["quadratic_weighted_kappa"] for d in dims]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Systematic Bias Deltas (LLM - Human)
    bar_colors = ["#d62728" if delta < -0.2 else "#2ca02c" if delta > 0.2 else "#1f77b4" for delta in bias_deltas]
    y_pos = np.arange(len(names))
    bars = ax1.barh(y_pos, bias_deltas, color=bar_colors, height=0.55, edgecolor="black", linewidth=0.8)
    ax1.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10, fontweight="semibold")
    ax1.set_xlabel("Mean Rating Bias (LLM Score − Human Score)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Directional Bias: Harshness vs. Leniency", fontsize=12, fontweight="bold")
    ax1.set_xlim(-1.2, 0.6)

    for bar, delta in zip(bars, bias_deltas):
        x_val = bar.get_width()
        offset = 0.03 if x_val >= 0 else -0.05
        ha = "left" if x_val >= 0 else "right"
        label = f"{delta:+.2f}"
        ax1.text(x_val + offset, bar.get_y() + bar.get_height() / 2.0, label, va="center", ha=ha, fontsize=9.5, fontweight="bold")

    # Add text labels indicating harshness vs leniency
    ax1.text(-0.85, 4.8, "← LLM Harshness Bias", color="#d62728", fontsize=10, fontweight="bold")
    ax1.text(0.15, 4.8, "LLM Leniency Bias →", color="#2ca02c", fontsize=10, fontweight="bold")

    # Panel 2: Agreement Rates (Exact vs Adjacent ±1)
    bar_width = 0.35
    ax2.bar(y_pos - bar_width/2, exact_rates, width=bar_width, label="Exact Agreement (%)", color="#4363d8", edgecolor="black", linewidth=0.8)
    ax2.bar(y_pos + bar_width/2, adjacent_rates, width=bar_width, label="Adjacent (±1 Point) Agreement (%)", color="#3cb44b", edgecolor="black", linewidth=0.8)

    ax2.set_xticks(y_pos)
    ax2.set_xticklabels([d[1].replace(" ", "\n") for d in dims], fontsize=9)
    ax2.set_ylabel("Agreement Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Inter-Rater Concordance Rates", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.axhline(80, color="#e6194b", linestyle=":", linewidth=1.2, label="High Reliability Target (80%)")
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    for idx, (ex, adj) in enumerate(zip(exact_rates, adjacent_rates)):
        ax2.text(idx - bar_width/2, ex + 2, f"{ex:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax2.text(idx + bar_width/2, adj + 2, f"{adj:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("Figure 7: LLM Judge Calibration, Bias Profiles, and Agreement Concordance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig7_judge_bias_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 7 to {out_path}")

def copy_to_brain():
    import shutil
    for fig_name in ["fig6_human_vs_llm_correlation.png", "fig7_judge_bias_analysis.png"]:
        src = os.path.join(FIG_DIR, fig_name)
        dst = os.path.join(BRAIN_DIR, fig_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"[OK] Copied {fig_name} to brain directory: {dst}")

def main():
    data = load_data()
    generate_figure_6(data)
    generate_figure_7(data)
    copy_to_brain()

if __name__ == "__main__":
    main()
