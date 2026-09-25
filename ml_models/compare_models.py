#!/usr/bin/env python3
"""Train separate regressors on SCAPS runs and create a shared comparison."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import MODELS, OUTPUT_ROOT, SEED, load_dataset, run_one, write_dataset


def main():
    frame = load_dataset()
    data_dir = OUTPUT_ROOT / "data"
    write_dataset(frame, data_dir / "scaps_efficiency_dataset.csv")
    summary = {
        "runs_parsed": len(frame),
        "source_files": ", ".join(sorted(frame["source_file"].unique())),
        "target": "eta(%)",
        "features": ", ".join(col for col in frame.columns if col not in {
            "source_file", "simulation_step", "group_id", "eta(%)"
        }),
        "seed": SEED,
        "split": "80/20 grouped by identical parameter configuration",
    }
    pd.DataFrame([summary]).to_csv(data_dir / "dataset_summary.csv", index=False)

    rows = []
    for name in MODELS:
        _, metrics = run_one(name)
        rows.append(metrics)
        print(f"{name}: MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | R2={metrics['R2']:.4f}")

    comparison_dir = OUTPUT_ROOT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows).sort_values("MAE")
    results.to_csv(comparison_dir / "model_metrics.csv", index=False)
    best = results.iloc[0]
    (comparison_dir / "summary.txt").write_text(
        f"Best by MAE: {best['model']} (MAE {best['MAE']:.4f} percentage points)\n"
        f"Runs: {len(frame)} | Train/test split: 80/20 grouped by identical parameter configuration\n"
        "Interpret metrics cautiously: current data are small SCAPS parameter sweeps; "
        "test rows can be nearby settings from the same sweep.\n"
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    ordered = results.sort_values("MAE", ascending=True)
    axes[0].barh(ordered["model"], ordered["MAE"], color="#4c78a8")
    axes[0].set(title="Absolute error (lower is better)", xlabel="MAE (eta percentage points)")
    axes[1].barh(ordered["model"], ordered["RMSE"], color="#f28e2b")
    axes[1].set(title="RMSE (lower is better)", xlabel="RMSE (eta percentage points)")
    axes[2].barh(ordered["model"], ordered["R2"], color="#59a14f")
    axes[2].axvline(0, color="black", linewidth=.8)
    axes[2].set(title="R² (higher is better)", xlabel="R²")
    fig.suptitle("SCAPS eta model comparison — shared grouped holdout")
    fig.savefig(comparison_dir / "metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for _, row in ordered.iterrows():
        pred = pd.read_csv(OUTPUT_ROOT / row["model"] / "predictions.csv")
        ax.scatter(pred["actual_eta(%)"], pred["error"], alpha=.65, label=row["model"])
    ax.axhline(0, color="black", linestyle="--")
    ax.set(title="Test residuals by model", xlabel="Actual eta (%)", ylabel="Prediction error (percentage points)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(comparison_dir / "residual_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Model artifacts: {OUTPUT_ROOT}")
    print(f"Comparison table and charts: {comparison_dir}")


if __name__ == "__main__":
    main()
