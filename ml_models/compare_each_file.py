#!/usr/bin/env python3
"""Run all regressors independently for each root-level SCAPS .iv file."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import MODELS, ROOT, clean_iv_file, load_dataset, run_one, write_dataset


RESULTS_ROOT = ROOT / "results"


def save_file_comparison(source_file: str, rows: list[dict], output: Path):
    results = pd.DataFrame(rows).sort_values("MAE")
    results.to_csv(output / "model_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for axis, metric, title, color in [
        (axes[0], "MAE", "MAE (lower is better)", "#4c78a8"),
        (axes[1], "RMSE", "RMSE (lower is better)", "#f28e2b"),
        (axes[2], "R2", "R² (higher is better)", "#59a14f"),
    ]:
        axis.barh(results["model"], results[metric], color=color)
        axis.set(title=title, xlabel=metric)
        if metric == "R2":
            axis.axvline(0, color="black", linewidth=.8)
    fig.suptitle(f"Model comparison for {source_file} — eta prediction")
    fig.savefig(output / "metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for _, row in results.iterrows():
        predictions = pd.read_csv(output / row["model"] / "predictions.csv")
        ax.scatter(predictions["actual_eta(%)"], predictions["error"],
                   alpha=.75, label=row["model"])
    ax.axhline(0, color="black", linestyle="--")
    ax.set(title=f"Test residuals for {source_file}", xlabel="Actual eta (%)",
           ylabel="Prediction error (percentage points)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output / "residual_comparison.png", dpi=180)
    plt.close(fig)
    return results


def main():
    iv_files = sorted(ROOT.glob("*.iv"))
    if not iv_files:
        raise FileNotFoundError(f"No .iv files found in {ROOT}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []
    source_summaries = []
    for iv_file in iv_files:
        source_dir = RESULTS_ROOT / iv_file.stem
        source_dir.mkdir(parents=True, exist_ok=True)
        cleaned_csv = source_dir / f"{iv_file.stem}_cleaned.csv"
        clean_iv_file(iv_file, cleaned_csv)
        frame = load_dataset(cleaned_csv)
        write_dataset(frame, source_dir / "dataset.csv")
        rows = []
        for name in MODELS:
            _, metrics = run_one(name, input_dir=cleaned_csv,
                                 output_dir=source_dir / name)
            metrics["source_file"] = iv_file.name
            rows.append(metrics)
        comparison = save_file_comparison(iv_file.name, rows, source_dir)
        all_results.extend(comparison.to_dict(orient="records"))
        best = comparison.iloc[0]
        source_summaries.append({
            "source_file": iv_file.name,
            "unique_runs": len(frame),
            "best_model_by_MAE": best["model"],
            "best_MAE": best["MAE"],
            "result_folder": str(source_dir.relative_to(ROOT)),
        })
        (source_dir / "README.txt").write_text(
            f"Source file: {iv_file.name}\n"
            f"Clean CSV: {cleaned_csv.name} (one row per I-V scan point)\n"
            f"Unique simulation configurations: {len(frame)}\n"
            "Target: eta (%) from the SCAPS per-run summary.\n"
            "All models use the same deterministic 80/20 grouped holdout for this file.\n"
            "Scores from small sweeps are estimates; nearby parameter settings can occur in both splits.\n"
        )
        print(f"{iv_file.name}: {len(frame)} runs | best MAE model={best['model']} ({best['MAE']:.4f})")

    pd.DataFrame(all_results).to_csv(RESULTS_ROOT / "model_metrics_by_file.csv", index=False)
    pd.DataFrame(source_summaries).to_csv(RESULTS_ROOT / "source_file_summary.csv", index=False)
    (RESULTS_ROOT / "README.md").write_text(
        "# Per-file SCAPS results\n\n"
        "Every root-level `.iv` file is exported as a clean CSV with one row per I-V scan point, then evaluated independently. "
        "Open the matching source-name folder to see `<source>_cleaned.csv`, the per-simulation `dataset.csv`, "
        "per-model metrics and predictions, model error graphs, and comparison charts.\n\n"
        "`model_metrics_by_file.csv` lists each file/model combination, while "
        "`source_file_summary.csv` lists the best model by MAE for each file.\n\n"
        "The target is per-run SCAPS solar-cell efficiency (`eta (%)`), not QE versus wavelength. "
        "Each file has its own grouped 80/20 split. Small sweeps can produce unstable metrics; "
        "the estimates do not establish performance on new materials or parameter regimes.\n\n"
        "Original `.iv` files remain in the project as source inputs; generated data and model outputs here are CSV/PNG/TXT.\n"
    )
    print(f"Per-file results saved under {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
