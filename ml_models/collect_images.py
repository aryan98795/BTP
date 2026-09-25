#!/usr/bin/env python3
"""Copy generated model charts into a single source/model hierarchy."""

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DESTINATION = RESULTS / "images"
MODEL_NAMES = (
    "linear_regression", "ridge_regression", "random_forest", "extra_trees",
    "gradient_boosting", "support_vector_regression", "knn_regression",
)


def copy_image(source: Path, target: Path, source_label: str, model: str, chart: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source_file": source_label,
        "model": model,
        "chart": chart,
        "image_path": target.relative_to(ROOT).as_posix(),
    }


def main():
    entries = []
    # Per-file results: one comparison folder, then a folder for each model.
    for source_dir in sorted(RESULTS.iterdir()):
        if not source_dir.is_dir() or source_dir.name == "images":
            continue
        if not (source_dir / "model_metrics.csv").is_file():
            continue
        label = f"{source_dir.name}.iv"
        for chart, filename in (
            ("metric_comparison", "metric_comparison.png"),
            ("residual_comparison", "residual_comparison.png"),
        ):
            source = source_dir / filename
            if source.is_file():
                entries.append(copy_image(
                    source, DESTINATION / source_dir.name / "comparison" / filename,
                    label, "all_models", chart,
                ))
        for model in MODEL_NAMES:
            model_dir = source_dir / model
            for chart, filename in (
                ("error_analysis", "error_analysis.png"),
                ("feature_importance", "feature_importance.png"),
            ):
                source = model_dir / filename
                if source.is_file():
                    entries.append(copy_image(
                        source, DESTINATION / source_dir.name / "models" / model / filename,
                        label, model, chart,
                    ))

    # Combined-data model charts live under ml_models/ rather than results/.
    combined = DESTINATION / "all_files_combined"
    for chart, source in (
        ("metric_comparison", ROOT / "ml_models/comparison/metric_comparison.png"),
        ("residual_comparison", ROOT / "ml_models/comparison/residual_comparison.png"),
    ):
        if source.is_file():
            entries.append(copy_image(
                source, combined / "comparison" / source.name,
                "all_files_combined", "all_models", chart,
            ))
    for model in MODEL_NAMES:
        for chart, filename in (
            ("error_analysis", "error_analysis.png"),
            ("feature_importance", "feature_importance.png"),
        ):
            source = ROOT / "ml_models" / model / filename
            if source.is_file():
                entries.append(copy_image(
                    source, combined / "models" / model / filename,
                    "all_files_combined", model, chart,
                ))

    # Earlier QE workflow outputs are retained in their own clearly labeled area.
    legacy_images = {
        "qe_linear_regression_plots.png": "linear_regression_results",
        "qe_random_forest_plots.png": "random_forest_results",
        "qe_model_comparison.png": "qe_model_comparison",
    }
    for filename, chart in legacy_images.items():
        source = ROOT / filename
        if source.is_file():
            entries.append(copy_image(
                source, DESTINATION / "legacy_qe_workflow" / filename,
                "legacy_qe_workflow", "legacy", chart,
            ))

    DESTINATION.mkdir(parents=True, exist_ok=True)
    index_path = DESTINATION / "image_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_file", "model", "chart", "image_path"])
        writer.writeheader()
        writer.writerows(entries)
    (DESTINATION / "README.md").write_text(
        "# Collected model images\n\n"
        "This folder contains copies of all generated model charts, organized by source file, model, and chart type.\n\n"
        "- `<source>/comparison/`: model-metric and residual comparisons for one `.iv` source.\n"
        "- `<source>/models/<model>/`: that model's error-analysis and feature-importance plots.\n"
        "- `all_files_combined/`: charts from the combined-data comparison.\n"
        "- `legacy_qe_workflow/`: charts from the earlier QE scripts.\n\n"
        "`image_index.csv` maps each image to its source, model, and chart type. These are copies; original charts remain in their existing folders.\n"
    )
    print(f"Collected {len(entries)} images under {DESTINATION}")
    print(f"Image index: {index_path}")


if __name__ == "__main__":
    main()
