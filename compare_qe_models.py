#!/usr/bin/env python3
"""Compare Linear Regression and Random Forest on QE prediction."""

import argparse
import csv
import os
from pathlib import Path

PLOT_CACHE = Path(".matplotlib-cache")
PLOT_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_qe_linear_regression import load_complete_rows


def model_metrics(actual, predicted):
    """Return comparable QE regression metrics."""
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def save_plot(actual, predictions, output):
    """Save actual-vs-predicted plots and a compact metric comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    all_predictions = np.concatenate(list(predictions.values()))
    low, high = min(actual.min(), all_predictions.min()), max(actual.max(), all_predictions.max())

    for axis, (name, predicted) in zip(axes[:2], predictions.items()):
        axis.scatter(actual, predicted, s=14, alpha=0.5)
        axis.plot([low, high], [low, high], "r--", label="perfect prediction")
        axis.set(title=name, xlabel="Actual QE (%)", ylabel="Predicted QE (%)")
        axis.legend()

    names = list(predictions)
    maes = [mean_absolute_error(actual, predictions[name]) for name in names]
    axis = axes[2]
    bars = axis.bar(names, maes, color=["#4c78a8", "#59a14f"])
    axis.set(title="Lower is better", ylabel="QE mean absolute error (%)")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_ylim(0, max(maes) * 1.15)

    fig.suptitle("QE model comparison — identical 80/20 test split", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("1_merged.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("qe_model_comparison.csv"))
    parser.add_argument("--plot", type=Path, default=Path("qe_model_comparison.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x, targets = load_complete_rows(args.input)
    qe = targets[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, qe, test_size=0.20, random_state=args.seed)
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=1.0,
            random_state=args.seed, n_jobs=-1,
        ),
    }
    predictions = {}
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions[name] = model.predict(x_test)
        rows.append({"model": name, **model_metrics(y_test, predictions[name])})

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "MAE", "MSE", "RMSE", "R2"])
        writer.writeheader()
        writer.writerows(rows)
    save_plot(y_test, predictions, args.plot)

    print(f"Rows: {len(x)} | Training: {len(x_train)} | Test: {len(x_test)}")
    for row in rows:
        print(f"{row['model']}: MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R^2={row['R2']:.4f}")
    print(f"Metrics written to: {args.metrics}")
    print(f"Chart written to: {args.plot}")


if __name__ == "__main__":
    main()
