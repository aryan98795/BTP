#!/usr/bin/env python3
"""Train a Random Forest model to predict quantum efficiency (QE)."""

import argparse
import csv
import os
from pathlib import Path

# Keep Matplotlib's cache inside the project so it works in restricted shells.
PLOT_CACHE = Path(".matplotlib-cache")
PLOT_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE.resolve()))

import matplotlib

matplotlib.use("Agg")  # Save images without requiring a desktop display.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from train_qe_linear_regression import FEATURES, load_complete_rows


TARGET = "QE(%)"


def save_plots(actual, predicted, importances, output: Path):
    """Save actual-vs-predicted, residual, and feature-importance plots."""
    output.parent.mkdir(parents=True, exist_ok=True)
    residuals = predicted - actual
    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    axes[0].scatter(actual, predicted, s=16, alpha=0.55)
    axes[0].plot([minimum, maximum], [minimum, maximum], "r--", label="perfect prediction")
    axes[0].set(title="QE: actual vs predicted", xlabel="Actual QE (%)", ylabel="Predicted QE (%)")
    axes[0].legend()

    axes[1].scatter(predicted, residuals, s=16, alpha=0.55)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set(title="QE residuals", xlabel="Predicted QE (%)", ylabel="Prediction − actual")

    order = np.argsort(importances)
    axes[2].barh(np.asarray(FEATURES)[order], importances[order])
    axes[2].set(title="Random Forest feature importance", xlabel="Importance")

    fig.suptitle("Random Forest QE regression test results", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("1_merged.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("qe_random_forest_predictions.csv"))
    parser.add_argument("--importances", type=Path, default=Path("qe_random_forest_feature_importance.csv"))
    parser.add_argument("--plots", type=Path, default=Path("qe_random_forest_plots.png"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--trees", type=int, default=300, help="Number of trees in the forest.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    if args.trees < 1:
        parser.error("--trees must be at least 1.")

    x, y = load_complete_rows(args.input)
    qe = y[:, 0]  # The loader returns QE first, then photon energy.
    x_train, x_test, y_train, y_test = train_test_split(
        x, qe, test_size=args.test_size, random_state=args.seed
    )
    model = RandomForestRegressor(
        n_estimators=args.trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Complete rows used: {len(x)}")
    print(f"Training rows: {len(x_train)} | Test rows: {len(x_test)}")
    print(f"{TARGET}: MAE = {mae:.6g}, MSE = {mse:.6g}, R^2 = {r2:.6f}")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual_QE(%)", "predicted_QE(%)", "residual"])
        writer.writerows(zip(y_test, predictions, predictions - y_test))

    args.importances.parent.mkdir(parents=True, exist_ok=True)
    with args.importances.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "importance"])
        writer.writerows(zip(FEATURES, model.feature_importances_))

    save_plots(y_test, predictions, model.feature_importances_, args.plots)
    print(f"Predictions written to: {args.predictions}")
    print(f"Feature importances written to: {args.importances}")
    print(f"Graphs written to: {args.plots}")


if __name__ == "__main__":
    main()
