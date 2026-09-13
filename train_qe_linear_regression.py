#!/usr/bin/env python3
"""Train a baseline linear-regression model for the merged SCAPS data."""

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


IV_FEATURES = [
    "v(V)", "jtot(mA/cm2)", "j_total_rec(mA/cm2)",
    "j_total_gen(mA/cm2)", "jbulk(mA/cm2)", "jifr(mA/cm2)",
    "jminor_left(mA/cm2)", "jminor_right(mA/cm2)", "j_SRH(mA/cm2)",
    "j_Radiative(mA/cm2)", "j_Auger(mA/cm2)",
]
FEATURES = IV_FEATURES + ["lambda(nm)"]
TARGETS = ["QE(%)", "photon energy (eV)"]


def load_complete_rows(path: Path):
    """Return complete, physically plausible rows for the baseline model."""
    path = Path(path)
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                values = [float(row[column]) for column in FEATURES + TARGETS]
            except (KeyError, TypeError, ValueError):
                continue
            wavelength, qe, energy = values[-3:]
            # The QE scan in this data set spans 300--900 nm.  This avoids
            # report values that happened to be parsed as three numeric fields.
            if not (300.0 <= wavelength <= 900.0 and 0.0 <= qe <= 100.0 and 1.0 <= energy <= 5.0):
                continue
            rows.append(values)

    if len(rows) < 10:
        raise ValueError("Need at least 10 complete rows to train and test the model.")

    data = np.asarray(rows, dtype=float)
    return data[:, :len(FEATURES)], data[:, len(FEATURES):]


def save_plots(actual, predicted, output: Path):
    """Create actual-vs-predicted and residual plots for both targets."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    for column, target in enumerate(TARGETS):
        actual_values = actual[:, column]
        predicted_values = predicted[:, column]
        residuals = predicted_values - actual_values

        minimum = min(actual_values.min(), predicted_values.min())
        maximum = max(actual_values.max(), predicted_values.max())
        axes[0, column].scatter(actual_values, predicted_values, s=16, alpha=0.55)
        axes[0, column].plot([minimum, maximum], [minimum, maximum], "r--", label="perfect prediction")
        axes[0, column].set(title=f"{target}: actual vs predicted", xlabel=f"Actual {target}", ylabel=f"Predicted {target}")
        axes[0, column].legend()

        axes[1, column].scatter(predicted_values, residuals, s=16, alpha=0.55)
        axes[1, column].axhline(0, color="r", linestyle="--")
        axes[1, column].set(title=f"{target}: residuals", xlabel=f"Predicted {target}", ylabel="Prediction − actual")

    fig.suptitle("Linear-regression test results", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("1_merged.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("qe_linear_predictions.csv"))
    parser.add_argument("--plots", type=Path, default=Path("qe_linear_regression_plots.png"))
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    x, y = load_complete_rows(args.input)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=42
    )
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Complete rows used: {len(x)}")
    print(f"Training rows: {len(x_train)} | Test rows: {len(x_test)}")
    for index, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_test[:, index], predictions[:, index])
        r2 = r2_score(y_test[:, index], predictions[:, index])
        print(f"{target}: MAE = {mae:.6g}, R^2 = {r2:.6f}")

    with args.predictions.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "actual_QE(%)", "predicted_QE(%)",
            "actual_photon_energy(eV)", "predicted_photon_energy(eV)",
        ])
        writer.writerows(np.column_stack((
            y_test[:, 0], predictions[:, 0], y_test[:, 1], predictions[:, 1]
        )))
    save_plots(y_test, predictions, args.plots)
    print(f"Predictions written to: {args.predictions}")
    print(f"Graphs written to: {args.plots}")


if __name__ == "__main__":
    main()
