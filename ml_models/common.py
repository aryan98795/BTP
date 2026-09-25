"""Shared SCAPS IV parsing, model evaluation, and artifact generation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT
OUTPUT_ROOT = Path(__file__).resolve().parent
SEED = 42

MODELS = {
    "linear_regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "ridge_regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1
    ),
    "extra_trees": lambda: ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.04, max_depth=2, loss="huber", random_state=SEED
    ),
    "support_vector_regression": lambda: make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1)),
    "knn_regression": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")),
}

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
BLOCK_START = re.compile(r"(?=Batch simulation #\s*\d+\s+step\s+\d+\s+of\s+\d+)")
PARAMETER = re.compile(r">>\s*(.+?)\s*:\s*(" + FLOAT + r")")
ETA = re.compile(r"^\s*eta\s*=\s*(" + FLOAT + r")\s*%", re.MULTILINE | re.IGNORECASE)
TABLE_HEADER = re.compile(r"^\s*v\(V\)\s+jtot\(mA/cm2\).*j_Auger\(mA/cm2\)", re.MULTILINE)


def parse_iv_file(path: Path) -> list[dict]:
    """Parse each SCAPS batch step as one sample with parameter features and eta target."""
    text = path.read_text(encoding="latin-1")
    starts = [match.start() for match in BLOCK_START.finditer(text)]
    records = []
    for index, start in enumerate(starts):
        block = text[start:starts[index + 1] if index + 1 < len(starts) else len(text)]
        eta_match = ETA.search(block)
        header_match = TABLE_HEADER.search(block)
        if not eta_match or not header_match:
            continue
        parameters = {}
        for match in PARAMETER.finditer(block[:header_match.start()]):
            name = re.sub(r"\s+", " ", match.group(1).strip())
            parameters[name] = float(match.group(2))
        if not parameters:
            continue

        # Hash the explicit parameter tuple so duplicated configurations in
        # overlapping SCAPS sweeps remain together in the train/test split.
        key = json.dumps(sorted(parameters.items()), separators=(",", ":"))
        records.append({
            **parameters,
            "source_file": path.name,
            "simulation_step": len(records) + 1,
            "group_id": hashlib.sha1(key.encode()).hexdigest(),
            "eta(%)": float(eta_match.group(1)),
        })
    return records


def clean_iv_file(path: Path, output: Path) -> pd.DataFrame:
    """Export measurement rows from one SCAPS report to a clean CSV table."""
    path = Path(path)
    text = path.read_text(encoding="latin-1")
    starts = [match.start() for match in BLOCK_START.finditer(text)]
    records = []
    parameter_names = set()
    iv_columns = ["v(V)", "jtot(mA/cm2)", "j_total_rec(mA/cm2)",
                  "j_total_gen(mA/cm2)", "jbulk(mA/cm2)", "jifr(mA/cm2)",
                  "jminor_left(mA/cm2)", "jminor_right(mA/cm2)",
                  "j_SRH(mA/cm2)", "j_Radiative(mA/cm2)", "j_Auger(mA/cm2)"]
    summary_patterns = {
        name: re.compile(r"^\s*" + name + r"\s*=\s*(" + FLOAT + r")", re.MULTILINE | re.IGNORECASE)
        for name in ("Voc", "Jsc", "FF", "eta", "V_MPP", "J_MPP")
    }
    for run_number, start in enumerate(starts, 1):
        block = text[start:starts[run_number] if run_number < len(starts) else len(text)]
        header_match = TABLE_HEADER.search(block)
        eta_match = ETA.search(block)
        if not header_match or not eta_match:
            continue
        parameters = {}
        for match in PARAMETER.finditer(block[:header_match.start()]):
            name = re.sub(r"\s+", " ", match.group(1).strip())
            parameters[name] = float(match.group(2))
        if not parameters:
            continue
        parameter_names.update(parameters)
        summary = {name: float(match.group(1)) for name, pattern in summary_patterns.items()
                   if (match := pattern.search(block))}
        scan_point = 0
        for line in block[header_match.end():].splitlines():
            values = line.split()
            if len(values) != len(iv_columns):
                continue
            try:
                measurements = [float(value) for value in values]
            except ValueError:
                continue
            scan_point += 1
            records.append({
                "source_file": path.name,
                "simulation_id": run_number,
                "scan_point": scan_point,
                **parameters,
                **dict(zip(iv_columns, measurements)),
                **summary,
            })
    if not records:
        raise ValueError(f"No clean I-V measurement rows could be parsed from {path.name}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["source_file", "simulation_id", "scan_point", *sorted(parameter_names),
               *iv_columns, "Voc", "Jsc", "FF", "eta", "V_MPP", "J_MPP"]
    cleaned = pd.DataFrame(records, columns=columns)
    cleaned.to_csv(output, index=False)
    return cleaned


def load_dataset(input_dir: Path = DATA_DIR) -> pd.DataFrame:
    input_dir = Path(input_dir)
    if input_dir.is_file() and input_dir.suffix.lower() == ".csv":
        cleaned = pd.read_csv(input_dir)
        fixed_columns = {"source_file", "simulation_id", "scan_point", "v(V)",
                         "jtot(mA/cm2)", "j_total_rec(mA/cm2)", "j_total_gen(mA/cm2)",
                         "jbulk(mA/cm2)", "jifr(mA/cm2)", "jminor_left(mA/cm2)",
                         "jminor_right(mA/cm2)", "j_SRH(mA/cm2)", "j_Radiative(mA/cm2)",
                         "j_Auger(mA/cm2)", "Voc", "Jsc", "FF", "eta", "V_MPP", "J_MPP"}
        parameter_columns = [column for column in cleaned.columns if column not in fixed_columns]
        if "simulation_id" not in cleaned or "eta" not in cleaned:
            raise ValueError(f"{input_dir} is not a cleaned SCAPS measurement CSV")
        frame = cleaned.groupby("simulation_id", sort=False).first().reset_index()
        frame = frame.rename(columns={"eta": "eta(%)", "simulation_id": "simulation_step"})
        parameters = frame[parameter_columns].apply(pd.to_numeric, errors="coerce")
        frame = pd.concat([frame[["source_file", "simulation_step", "eta(%)"]], parameters], axis=1)
        group_ids = []
        for row in parameters.to_dict(orient="records"):
            key = json.dumps(sorted((name, value) for name, value in row.items() if pd.notna(value)), separators=(",", ":"))
            group_ids.append(hashlib.sha1(key.encode()).hexdigest())
        frame["group_id"] = group_ids
        return frame.drop_duplicates("group_id", keep="first").reset_index(drop=True)

    files = [input_dir] if input_dir.is_file() and input_dir.suffix.lower() == ".iv" else sorted(input_dir.glob("*.iv"))
    if not files:
        raise FileNotFoundError(f"No .iv files found in {input_dir}")
    records = [row for path in files for row in parse_iv_file(path)]
    if len(records) < 7:
        raise ValueError(f"Only {len(records)} valid SCAPS runs were parsed from {input_dir}; at least 7 are needed.")
    frame = pd.DataFrame(records)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["eta(%)"])
    # Repeated settings in overlapping sweeps are one experimental condition;
    # retain the first copy to avoid duplicate configurations inflating scores.
    frame = frame.drop_duplicates("group_id", keep="first").reset_index(drop=True)
    return frame


def split_data(frame: pd.DataFrame):
    feature_columns = [col for col in frame.columns if col not in {
        "source_file", "simulation_step", "group_id", "eta(%)"
    }]
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = frame["eta(%)"].to_numpy()
    groups = frame["group_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train, test = next(splitter.split(x, y, groups))
    if len(train) < 5 or len(test) < 2:
        raise ValueError("Not enough distinct parameter configurations for an 80/20 split.")
    return x, y, train, test, feature_columns


def calculate_metrics(actual, predicted) -> dict:
    error = predicted - actual
    denominator = np.maximum(np.abs(actual), 1e-9)
    smape_den = np.abs(actual) + np.abs(predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
        "MedianAE": median_absolute_error(actual, predicted),
        "MaxAE": np.max(np.abs(error)),
        "MAPE_pct": np.mean(np.abs(error) / denominator) * 100,
        "sMAPE_pct": np.mean(2 * np.abs(error) / np.maximum(smape_den, 1e-9)) * 100,
        "ExplainedVariance": explained_variance_score(actual, predicted),
    }


def save_model_artifacts(model_name: str, model, frame: pd.DataFrame,
                         x, y, train, test, features: list[str], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    # Imputation is fitted on training data only; the model's own preprocessing
    # (scaling for scale-sensitive estimators) remains in the estimator pipeline.
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x_train = imputer.fit_transform(x.iloc[train])
    x_test = imputer.transform(x.iloc[test])
    model.fit(x_train, y[train])
    predicted = model.predict(x_test)
    actual = y[test]
    metrics = calculate_metrics(actual, predicted)
    metrics.update({"model": model_name, "train_rows": len(train), "test_rows": len(test)})

    predictions = pd.DataFrame({
        "actual_eta(%)": actual,
        "predicted_eta(%)": predicted,
        "error": predicted - actual,
        "absolute_error": np.abs(predicted - actual),
        "source_file": frame.iloc[test]["source_file"].to_numpy(),
    })
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    axes[0, 0].scatter(actual, predicted, alpha=.75)
    axes[0, 0].plot([lo, hi], [lo, hi], "r--")
    axes[0, 0].set(title="Actual vs predicted efficiency", xlabel="Actual eta (%)", ylabel="Predicted eta (%)")
    residual = predicted - actual
    axes[0, 1].scatter(predicted, residual, alpha=.75)
    axes[0, 1].axhline(0, color="red", linestyle="--")
    axes[0, 1].set(title="Residuals vs prediction", xlabel="Predicted eta (%)", ylabel="Error (predicted − actual)")
    axes[1, 0].hist(residual, bins="auto", edgecolor="white")
    axes[1, 0].axvline(0, color="red", linestyle="--")
    axes[1, 0].set(title="Prediction error distribution", xlabel="Error (percentage points)", ylabel="Count")
    order = np.argsort(np.abs(residual))
    axes[1, 1].bar(np.arange(len(test)), np.abs(residual[order]))
    axes[1, 1].set(title="Absolute error by test sample", xlabel="Test sample (sorted by error)", ylabel="Absolute error (percentage points)")
    fig.suptitle(model_name.replace("_", " ").title() + " — SCAPS eta prediction")
    fig.savefig(output / "error_analysis.png", dpi=170)
    plt.close(fig)

    # Permutation importance allows consistent interpretation for any estimator.
    importance = permutation_importance(model, x_test, actual, n_repeats=12,
                                        random_state=SEED, scoring="neg_mean_absolute_error")
    missing_features = [features[i] for i in imputer.indicator_.features_]
    names = features + [f"missing({name})" for name in missing_features]
    importance_frame = pd.DataFrame({"feature": names, "importance_mean": importance.importances_mean,
                                     "importance_std": importance.importances_std}).sort_values("importance_mean")
    importance_frame.to_csv(output / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, max(4, .45 * len(importance_frame))))
    ax.barh(importance_frame["feature"], importance_frame["importance_mean"],
            xerr=importance_frame["importance_std"], alpha=.85)
    ax.set(title="Permutation importance (higher = more useful)", xlabel="MAE increase when shuffled")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=170)
    plt.close(fig)
    return metrics


def run_one(model_name: str, input_dir: Path = DATA_DIR, output_dir: Path | None = None):
    frame = load_dataset(input_dir)
    x, y, train, test, features = split_data(frame)
    model_dir = output_dir or OUTPUT_ROOT / model_name
    metrics = save_model_artifacts(model_name, MODELS[model_name](), frame,
                                   x, y, train, test, features, model_dir)
    return frame, metrics


def write_dataset(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["group_id"]).to_csv(path, index=False)
