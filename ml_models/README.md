# SCAPS model comparison

Run from the repository root:

```bash
./.venv/bin/python ml_models/compare_models.py
```

The command reads the current root-level `.iv` files, parses each SCAPS batch
step into one simulation sample, predicts **solar-cell efficiency (`eta (%)`)**
from the batch parameter values, removes repeated parameter configurations,
and uses the same 80/20 grouped holdout for each model. The model folders each
receive their own `metrics.csv`, `metrics.json`, `predictions.csv`,
`error_analysis.png`, `feature_importance.png`, and `permutation_importance.csv`.

Seven regressors are included: Linear Regression, Ridge, Random Forest, Extra
Trees, Gradient Boosting, Support Vector Regression, and K Nearest Neighbors.
The `comparison/` folder contains the combined metric table and comparison
graphs. The `data/` folder contains the parsed samples and dataset summary.

The current `.iv` data files contain I-V simulations and per-run eta values;
they do not contain QE-versus-wavelength labels. This pipeline therefore
compares **eta prediction**, rather than the existing QE scripts. The current
sample is small and contains parameter sweeps. Its holdout evaluates settings
not repeated verbatim in training, but nearby settings from the same sweeps
can occur on both sides, so scores do not establish generalization to new
materials or simulation regimes.

Each model can also be run alone, for example:

```bash
./.venv/bin/python ml_models/random_forest/train.py
```

Run the comparison command again after adding or changing `.iv` files.

## Results for each IV file separately

To run every model independently on each root-level `.iv` file and write the
outputs to `results/`, run:

```bash
./.venv/bin/python ml_models/compare_each_file.py
```

The output layout is `results/<source-file-name>/`. Each source folder has a
`<source>_cleaned.csv` with one row per voltage/current scan point and a
`dataset.csv` with one row per simulation. Model folders contain metrics,
predictions, and graphs. The model runs read the cleaned CSV.
