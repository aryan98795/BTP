# QE linear-regression demo

Run the model from this directory:

```bash
.venv/bin/python train_qe_linear_regression.py
```

The script uses complete, physically plausible QE rows from `1_merged.csv`, with the eleven IV values and wavelength as inputs. It predicts `QE(%)` and `photon energy (eV)`, evaluates a fixed 80/20 train/test split, writes held-out predictions to `qe_linear_predictions.csv`, and saves the four-panel graph to `qe_linear_regression_plots.png`. It excludes report artifacts by requiring wavelength 300--900 nm, QE 0--100%, and energy 1--5 eV; it does not change the merged source file.

This is a baseline demonstration. The source files have different numbers of rows and were merged by row order, so a scientifically reliable model should instead join measurements using a shared simulation/batch identifier and physically meaningful alignment.

## Random Forest QE model

The linear-regression baseline is unchanged. To train a separate nonlinear
Random Forest model for `QE(%)`, run:

```bash
.venv/bin/python train_qe_random_forest.py
```

It uses the same valid input rows and 80/20 split, but predicts only `QE(%)`.
It writes `qe_random_forest_predictions.csv`,
`qe_random_forest_feature_importance.csv`, and `qe_random_forest_plots.png`.
Photon energy is not modeled because it is derived from wavelength
(approximately `1240 / lambda(nm)`).

## Interactive dashboard

Start the local dashboard with:

```bash
.venv/bin/streamlit run qe_dashboard.py
```

It uses a shuffled 60% training, 20% test, and 20% final-demo split. The dashboard shows MSE and R² for each target, actual-vs-predicted plots, residual plots, and interactive predictions for the final demo set.
