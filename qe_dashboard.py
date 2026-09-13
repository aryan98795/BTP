"""Interactive 60/20/20 linear-regression dashboard for the QE data."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_qe_linear_regression import FEATURES, TARGETS, load_complete_rows


st.set_page_config(page_title="QE Regression Demo", page_icon="📈", layout="wide")


@st.cache_data
def get_data():
    return load_complete_rows("1_merged.csv")


def metric_frame(actual, predicted):
    return pd.DataFrame({
        "Target": TARGETS,
        "MSE": [mean_squared_error(actual[:, i], predicted[:, i]) for i in range(2)],
        "R²": [r2_score(actual[:, i], predicted[:, i]) for i in range(2)],
    })


def scatter_chart(frame, target):
    minimum = min(frame["Actual"].min(), frame["Predicted"].min())
    maximum = max(frame["Actual"].max(), frame["Predicted"].max())
    points = alt.Chart(frame).mark_circle(size=45, opacity=0.5).encode(
        x=alt.X("Actual:Q", title=f"Actual {target}"),
        y=alt.Y("Predicted:Q", title=f"Predicted {target}"),
        tooltip=["Actual", "Predicted", "Residual"],
    )
    diagonal = alt.Chart(pd.DataFrame({"value": [minimum, maximum]})).mark_line(
        color="#e45756", strokeDash=[6, 4]
    ).encode(x="value:Q", y="value:Q")
    return (points + diagonal).properties(height=350, title=f"{target}: actual vs predicted").interactive()


def residual_chart(frame, target):
    points = alt.Chart(frame).mark_circle(size=45, opacity=0.5).encode(
        x=alt.X("Predicted:Q", title=f"Predicted {target}"),
        y=alt.Y("Residual:Q", title="Prediction − actual"),
        tooltip=["Actual", "Predicted", "Residual"],
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#e45756", strokeDash=[6, 4]).encode(y="zero:Q")
    return (points + zero).properties(height=350, title=f"{target}: residuals").interactive()


st.title("QE Linear Regression — Live Demo")
st.caption("Train 60% • Test 20% • Final demo 20%. The test set is held out during fitting; the demo set is reserved for the interactive results below.")

with st.sidebar:
    st.header("Model controls")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    run_model = st.button("Train and update dashboard", type="primary", use_container_width=True)
    st.caption("Uses the 11 IV columns plus wavelength as features.")

if run_model or "results" not in st.session_state:
    progress = st.progress(0, text="Loading valid QE rows…")
    x, y = get_data()
    progress.progress(25, text="Creating the 60/20/20 split…")
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        x, y, test_size=0.40, random_state=int(seed)
    )
    x_test, x_demo, y_test, y_demo = train_test_split(
        x_remaining, y_remaining, test_size=0.50, random_state=int(seed)
    )
    progress.progress(55, text="Training linear regression…")
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x_train, y_train)
    progress.progress(80, text="Calculating live predictions and metrics…")
    st.session_state.results = {
        "x_train": x_train, "x_test": x_test, "x_demo": x_demo,
        "y_train": y_train, "y_test": y_test, "y_demo": y_demo,
        "test_prediction": model.predict(x_test),
        "demo_prediction": model.predict(x_demo),
    }
    progress.progress(100, text="Dashboard ready.")
    progress.empty()

results = st.session_state.results
test_metrics = metric_frame(results["y_test"], results["test_prediction"])
demo_metrics = metric_frame(results["y_demo"], results["demo_prediction"])

counts = st.columns(3)
counts[0].metric("Training rows (60%)", len(results["y_train"]))
counts[1].metric("Test rows (20%)", len(results["y_test"]))
counts[2].metric("Demo rows (20%)", len(results["y_demo"]))

test_tab, demo_tab, data_tab = st.tabs(["Test results", "Live demo", "Data split"])
with test_tab:
    st.subheader("Held-out test metrics")
    metrics = st.columns(2)
    for i, target in enumerate(TARGETS):
        metrics[i].metric(f"{target} MSE", f"{test_metrics.loc[i, 'MSE']:.4f}")
        metrics[i].metric(f"{target} R²", f"{test_metrics.loc[i, 'R²']:.4f}")
    for i, target in enumerate(TARGETS):
        frame = pd.DataFrame({"Actual": results["y_test"][:, i], "Predicted": results["test_prediction"][:, i]})
        frame["Residual"] = frame["Predicted"] - frame["Actual"]
        left, right = st.columns(2)
        left.altair_chart(scatter_chart(frame, target), use_container_width=True)
        right.altair_chart(residual_chart(frame, target), use_container_width=True)

with demo_tab:
    st.subheader("Final 20% demo set")
    st.dataframe(demo_metrics.style.format({"MSE": "{:.4f}", "R²": "{:.4f}"}), hide_index=True, use_container_width=True)
    target = st.selectbox("Target to inspect", TARGETS)
    target_index = TARGETS.index(target)
    demo_frame = pd.DataFrame({
        "Sample": np.arange(1, len(results["y_demo"]) + 1),
        "Actual": results["y_demo"][:, target_index],
        "Predicted": results["demo_prediction"][:, target_index],
    })
    long_demo = demo_frame.melt("Sample", value_vars=["Actual", "Predicted"], var_name="Series", value_name="Value")
    chart = alt.Chart(long_demo).mark_line(point=True).encode(
        x=alt.X("Sample:Q", title="Demo sample"), y=alt.Y("Value:Q", title=target),
        color="Series:N", tooltip=["Sample", "Series", "Value"],
    ).properties(height=420, title=f"Live predictions for {target}").interactive()
    st.altair_chart(chart, use_container_width=True)
    sample = st.slider("Inspect a demo sample", 1, len(demo_frame), 1)
    row = demo_frame.iloc[sample - 1]
    values = st.columns(3)
    values[0].metric("Actual", f"{row['Actual']:.5g}")
    values[1].metric("Predicted", f"{row['Predicted']:.5g}")
    values[2].metric("Absolute error", f"{abs(row['Predicted'] - row['Actual']):.5g}")

with data_tab:
    st.write("Complete, physically plausible rows used for modeling:", len(results["y_train"]) + len(results["y_test"]) + len(results["y_demo"]))
    st.caption("The source CSV is not modified. Data rows are shuffled before the 60/20/20 split using the selected seed.")
    st.dataframe(pd.DataFrame(results["x_train"][:10], columns=FEATURES), use_container_width=True)
