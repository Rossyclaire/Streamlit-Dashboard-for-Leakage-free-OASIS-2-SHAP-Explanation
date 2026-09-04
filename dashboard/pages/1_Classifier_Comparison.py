from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


from loaders import load_table


st.set_page_config(
    page_title="Classifier Comparison",
    layout="wide",
)


st.title("Classifier comparison")

st.write(
    "Interactive comparison of the three saved classifiers using "
    "the held-out test-set results."
)


performance = load_table("test_performance")

if performance.empty:
    st.error("The performance table is empty.")
    st.stop()


model_column = "model"

if model_column not in performance.columns:
    st.error("The performance table does not contain a model column.")
    st.stop()


preferred_metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_weighted",
    "roc_auc",
    "pr_auc",
]

metric_options = [
    metric
    for metric in preferred_metrics
    if metric in performance.columns
]

if not metric_options:
    numeric_columns = performance.select_dtypes(
        include="number"
    ).columns.tolist()

    metric_options = numeric_columns


selected_metric = st.selectbox(
    "Select a metric",
    options=metric_options,
)


chart = px.bar(
    performance,
    x=model_column,
    y=selected_metric,
    color=model_column,
    text_auto=".3f",
    title=f"Classifier comparison by {selected_metric}",
)

chart.update_layout(
    showlegend=False,
    yaxis_title=selected_metric,
    xaxis_title="Classifier",
)

st.plotly_chart(chart, width="stretch")


st.download_button(
    label="Download performance table",
    data=performance.to_csv(index=False),
    file_name="table_test_set_performance.csv",
    mime="text/csv",
)


st.subheader("Performance table")

st.dataframe(
    performance,
    width="stretch",
    hide_index=True,
)


confusion_columns = [
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
]

if all(column in performance.columns for column in confusion_columns):
    st.subheader("Confusion matrix")

    selected_model = st.selectbox(
        "Select a classifier for the confusion matrix",
        options=performance[model_column].astype(str).tolist(),
    )

    selected_row = performance[
        performance[model_column].astype(str) == selected_model
    ].iloc[0]

    true_negatives = int(selected_row["true_negatives"])
    false_positives = int(selected_row["false_positives"])
    false_negatives = int(selected_row["false_negatives"])
    true_positives = int(selected_row["true_positives"])

    matrix = [
        [true_negatives, false_positives],
        [false_negatives, true_positives],
    ]

    matrix_chart = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[
                "Predicted Nondemented",
                "Predicted Demented",
            ],
            y=[
                "Actual Nondemented",
                "Actual Demented",
            ],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )

    matrix_chart.update_layout(
        title=f"Confusion matrix for {selected_model}",
        xaxis_title="Prediction",
        yaxis_title="Observed class",
    )

    matrix_chart.update_yaxes(autorange="reversed")

    st.plotly_chart(matrix_chart, width="stretch")


st.subheader("Pairwise DeLong AUC comparisons")

delong = load_table("delong_auc")

st.dataframe(
    delong,
    width="stretch",
    hide_index=True,
)

st.download_button(
    label="Download DeLong comparison table",
    data=delong.to_csv(index=False),
    file_name="table_pairwise_delong_auc.csv",
    mime="text/csv",
)
