from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


from loaders import load_table


st.set_page_config(
    page_title="SHAP Visualisations",
    layout="wide",
)


st.title("SHAP visualisations")

st.write(
    "Interactive global feature-importance and SHAP-value "
    "visualisations based on the saved analysis tables."
)


def find_column(
    dataframe: pd.DataFrame,
    search_terms: list[str],
) -> str | None:
    """Find a column whose name contains one of the search terms."""
    for column in dataframe.columns:
        column_text = str(column).lower()

        if any(term in column_text for term in search_terms):
            return column

    return None


st.subheader("Consensus feature ranking")

consensus = load_table("shap_consensus")

consensus_feature_column = find_column(
    consensus,
    ["feature", "variable", "predictor"],
)

consensus_value_column = find_column(
    consensus,
    ["mean", "importance", "score", "rank"],
)

if (
    consensus_feature_column is not None
    and consensus_value_column is not None
):
    consensus_plot_data = consensus.copy()

    consensus_plot_data[consensus_value_column] = pd.to_numeric(
        consensus_plot_data[consensus_value_column],
        errors="coerce",
    )

    consensus_plot_data = consensus_plot_data.dropna(
        subset=[consensus_value_column]
    )

    consensus_plot_data = consensus_plot_data.sort_values(
        consensus_value_column,
        ascending=True,
    )

    consensus_chart = px.bar(
        consensus_plot_data,
        x=consensus_value_column,
        y=consensus_feature_column,
        orientation="h",
        text_auto=".3f",
        title="Consensus SHAP feature ranking",
    )

    st.plotly_chart(
        consensus_chart,
        width="stretch",
    )

else:
    st.dataframe(
        consensus,
        width="stretch",
        hide_index=True,
    )


st.subheader("Model-specific feature importance")

importance_table_name = st.selectbox(
    "Select a SHAP importance table",
    options=[
        "shap_original",
        "shap_transformed",
    ],
)


importance = load_table(importance_table_name)

importance_feature_column = find_column(
    importance,
    ["feature", "variable", "predictor"],
)

importance_value_column = find_column(
    importance,
    ["mean_abs", "mean", "importance", "shap"],
)

importance_model_column = find_column(
    importance,
    ["model", "classifier", "estimator"],
)


if importance_model_column is not None:
    model_options = (
        importance[importance_model_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_model = st.selectbox(
        "Select a model",
        options=model_options,
    )

    importance = importance[
        importance[importance_model_column].astype(str)
        == selected_model
    ]


if (
    importance_feature_column is not None
    and importance_value_column is not None
):
    importance[importance_value_column] = pd.to_numeric(
        importance[importance_value_column],
        errors="coerce",
    )

    importance = importance.dropna(
        subset=[importance_value_column]
    )

    importance = importance.sort_values(
        importance_value_column,
        ascending=True,
    )

    importance_chart = px.bar(
        importance,
        x=importance_value_column,
        y=importance_feature_column,
        orientation="h",
        text_auto=".3f",
        title="Feature importance",
    )

    st.plotly_chart(
        importance_chart,
        width="stretch",
    )

else:
    st.dataframe(
        importance,
        width="stretch",
        hide_index=True,
    )


st.subheader("SHAP value distribution")

shap_values = load_table("shap_values_long")

shap_feature_column = find_column(
    shap_values,
    ["feature", "variable", "predictor"],
)

shap_value_column = find_column(
    shap_values,
    ["shap_value", "shap", "contribution"],
)

shap_model_column = find_column(
    shap_values,
    ["model", "classifier", "estimator"],
)


if shap_model_column is not None:
    shap_model_options = (
        shap_values[shap_model_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_shap_model = st.selectbox(
        "Select a model for the SHAP distribution",
        options=shap_model_options,
    )

    shap_values = shap_values[
        shap_values[shap_model_column].astype(str)
        == selected_shap_model
    ]


if (
    shap_feature_column is not None
    and shap_value_column is not None
):
    shap_values["_shap_numeric"] = pd.to_numeric(
        shap_values[shap_value_column],
        errors="coerce",
    )

    shap_values = shap_values.dropna(
        subset=["_shap_numeric"]
    )

    shap_chart = px.strip(
        shap_values,
        x="_shap_numeric",
        y=shap_feature_column,
        title="Distribution of SHAP values",
        hover_data=shap_values.columns.tolist(),
    )

    shap_chart.update_layout(
        xaxis_title="SHAP value",
        yaxis_title="Feature",
    )

    st.plotly_chart(
        shap_chart,
        width="stretch",
    )

else:
    st.warning(
        "The SHAP value table does not expose recognisable "
        "feature and SHAP-value columns."
    )


st.subheader("SHAP values table")

st.dataframe(
    shap_values,
    width="stretch",
    hide_index=True,
)

st.download_button(
    label="Download SHAP values",
    data=shap_values.to_csv(index=False),
    file_name="table_shap_values_long.csv",
    mime="text/csv",
)
