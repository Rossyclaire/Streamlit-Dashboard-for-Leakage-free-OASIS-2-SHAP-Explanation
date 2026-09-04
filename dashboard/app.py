from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parent

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


from loaders import available_models, available_tables


st.set_page_config(
    page_title="OASIS-2 Dementia Dashboard",
    page_icon="D",
    layout="wide",
)


st.title("OASIS-2 Dementia Classification Dashboard")

st.markdown(
    """
This dashboard presents the leakage-aware OASIS-2 dementia
classification analysis.

Use the navigation menu on the left to access:

- Classifier comparison charts.
- SHAP visualisations.
- Patient risk scoring.
- Result tables.
- Existing saved figures.
"""
)


st.info(
    "Scope: Visit 1 baseline records, 136 participants, "
    "and eight approved predictors. CDR and MR Delay are excluded."
)


models = available_models()
tables = available_tables()


model_column, table_column = st.columns(2)

with model_column:
    st.metric(
        label="Validated models",
        value=len(models),
    )

with table_column:
    st.metric(
        label="Available result tables",
        value=len(tables),
    )


st.subheader("Validated models")

for model_name in models:
    st.write(f"• {model_name}")


st.subheader("Dashboard purpose")

st.write(
    "This dashboard is a research visualisation and prediction "
    "interface. It is not a clinical diagnostic system."
)

st.write(
    "All displayed model results are based on the existing saved "
    "pipelines and analysis tables. The dashboard does not retrain "
    "the models."
)
