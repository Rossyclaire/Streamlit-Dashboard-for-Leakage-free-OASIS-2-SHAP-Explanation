from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


from loaders import available_tables, load_table


st.set_page_config(
    page_title="Result Tables",
    layout="wide",
)


st.title("Result tables")

tables = available_tables()

if not tables:
    st.error("No result tables are available.")
    st.stop()


selected_table = st.selectbox(
    "Select a result table",
    options=sorted(tables),
)


table = load_table(selected_table)

st.write(
    f"Rows: {table.shape[0]} | "
    f"Columns: {table.shape[1]}"
)

st.dataframe(
    table,
    width="stretch",
    hide_index=True,
)

st.download_button(
    label="Download selected table",
    data=table.to_csv(index=False),
    file_name=f"{selected_table}.csv",
    mime="text/csv",
)
