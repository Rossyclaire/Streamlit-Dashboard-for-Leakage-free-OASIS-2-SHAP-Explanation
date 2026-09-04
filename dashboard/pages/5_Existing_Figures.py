from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


from loaders import figure_paths


st.set_page_config(
    page_title="Existing Figures",
    layout="wide",
)


st.title("Existing figures")

figures = figure_paths()

if not figures:
    st.info(
        "No supported image or PDF files were found in the "
        "04_figures folder."
    )
    st.stop()


for figure_path in figures:
    st.subheader(figure_path.name)

    if figure_path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
    }:
        st.image(
            str(figure_path),
            width="stretch",
        )

    st.download_button(
        label=f"Download {figure_path.name}",
        data=figure_path.read_bytes(),
        file_name=figure_path.name,
        key=f"download_{figure_path.name}",
    )
