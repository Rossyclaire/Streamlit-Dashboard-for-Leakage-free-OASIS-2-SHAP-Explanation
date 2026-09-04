from __future__ import annotations

import sys
from pathlib import Path
from html import escape
import math
import hashlib


import pandas as pd
import plotly.express as px
import streamlit as st


def prepare_display_dataframe(df):
    """Prepare a dataframe safely for Streamlit display."""
    display_df = df.copy()

    # Mixed values such as 28, F, and 0.75 must be displayed as text
    if "patient_value" in display_df.columns:
        display_df["patient_value"] = (
            display_df["patient_value"]
            .fillna("")
            .astype(str)
        )

    return display_df


DASHBOARD_DIR = Path(__file__).resolve().parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


from loaders import (
    available_models,
    calculate_local_shap_transformed,
    expected_features,
    predict_dataframe,
)

from ng97_mapping import (
    prepare_mapped_local_shap_table,
)

def render_wrapped_html_table(
    dataframe: pd.DataFrame,
    column_widths: list[str],
) -> None:
    """Render a complete table with wrapped text cells."""
    if len(dataframe.columns) != len(column_widths):
        raise ValueError(
            "The number of column widths must match the number "
            "of dataframe columns."
        )

    column_group = "".join(
        f"<col style='width:{width}'>"
        for width in column_widths
    )

    header_cells = "".join(
        f"<th>{escape(str(column))}</th>"
        for column in dataframe.columns
    )

    body_rows = []

    for _, row in dataframe.iterrows():
        cells = []

        for value in row.tolist():
            if pd.isna(value):
                cell_text = ""
            else:
                cell_text = escape(str(value))

            cells.append(f"<td>{cell_text}</td>")

        body_rows.append(
            "<tr>" + "".join(cells) + "</tr>"
        )

    html_document = (
        "<style>"
        ".ng97-table-wrapper {"
        "width:100%;"
        "overflow-x:auto;"
        "}"
        ".ng97-wrapped-table {"
        "width:100%;"
        "table-layout:fixed;"
        "border-collapse:collapse;"
        "font-size:0.92rem;"
        "}"
        ".ng97-wrapped-table th,"
        ".ng97-wrapped-table td {"
        "border:1px solid #d9dee7;"
        "padding:0.55rem;"
        "text-align:left;"
        "vertical-align:top;"
        "white-space:normal;"
        "overflow-wrap:anywhere;"
        "word-break:break-word;"
        "line-height:1.35;"
        "}"
        ".ng97-wrapped-table th {"
        "background-color:#f1f4f8;"
        "font-weight:600;"
        "}"
        "</style>"
        "<div class='ng97-table-wrapper'>"
        "<table class='ng97-wrapped-table'>"
        "<colgroup>"
        + column_group
        + "</colgroup>"
        "<thead><tr>"
        + header_cells
        + "</tr></thead>"
        "<tbody>"
        + "".join(body_rows)
        + "</tbody>"
        "</table>"
        "</div>"
    )

    if hasattr(st, "html"):
        st.html(html_document)
    else:
        st.markdown(
            html_document,
            unsafe_allow_html=True,
        )

st.set_page_config(
    page_title="Patient Risk Scoring",
    layout="wide",
)


st.title("Patient risk scoring")

st.warning(
    "This is a research demonstration interface. It is not a "
    "clinical diagnostic or treatment tool."
)


models = available_models()

if not models:
    st.error("No saved models are available.")
    st.stop()


selected_model = st.selectbox(
    "Select a saved classifier",
    options=models,
)


features = expected_features(selected_model)

if not features:
    st.error(
        "The expected predictor names could not be extracted from "
        "the saved pipeline."
    )
    st.stop()


st.subheader("Required predictors")

st.write(
    "Enter the predictor names and values exactly as used in the "
    "original modelling dataset."
)

st.code("\n".join(features))


threshold = st.slider(
    "Exploratory model decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

st.caption(
    "The default 0.50 threshold is an exploratory display setting. "
    "It is not a clinically validated cutoff and is not a NICE NG97 "
    "threshold."
)


manual_tab, csv_tab = st.tabs(
    [
        "Manual patient entry",
        "CSV batch scoring",
    ]
)


NUMERIC_INPUT_RULES = {
    "Age": {
        "min_value": 60,
        "max_value": 98,
        "value": 70,
        "step": 1,
        "format": "%d",
    },
    "EDUC": {
        "min_value": 6,
        "max_value": 23,
        "value": 14,
        "step": 1,
        "format": "%d",
    },
    "SES": {
        "min_value": 1,
        "max_value": 5,
        "value": 2,
        "step": 1,
        "format": "%d",
    },
    "MMSE": {
        "min_value": 4,
        "max_value": 30,
        "value": 28,
        "step": 1,
        "format": "%d",
    },
    "eTIV": {
        "min_value": 1105.65,
        "max_value": 2004.48,
        "value": 1500.0,
        "step": 0.01,
        "format": "%.2f",
    },
    "nWBV": {
        "min_value": 0.644399,
        "max_value": 0.836842,
        "value": 0.75,
        "step": 0.000001,
        "format": "%.6f",
    },
    "ASF": {
        "min_value": 0.875539,
        "max_value": 1.587298,
        "value": 1.00,
        "step": 0.000001,
        "format": "%.6f",
    },
}

def validate_patient_dataframe(
    dataframe: pd.DataFrame,
) -> list[str]:
    errors = []

    if dataframe.empty:
        errors.append("The uploaded CSV is empty.")
        return errors

    missing_columns = [
        feature
        for feature in features
        if feature not in dataframe.columns
    ]

    if missing_columns:
        errors.append(
            "The CSV is missing these required columns: "
            + ", ".join(missing_columns)
        )
        return errors

    sex_values = (
        dataframe["M/F"]
        .astype("string")
        .str.strip()
    )

    if not sex_values.isin(["F", "M"]).all():
        errors.append(
            "The M/F column must contain only F or M."
        )

    for feature, rule in NUMERIC_INPUT_RULES.items():
        numeric_values = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        )

        invalid_values = (
            numeric_values.isna()
            | ~numeric_values.between(
                rule["min_value"],
                rule["max_value"],
            )
        )

        if rule["step"] == 1:
            invalid_values = invalid_values | (
                numeric_values.notna()
                & numeric_values.mod(1).ne(0)
            )

        if invalid_values.any():
            errors.append(
                f"{feature} contains missing, non-numeric, "
                "non-whole-number or out-of-range values."
            )

    return errors


with manual_tab:
    st.subheader("Score one patient")

    with st.form("manual_patient_form"):
        input_values: dict[str, str] = {}

        input_columns = st.columns(2)

        for index, feature in enumerate(features):
            with input_columns[index % 2]:
                if feature == "M/F":
                    input_values[feature] = st.selectbox(
                        label="M/F",
                        options=["F", "M"],
                        index=0,
                        key=f"manual_input_{feature}",
                    )
                else:
                    rule = NUMERIC_INPUT_RULES[feature]

                    input_values[feature] = st.text_input(
                        label=str(feature),
                        value=str(rule["value"]),
                        key=f"manual_input_{feature}",
                    )
  
        submitted = st.form_submit_button(
            "Calculate patient score"
        )

    if submitted:
        validation_errors = []

        validated_values: dict[str, object] = {
            "M/F": input_values["M/F"].strip()
        }

        if validated_values["M/F"] not in {"F", "M"}:
            validation_errors.append(
                "M/F must be either F or M."
            )

        for feature, rule in NUMERIC_INPUT_RULES.items():
            raw_value = input_values[feature].strip()

            if raw_value == "":
                validation_errors.append(
                    f"{feature} is required."
                )
                continue

            try:
                numeric_value = float(raw_value)
            except ValueError:
                validation_errors.append(
                    f"{feature} must contain a numeric value."
                )
                continue

            if not math.isfinite(numeric_value):
                validation_errors.append(
                    f"{feature} must contain a finite numeric value."
                )
                continue

            if (
                rule["step"] == 1
                and not numeric_value.is_integer()
            ):
                validation_errors.append(
                    f"{feature} must be a whole number."
                )
                continue

            if not (
                rule["min_value"]
                <= numeric_value
                <= rule["max_value"]
            ):
                validation_errors.append(
                    f"{feature} must be between "
                    f"{rule['min_value']} and "
                    f"{rule['max_value']}."
                )
                continue

            if numeric_value.is_integer():
                validated_values[feature] = int(numeric_value)
            else:
                validated_values[feature] = numeric_value

        if validation_errors:
            for error_message in validation_errors:
                st.error(error_message)

            st.stop()

        patient_dataframe = pd.DataFrame(
            [validated_values],
            columns=features,
        )

        try:
            prediction_result = predict_dataframe(
                selected_model,
                patient_dataframe,
            )

            probability = float(
                prediction_result[
                    "dementia_probability"
                ].iloc[0]
            )

            threshold_decision = (
                probability >= threshold
            )

            st.subheader("Patient score")

            score_column, decision_column = st.columns(2)

            with score_column:
                st.metric(
                    label="Dementia probability",
                    value=f"{probability:.3f}",
                )

                st.progress(probability)

            with decision_column:
                if threshold_decision:
                    st.error(
                        "Model probability is at or above the "
                        "exploratory threshold."
                    )
                else:
                    st.success(
                        "Model probability is below the exploratory "
                        "threshold."
                    )

            prediction_result["threshold"] = threshold
            prediction_result["threshold_decision"] = (
                threshold_decision
            )

            st.dataframe(
                prepare_display_dataframe(prediction_result),
                width="stretch",
                hide_index=True,
            )

            st.subheader(
                "Patient-specific SHAP explanation"
            )

            st.caption(
                "Positive SHAP values increase the model's "
                "Demented probability. Negative SHAP values "
                "decrease it. These explanations are model-based "
                "and are not clinical recommendations."
            )

            try:
                (
                    local_shap_table,
                    shap_base_value,
                    shap_probability,
                ) = calculate_local_shap_transformed(
                    selected_model,
                    patient_dataframe,
                )

                st.write(
                    f"SHAP base value: "
                    f"{shap_base_value:.4f}"
                )

                shap_chart = px.bar(
                    local_shap_table.sort_values(
                        "shap_value",
                        ascending=True,
                    ),
                    x="shap_value",
                    y="feature",
                    color="direction",
                    orientation="h",
                    hover_data=[
                        "patient_value",
                        "absolute_shap_value",
                    ],
                    title=(
                        "Local SHAP contributions for "
                        "this patient"
                    ),
                    color_discrete_map={
                        "Raises Demented probability": (
                            "#d62728"
                        ),
                        "Lowers Demented probability": (
                            "#1f77b4"
                        ),
                        "Neutral": "#7f7f7f",
                    },
                )

                shap_chart.update_layout(
                    xaxis_title=(
                        "SHAP contribution to "
                        "Demented probability"
                    ),
                    yaxis_title="Predictor",
                    showlegend=True,
                )

                st.plotly_chart(
                    shap_chart,
                    width="stretch",
                )

                st.dataframe(
                    prepare_display_dataframe(local_shap_table),
                    width="stretch",
                    hide_index=True,
                )

                mapped_shap_table = (
                    prepare_mapped_local_shap_table(
                        local_shap_table
                    )
                )

                st.subheader(
                    "SHAP to NICE NG97 conceptual crosswalk"
                )

                st.caption(
                    "This table joins the patient's model-derived "
                    "SHAP contributions to the approved feature-to-"
                    "NG97 mapping. It provides clinical context only. "
                    "It does not establish a NICE recommendation, "
                    "clinical validity, or NHS deployment readiness."
                )
                mapped_display_table = pd.DataFrame(
                    {
                        "Feature": (
                            mapped_shap_table["feature"]
                            .astype(str)
                        ),
                        "Patient value": (
                            mapped_shap_table["patient_value"]
                        ),
                        "SHAP contribution": (
                            pd.to_numeric(
                                mapped_shap_table["shap_value"],
                                errors="coerce",
                            ).round(4)
                        ),
                        "Model direction": (
                            mapped_shap_table["direction"]
                            .astype(str)
                        ),
                        "NG97 domain": (
                            mapped_shap_table["ng97_domain"]
                            .astype(str)
                        ),
                        "Mapping relationship": (
                            mapped_shap_table["relationship"]
                            .astype(str)
                        ),
                        "Patient-specific model interpretation": (
                            mapped_shap_table[
                                "patient_model_interpretation"
                            ].astype(str)
                        ),
                    }
                )

                render_wrapped_html_table(
                    mapped_display_table,
                    [
                        "8%",
                        "9%",
                        "12%",
                        "15%",
                        "15%",
                        "18%",
                        "23%",
                    ],
                )
                

                with st.expander(
                    "View mapping rationale and limitations"
                ):
                    rationale_display_table = pd.DataFrame(
                        {
                            "Feature": (
                                mapped_shap_table["feature"]
                                .astype(str)
                            ),
                            "NG97 domain": (
                                mapped_shap_table["ng97_domain"]
                                .astype(str)
                            ),
                            "Relationship": (
                                mapped_shap_table["relationship"]
                                .astype(str)
                            ),
                            "Approved mapping interpretation": (
                                mapped_shap_table[
                                    "mapping_interpretation"
                                ].astype(str)
                            ),
                            "Mapping status": (
                                mapped_shap_table["mapping_status"]
                                .astype(str)
                            ),
                            "Clinical-use status": (
                                mapped_shap_table[
                                    "clinical_use_status"
                                ].astype(str)
                            ),
                        }
                    )

                    render_wrapped_html_table(
                        rationale_display_table,
                        [
                            "8%",
                            "17%",
                            "18%",
                            "32%",
                            "12%",
                            "13%",
                        ],
                    )


                st.warning(
                    "Conceptual limitation: a feature-to-NG97 "
                    "mapping does not mean that NICE recommends the "
                    "feature as a diagnostic predictor. SHAP values "
                    "describe model behaviour for this input and do "
                    "not establish causality, diagnostic necessity, "
                    "or treatment relevance."
                )

                st.info(
                    "NHS limitation: this interface uses the saved "
                    "OASIS-2 research model and has not demonstrated "
                    "external NHS validation, calibration, prospective "
                    "clinical utility, subgroup fairness, workflow "
                    "compatibility, or information-governance "
                    "readiness. It must not be used to diagnose, "
                    "triage, or treat a patient."
                )

                st.download_button(
                    label=(
                        "Download SHAP to NG97 explanation"
                    ),
                    data=mapped_shap_table.to_csv(
                        index=False
                    ),
                    file_name=(
                        "patient_shap_to_ng97_crosswalk.csv"
                    ),
                    mime="text/csv",
                    key="manual_shap_ng97_download",
                )


                st.download_button(
                    label=(
                        "Download patient SHAP explanation"
                    ),
                    data=local_shap_table.to_csv(
                        index=False
                    ),
                    file_name=(
                        "patient_local_shap_explanation.csv"
                    ),
                    mime="text/csv",
                    key="manual_local_shap_download",
                )

            except Exception as error:
                st.warning(
                    "The patient prediction was calculated, "
                    "but the local SHAP explanation could not "
                    "be generated."
                )

                st.code(str(error))


        except Exception as error:
            st.error(
                "The model could not score this input. "
                f"Details: {error}"
            )


with csv_tab:
    st.subheader("Score multiple patients from CSV")

    uploaded_file = st.file_uploader(
        "Upload a CSV containing the required predictor columns",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            uploaded_dataframe = pd.read_csv(uploaded_file)
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            st.error(
                "The uploaded file could not be read as a CSV. "
                f"Details: {error}"
            )
            st.stop()

        st.write("Uploaded data preview")

        st.dataframe(
            prepare_display_dataframe(uploaded_dataframe.head()),
            width="stretch",
            hide_index=True,
        )

        csv_validation_errors = (
            validate_patient_dataframe(
                uploaded_dataframe
            )
        )

        if csv_validation_errors:
            for error_message in csv_validation_errors:
                st.error(error_message)

            st.stop()

        validated_uploaded_dataframe = uploaded_dataframe.copy()

        validated_uploaded_dataframe["M/F"] = (
            validated_uploaded_dataframe["M/F"]
            .astype("string")
            .str.strip()
        )

        for feature, rule in NUMERIC_INPUT_RULES.items():
            numeric_values = pd.to_numeric(
                validated_uploaded_dataframe[feature],
                errors="coerce",
            )

            if rule["step"] == 1:
                validated_uploaded_dataframe[feature] = (
                    numeric_values.astype("int64")
                )
            else:
                validated_uploaded_dataframe[feature] = (
                    numeric_values.astype("float64")
                )
      
        upload_key = (
            f"{selected_model}:"
            f"{hashlib.sha256(uploaded_file.getvalue()).hexdigest()}"
        )

        if (
            st.session_state.get("batch_upload_key")
            != upload_key
        ):
            st.session_state["batch_upload_key"] = upload_key
            st.session_state["batch_result"] = None
            st.session_state["batch_dataframe"] = None

        if st.button("Score uploaded patients"):
            try:
                scored_result = predict_dataframe(
                    selected_model,
                    validated_uploaded_dataframe,
                )

                scored_result["threshold"] = threshold

                scored_result["threshold_decision"] = (
                    scored_result["dementia_probability"]
                    >= threshold
                )

                st.session_state["batch_result"] = (
                    scored_result
                )

                st.session_state["batch_dataframe"] = (
                    validated_uploaded_dataframe.copy()
                )

            except Exception as error:
                st.session_state["batch_result"] = None
                st.session_state["batch_dataframe"] = None

                st.error(
                    "The uploaded data could not be scored. "
                    f"Details: {error}"
                )

        batch_result = st.session_state.get(
            "batch_result"
        )

        scored_dataframe = st.session_state.get(
            "batch_dataframe"
        )

        if (
            batch_result is not None
            and scored_dataframe is not None
        ):
            batch_result = batch_result.copy()

            batch_result["threshold"] = threshold

            batch_result["threshold_decision"] = (
                batch_result["dementia_probability"]
                >= threshold
            )

            st.subheader("Batch scoring results")

            st.dataframe(
                prepare_display_dataframe(batch_result),
                width="stretch",
                hide_index=True,
            )

            st.subheader(
                "Explain one uploaded patient"
            )

            selected_batch_row = st.selectbox(
                "Select a scored row for local explanation",
                options=list(
                    range(len(scored_dataframe))
                ),
                format_func=lambda row_number: (
                    f"Uploaded row {row_number + 1}"
                ),
            )

            if st.button(
                "Generate SHAP to NG97 explanation",
                key="batch_shap_ng97_button",
            ):
                try:
                    selected_patient_dataframe = (
                        scored_dataframe
                        .iloc[[selected_batch_row]]
                        .loc[:, features]
                        .copy()
                    )

                    (
                        batch_local_shap_table,
                        batch_shap_base_value,
                        batch_shap_probability,
                    ) = calculate_local_shap_transformed(
                        selected_model,
                        selected_patient_dataframe,
                    )

                    batch_mapped_shap_table = (
                        prepare_mapped_local_shap_table(
                            batch_local_shap_table
                        )
                    )

                    st.write(
                        f"SHAP base value: "
                        f"{batch_shap_base_value:.4f}"
                    )

                    st.write(
                        f"Model Demented probability: "
                        f"{batch_shap_probability:.4f}"
                    )

                    st.dataframe(
                        prepare_display_dataframe(
                            batch_mapped_shap_table
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    st.warning(
                        "This is a conceptual model "
                        "explanation for the selected "
                        "uploaded row. It is not a clinical "
                        "assessment or NHS decision."
                    )

                    st.download_button(
                        label=(
                            "Download selected-row "
                            "SHAP to NG97 crosswalk"
                        ),
                        data=(
                            batch_mapped_shap_table
                            .to_csv(index=False)
                        ),
                        file_name=(
                            "selected_patient_shap_to_ng97.csv"
                        ),
                        mime="text/csv",
                        key="batch_shap_ng97_download",
                    )

                except Exception as error:
                    st.error(
                        "The local SHAP explanation could "
                        "not be generated. "
                        f"Details: {error}"
                    )

            st.download_button(
                label="Download scored patients",
                data=batch_result.to_csv(
                    index=False
                ),
                file_name="patient_risk_scores.csv",
                mime="text/csv",
                key="batch_scores_download",
            )
