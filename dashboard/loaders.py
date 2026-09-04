from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "06_models"
TABLES_DIR = PROJECT_ROOT / "05_tables"
FIGURES_DIR = PROJECT_ROOT / "04_figures"


MODEL_PATHS = {
    "Logistic Regression": (
        MODELS_DIR / "logistic_regression_best_pipeline.joblib"
    ),
    "Random Forest": (
        MODELS_DIR / "random_forest_best_pipeline.joblib"
    ),
    "SVC": MODELS_DIR / "svc_best_pipeline.joblib",
}


TABLE_PATHS = {
    "descriptive_statistics": (
        TABLES_DIR / "table_descriptive_statistics.csv"
    ),
    "missing_values": (
        TABLES_DIR / "table_missing_values.csv"
    ),
    "delong_auc": (
        TABLES_DIR / "table_pairwise_delong_auc.csv"
    ),
    "shap_original": (
        TABLES_DIR / "table_shap_feature_importance_original.csv"
    ),
    "shap_transformed": (
        TABLES_DIR / "table_shap_feature_importance_transformed.csv"
    ),
    "shap_consensus": (
        TABLES_DIR / "table_shap_feature_ranking_consensus.csv"
    ),
    "shap_values_long": (
        TABLES_DIR / "table_shap_values_long.csv"
    ),
    "test_predictions": (
        TABLES_DIR / "table_test_predictions.csv"
    ),
    "test_performance": (
        TABLES_DIR / "table_test_set_performance.csv"
    ),
    "tuning_results": (
        TABLES_DIR / "table_tuning_results.csv"
    ),
}


@st.cache_data
def load_table(table_name: str) -> pd.DataFrame:
    """Load one approved dashboard result table."""
    if table_name not in TABLE_PATHS:
        valid_names = ", ".join(sorted(TABLE_PATHS))
        raise KeyError(
            f"Unknown table '{table_name}'. "
            f"Valid names: {valid_names}"
        )

    path = TABLE_PATHS[table_name]

    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")

    return pd.read_csv(path)


@st.cache_resource
def load_model(model_name: str) -> Any:
    """Load one approved saved model pipeline."""
    if model_name not in MODEL_PATHS:
        valid_names = ", ".join(sorted(MODEL_PATHS))
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Valid names: {valid_names}"
        )

    path = MODEL_PATHS[model_name]

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def available_models() -> list[str]:
    """Return the saved models that are available."""
    return [
        model_name
        for model_name, path in MODEL_PATHS.items()
        if path.exists()
    ]


def available_tables() -> list[str]:
    """Return the result tables that are available."""
    return [
        table_name
        for table_name, path in TABLE_PATHS.items()
        if path.exists()
    ]


def load_all_tables() -> dict[str, pd.DataFrame]:
    """Load all available result tables."""
    return {
        table_name: load_table(table_name)
        for table_name in available_tables()
    }


def _find_feature_column(dataframe: pd.DataFrame) -> str | None:
    """Find a likely feature-name column in a dataframe."""
    preferred_terms = (
        "feature",
        "variable",
        "predictor",
        "parameter",
    )

    for column in dataframe.columns:
        column_text = str(column).lower()

        if any(term in column_text for term in preferred_terms):
            if "value" not in column_text:
                if "rank" not in column_text:
                    return column

    for column in dataframe.columns:
        if dataframe[column].dtype == "object":
            return column

    return None


def _extract_feature_names(model: Any) -> list[str]:
    """Extract input feature names from a fitted model or pipeline."""
    direct_names = getattr(model, "feature_names_in_", None)

    if direct_names is not None:
        return [str(name) for name in direct_names]

    named_steps = getattr(model, "named_steps", {})

    for step in reversed(list(named_steps.values())):
        step_names = getattr(step, "feature_names_in_", None)

        if step_names is not None:
            return [str(name) for name in step_names]

        transformers = getattr(step, "transformers_", None)

        if transformers is not None:
            extracted_names: list[str] = []

            for _, _, columns in transformers:
                if isinstance(columns, (list, tuple)):
                    extracted_names.extend(
                        str(column) for column in columns
                    )

            if extracted_names:
                return extracted_names

    return []


@st.cache_data
def expected_features(model_name: str) -> list[str]:
    """
    Return the expected input features for a saved model.

    If the model does not expose feature_names_in_, the SHAP consensus
    table is used as a fallback source for the original predictor names.
    """
    model = load_model(model_name)

    model_features = _extract_feature_names(model)

    if model_features:
        return model_features

    try:
        consensus_table = load_table("shap_consensus")
        feature_column = _find_feature_column(consensus_table)

        if feature_column is not None:
            return (
                consensus_table[feature_column]
                .dropna()
                .astype(str)
                .tolist()
            )

    except Exception:
        pass

    return []


def _dementia_class_index(model: Any) -> int:
    """Find the probability column representing the Demented class."""
    classes = list(getattr(model, "classes_", []))

    if not classes:
        return 1

    dementia_labels = {
        "demented",
        "d",
        "1",
        "true",
    }

    for index, class_label in enumerate(classes):
        normalised_label = str(class_label).strip().lower()

        if normalised_label in dementia_labels:
            return index

    return len(classes) - 1


def predict_dataframe(
    model_name: str,
    input_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions from a saved pipeline.

    The input dataframe must contain the exact predictors expected by
    the saved model. Additional columns are retained but ignored by
    the model.
    """
    model = load_model(model_name)
    features = expected_features(model_name)

    if not features:
        raise ValueError(
            "The saved model does not expose its expected feature names."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in input_dataframe.columns
    ]

    if missing_features:
        missing_text = ", ".join(missing_features)
        raise ValueError(
            f"Missing required predictor columns: {missing_text}"
        )

    model_input = input_dataframe.loc[:, features].copy()

    predictions = model.predict(model_input)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "This saved model does not provide probability estimates."
        )

    probabilities = model.predict_proba(model_input)
    dementia_index = _dementia_class_index(model)

    result = input_dataframe.copy()
    result["model_prediction"] = predictions
    result["dementia_probability"] = probabilities[:, dementia_index]

    return result


def figure_paths() -> list[Path]:
    """Return available saved figure files."""
    if not FIGURES_DIR.exists():
        return []

    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".pdf",
    }

    return sorted(
        path
        for path in FIGURES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )


def _find_shap_column(
    dataframe: pd.DataFrame,
    include_terms: tuple[str, ...],
    exclude_terms: tuple[str, ...] = (),
) -> str | None:
    """Find a column based on included and excluded name fragments."""
    for column in dataframe.columns:
        column_text = str(column).lower()

        includes_match = any(
            term in column_text
            for term in include_terms
        )

        excludes_match = any(
            term in column_text
            for term in exclude_terms
        )

        if includes_match and not excludes_match:
            return column

    return None


def _load_shap_background(model_name: str) -> pd.DataFrame:
    """
    Reconstruct a background dataset from the long-format SHAP table.

    The background contains existing feature values used to calculate
    patient-specific SHAP values.
    """
    shap_values = load_table("shap_values_long")
    features = expected_features(model_name)

    if not features:
        raise ValueError(
            "The model's expected feature names could not be found."
        )

    feature_column = _find_shap_column(
        shap_values,
        (
            "feature",
            "variable",
            "predictor",
        ),
    )

    shap_value_column = _find_shap_column(
        shap_values,
        (
            "shap_value",
            "shap",
            "contribution",
        ),
        (
            "mean",
            "importance",
            "rank",
        ),
    )

    feature_value_column = _find_shap_column(
        shap_values,
        (
            "feature_value",
            "input_value",
            "original_value",
            "featurevalue",
            "value",
        ),
        (
            "shap_value",
            "importance",
            "mean",
            "rank",
        ),
    )

    model_column = _find_shap_column(
        shap_values,
        (
            "model",
            "classifier",
            "estimator",
        ),
    )

    sample_column = _find_shap_column(
        shap_values,
        (
            "sample_id",
            "sample",
            "observation",
            "row_id",
        ),
    )

    if feature_column is None:
        raise ValueError(
            "The SHAP table does not contain a feature column."
        )

    if shap_value_column is None:
        raise ValueError(
            "The SHAP table does not contain a SHAP-value column."
        )

    if feature_value_column is None:
        raise ValueError(
            "The SHAP table does not contain feature values. "
            "Patient-specific SHAP calculation requires the original "
            "feature values used to create the background dataset."
        )

    working_data = shap_values.copy()

    if model_column is not None:
        model_names = (
            working_data[model_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if model_name in model_names:
            working_data = working_data[
                working_data[model_column].astype(str)
                == model_name
            ]

    working_data[feature_column] = (
        working_data[feature_column]
        .astype(str)
    )

    working_data = working_data[
        working_data[feature_column].isin(features)
    ]

    if sample_column is not None:
        working_data["_sample_id"] = (
            working_data[sample_column].astype(str)
        )
    else:
        working_data["_sample_id"] = (
            working_data.groupby(feature_column)
            .cumcount()
        )

    background = working_data.pivot_table(
        index="_sample_id",
        columns=feature_column,
        values=feature_value_column,
        aggfunc="first",
    )

    missing_features = [
        feature
        for feature in features
        if feature not in background.columns
    ]

    if missing_features:
        raise ValueError(
            "The SHAP background is missing these features: "
            + ", ".join(missing_features)
        )

    background = background.loc[:, features].copy()

    for feature in features:
        numeric_values = pd.to_numeric(
            background[feature],
            errors="coerce",
        )

        non_missing_count = background[feature].notna().sum()

        if numeric_values.notna().sum() == non_missing_count:
            background[feature] = numeric_values

    background = background.dropna(
        how="any"
    ).reset_index(drop=True)

    if len(background) < 2:
        raise ValueError(
            "At least two complete background observations are "
            "required for local SHAP calculation."
        )

    return background


def calculate_local_shap(
    model_name: str,
    patient_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """
    Calculate patient-specific SHAP values for the Demented probability.

    Returns:
        A table of local SHAP values.
        The SHAP base value.
        The model's Demented probability.
    """
    import shap

    model = load_model(model_name)
    features = expected_features(model_name)

    if not features:
        raise ValueError(
            "The model's expected feature names could not be found."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in patient_dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "The patient input is missing these features: "
            + ", ".join(missing_features)
        )

    patient_input = patient_dataframe.loc[:, features].copy()
    background = _load_processed_background(model_name)

    shap_background = background.copy()
    shap_patient = patient_input.copy()
    categorical_maps = {}

    for feature in features:
        background_is_text = (
            background[feature].dtype == "object"
        )

        patient_is_text = (
            patient_input[feature].dtype == "object"
        )

        if background_is_text or patient_is_text:
            background_categories = (
                background[feature]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            patient_category = str(
                patient_input.iloc[0][feature]
            )

            categories = sorted(
                set(background_categories + [patient_category])
            )

            category_to_code = {
                category: float(index)
                for index, category in enumerate(categories)
            }

            code_to_category = {
                float(index): category
                for category, index in category_to_code.items()
            }

            categorical_maps[feature] = (
                code_to_category,
                categories[0],
            )

            shap_background[feature] = (
                background[feature]
                .astype(str)
                .map(category_to_code)
                .astype(float)
            )

            if patient_category not in category_to_code:
                raise ValueError(
                    f"Unknown category '{patient_category}' "
                    f"for feature '{feature}'."
                )

            shap_patient[feature] = [
                category_to_code[patient_category]
            ]

        else:
            shap_background[feature] = pd.to_numeric(
                background[feature],
                errors="coerce",
            ).astype(float)

            shap_patient[feature] = pd.to_numeric(
                patient_input[feature],
                errors="coerce",
            ).astype(float)

    shap_background = shap_background.dropna(
        how="any"
    ).reset_index(drop=True)

    dementia_index = _dementia_class_index(model)

    def decode_categorical_value(
        value,
        code_to_category,
        default_category,
    ):
        try:
            code = float(round(float(value)))
        except (TypeError, ValueError):
            return default_category

        return code_to_category.get(
            code,
            default_category,
        )

    def predict_dementia_probability(data):
        if isinstance(data, pd.DataFrame):
            encoded_input = data.copy()
        else:
            encoded_input = pd.DataFrame(
                data,
                columns=features,
            )

        model_input = encoded_input.copy()

        for feature, mapping in categorical_maps.items():
            code_to_category, default_category = mapping

            model_input[feature] = (
                model_input[feature]
                .apply(
                    lambda value: decode_categorical_value(
                        value,
                        code_to_category,
                        default_category,
                    )
                )
            )

        if model_input.empty:
            return pd.Series(dtype=float).to_numpy()

        probabilities = model.predict_proba(model_input)

        return probabilities[:, dementia_index]

    explainer = shap.KernelExplainer(
        predict_dementia_probability,
        shap_background.to_numpy(),
    )

    raw_shap_values = explainer.shap_values(
        shap_patient.to_numpy(),
        nsamples=200,
    )

    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[0]
    else:
        shap_values = raw_shap_values

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    patient_shap_values = shap_values[0]

    expected_value = explainer.expected_value

    if hasattr(expected_value, "__len__"):
        base_value = float(expected_value[0])
    else:
        base_value = float(expected_value)

    dementia_probability = float(
        predict_dementia_probability(shap_patient)[0]
    )

    local_shap_table = pd.DataFrame(
        {
            "feature": features,
            "patient_value": [
                patient_input.iloc[0][feature]
                for feature in features
            ],
            "shap_value": patient_shap_values,
        }
    )

    local_shap_table["absolute_shap_value"] = (
        local_shap_table["shap_value"].abs()
    )

    local_shap_table["direction"] = (
        local_shap_table["shap_value"]
        .apply(
            lambda value: (
                "Raises Demented probability"
                if value > 0
                else (
                    "Lowers Demented probability"
                    if value < 0
                    else "Neutral"
                )
            )
        )
    )

    local_shap_table = local_shap_table.sort_values(
        "absolute_shap_value",
        ascending=False,
    ).reset_index(drop=True)

    return (
        local_shap_table,
        base_value,
        dementia_probability,
    )

def _load_processed_background(
    model_name: str,
) -> pd.DataFrame:
    """
    Load the processed baseline predictor data for SHAP.

    Identifiers, target labels, diagnosis groups, and Visit are
    excluded because only the model predictors should be used as
    the SHAP background.
    """
    background_path = (
        PROJECT_ROOT
        / "02_processed_data"
        / "oasis2_binary_baseline.csv"
    )

    if not background_path.exists():
        raise FileNotFoundError(
            f"Processed background file not found: "
            f"{background_path}"
        )

    background_data = pd.read_csv(background_path)
    features = expected_features(model_name)

    if not features:
        raise ValueError(
            "The model's expected feature names could not be found."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in background_data.columns
    ]

    if missing_features:
        raise ValueError(
            "The processed background file is missing: "
            + ", ".join(missing_features)
        )

    background = background_data.loc[:, features].copy()

    for feature in features:
        numeric_values = pd.to_numeric(
            background[feature],
            errors="coerce",
        )

        non_missing_count = background[feature].notna().sum()

        if numeric_values.notna().sum() == non_missing_count:
            background[feature] = numeric_values

    background = background.dropna(
        how="any"
    ).reset_index(drop=True)

    if len(background) < 2:
        raise ValueError(
            "The processed background contains fewer than two "
            "complete observations."
        )

    return background

def calculate_local_shap_transformed(
    model_name: str,
    patient_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """
    Calculate local SHAP values on the fitted estimator after
    preprocessing, then aggregate transformed features back to the
    original predictor names.
    """
    import numpy as np
    import shap

    pipeline = load_model(model_name)

    if not hasattr(pipeline, "named_steps"):
        raise ValueError(
            "The saved model is not a fitted scikit-learn pipeline."
        )

    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]

    features = expected_features(model_name)

    if not features:
        raise ValueError(
            "The original model feature names could not be found."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in patient_dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "The patient input is missing these features: "
            + ", ".join(missing_features)
        )

    patient_raw = patient_dataframe.loc[:, features].copy()
    background_raw = _load_processed_background(model_name)

    background_transformed = preprocess.transform(
        background_raw
    )

    patient_transformed = preprocess.transform(
        patient_raw
    )

    if hasattr(background_transformed, "toarray"):
        background_transformed = (
            background_transformed.toarray()
        )

    if hasattr(patient_transformed, "toarray"):
        patient_transformed = (
            patient_transformed.toarray()
        )

    background_transformed = np.asarray(
        background_transformed,
        dtype=float,
    )

    patient_transformed = np.asarray(
        patient_transformed,
        dtype=float,
    )

    transformed_names = list(
        preprocess.get_feature_names_out()
    )

    if len(transformed_names) != background_transformed.shape[1]:
        raise ValueError(
            "The number of transformed feature names does not "
            "match the transformed data."
        )

    dementia_index = _dementia_class_index(estimator)

    def predict_dementia_probability(data):
        data = np.asarray(data, dtype=float)

        if data.shape[0] == 0:
            return np.empty(
                shape=(0,),
                dtype=float,
            )

        probabilities = estimator.predict_proba(data)

        return probabilities[:, dementia_index]

    explainer = shap.KernelExplainer(
        predict_dementia_probability,
        background_transformed,
    )

    raw_shap_values = explainer.shap_values(
        patient_transformed,
        nsamples=1000,
    )

    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) == 1:
            shap_values = raw_shap_values[0]
        else:
            shap_values = raw_shap_values[dementia_index]
    else:
        shap_values = raw_shap_values

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[-1] > 1:
            shap_values = shap_values[:, :, dementia_index]
        else:
            shap_values = shap_values[:, :, 0]

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    transformed_shap_values = shap_values[0]

    expected_value = np.asarray(
        explainer.expected_value
    ).reshape(-1)

    base_value = float(expected_value[0])

    model_probability = float(
        pipeline.predict_proba(patient_raw)[0, dementia_index]
    )

    reconstructed_probability = (
        base_value
        + float(transformed_shap_values.sum())
    )

    additivity_error = abs(
        reconstructed_probability
        - model_probability
    )

    if additivity_error > 0.05:
        raise ValueError(
            "The SHAP additivity check failed. "
            f"SHAP reconstruction was "
            f"{reconstructed_probability:.4f}, while the "
            f"model probability was "
            f"{model_probability:.4f}. "
            f"Absolute error: {additivity_error:.4f}."
        )

    def original_feature_name(
        transformed_name: str,
    ) -> str:
        if transformed_name.startswith("num__"):
            return transformed_name[len("num__"):]

        if transformed_name.startswith("num_"):
            return transformed_name[len("num_"):]

        if transformed_name.startswith("cat__"):
            category_name = transformed_name[len("cat__"):]
        elif transformed_name.startswith("cat_"):
            category_name = transformed_name[len("cat_"):]
        else:
            category_name = transformed_name

        for feature in features:
            if category_name == feature:
                return feature

            if category_name.startswith(
                f"{feature}_"
            ):
                return feature

        return category_name

    transformed_to_original = {
        transformed_name: original_feature_name(
            transformed_name
        )
        for transformed_name in transformed_names
    }

    explanation_rows = []

    for feature in features:
        matching_indices = [
            index
            for index, transformed_name in enumerate(
                transformed_names
            )
            if transformed_to_original[
                transformed_name
            ] == feature
        ]

        if not matching_indices:
            raise ValueError(
                f"No transformed SHAP feature mapped to "
                f"original feature '{feature}'."
            )

        contribution = float(
            transformed_shap_values[matching_indices].sum()
        )

        transformed_feature_list = [
            transformed_names[index]
            for index in matching_indices
        ]

        patient_value = patient_raw.iloc[0][feature]

        if contribution > 0:
            direction = "Raises Demented probability"
        elif contribution < 0:
            direction = "Lowers Demented probability"
        else:
            direction = "Neutral"

        explanation_rows.append(
            {
                "feature": feature,
                "patient_value": patient_value,
                "shap_value": contribution,
                "absolute_shap_value": abs(
                    contribution
                ),
                "direction": direction,
                "transformed_features": "; ".join(
                    transformed_feature_list
                ),
            }
        )

    local_shap_table = pd.DataFrame(
        explanation_rows
    )

    local_shap_table = local_shap_table.sort_values(
        "absolute_shap_value",
        ascending=False,
    ).reset_index(drop=True)

    return (
        local_shap_table,
        base_value,
        model_probability,
    )
