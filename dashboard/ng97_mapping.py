from __future__ import annotations

import pandas as pd


FEATURE_TO_NG97_ROWS = [
    {
        "feature": "MMSE",
        "ng97_domain": "Cognitive assessment",
        "relationship": "Cognitive-domain proxy",
        "mapping_interpretation": (
            "MMSE represents one cognitive-assessment dimension relevant "
            "to NG97. It is not the complete NICE cognitive assessment "
            "and is not one of the six named examples in recommendation "
            "1.2.3."
        ),
    },
    {
        "feature": "eTIV",
        "ng97_domain": "Structural brain imaging",
        "relationship": "Indirect imaging-related feature",
        "mapping_interpretation": (
            "eTIV is an MRI-derived measurement that may be related to "
            "the structural-imaging domain. It is not equivalent to a "
            "clinical radiological assessment and is not a NICE diagnostic "
            "criterion."
        ),
    },
    {
        "feature": "nWBV",
        "ng97_domain": "Structural brain imaging",
        "relationship": "Indirect imaging-related feature",
        "mapping_interpretation": (
            "nWBV is an MRI-derived measurement that may provide "
            "information related to structural brain characteristics. "
            "Its SHAP importance is not proof of a dementia subtype or "
            "pathological lesion."
        ),
    },
    {
        "feature": "ASF",
        "ng97_domain": "Structural brain imaging",
        "relationship": "Indirect imaging-related feature",
        "mapping_interpretation": (
            "ASF is an MRI-derived morphometric variable. It has an "
            "indirect relationship with structural imaging but is not "
            "itself a NICE-recommended diagnostic test."
        ),
    },
    {
        "feature": "Age",
        "ng97_domain": "Demographic and socioeconomic context",
        "relationship": "Contextual feature, not direct NG97 alignment",
        "mapping_interpretation": (
            "Age may influence the distribution of dementia-related "
            "characteristics and the interpretation of test results. "
            "It is not labelled as a direct NICE diagnostic feature."
        ),
    },
    {
        "feature": "M/F",
        "ng97_domain": "Demographic and socioeconomic context",
        "relationship": "Contextual feature, not direct NG97 alignment",
        "mapping_interpretation": (
            "Sex is included as a model predictor but is not classified "
            "as a direct NG97 assessment domain."
        ),
    },
    {
        "feature": "EDUC",
        "ng97_domain": "Demographic and socioeconomic context",
        "relationship": "Contextual feature, not direct NG97 alignment",
        "mapping_interpretation": (
            "Education may affect performance on cognitive measures, "
            "but the current mapping does not classify it as a NICE "
            "diagnostic feature."
        ),
    },
    {
        "feature": "SES",
        "ng97_domain": "Demographic and socioeconomic context",
        "relationship": "Contextual feature, not direct NG97 alignment",
        "mapping_interpretation": (
            "SES is treated as contextual information rather than as "
            "a direct NICE assessment or diagnostic variable."
        ),
    },
]


REQUIRED_SHAP_COLUMNS = {
    "feature",
    "patient_value",
    "shap_value",
    "absolute_shap_value",
    "direction",
}


def load_feature_ng97_mapping() -> pd.DataFrame:
    """Return the approved feature-to-NG97 conceptual crosswalk."""
    return pd.DataFrame(FEATURE_TO_NG97_ROWS).copy()


def join_local_shap_to_ng97(
    local_shap_table: pd.DataFrame,
) -> pd.DataFrame:
    """Join validated local SHAP values to the NG97 crosswalk."""
    missing_columns = sorted(
        REQUIRED_SHAP_COLUMNS.difference(local_shap_table.columns)
    )

    if missing_columns:
        raise ValueError(
            "The local SHAP table is missing required columns: "
            + ", ".join(missing_columns)
        )

    mapping = load_feature_ng97_mapping()

    if mapping["feature"].duplicated().any():
        raise ValueError(
            "The NG97 mapping contains duplicate features."
        )

    joined = local_shap_table.merge(
        mapping,
        on="feature",
        how="left",
        validate="one_to_one",
    )

    missing_mappings = joined.loc[
        joined["ng97_domain"].isna(),
        "feature",
    ].astype(str).tolist()

    if missing_mappings:
        raise ValueError(
            "No approved NG97 mapping was found for: "
            + ", ".join(missing_mappings)
        )

    joined["mapping_status"] = (
        "Conceptual crosswalk only"
    )

    joined["clinical_use_status"] = (
        "Not a clinical finding, recommendation, "
        "or diagnostic rule"
    )

    return joined


def add_local_interpretation(
    mapped_shap_table: pd.DataFrame,
) -> pd.DataFrame:
    """Add cautious wording about the patient-specific contribution."""
    output = mapped_shap_table.copy()

    def describe(row: pd.Series) -> str:
        feature = row["feature"]
        direction = row["direction"]

        if direction == "Raises Demented probability":
            movement = "raises"
        elif direction == "Lowers Demented probability":
            movement = "lowers"
        else:
            movement = "does not materially change"

        return (
            f"For this model and input, {feature} {movement} "
            "the model's Demented probability. This is a model "
            "contribution, not a clinical conclusion."
        )

    output["patient_model_interpretation"] = output.apply(
        describe,
        axis=1,
    )

    return output


def prepare_mapped_local_shap_table(
    local_shap_table: pd.DataFrame,
) -> pd.DataFrame:
    """Validate, join and annotate a local SHAP table."""
    joined = join_local_shap_to_ng97(local_shap_table)
    return add_local_interpretation(joined)
