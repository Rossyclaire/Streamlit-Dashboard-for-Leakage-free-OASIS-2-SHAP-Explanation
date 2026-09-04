import hashlib
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import json
import joblib
import shap

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"The `probability` parameter was deprecated.*",
    category=FutureWarning,
    module=r"sklearn\.svm\._base",
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
    RandomizedSearchCV,
    ParameterGrid,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



def sha256_file(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()



RANDOM_STATE = 42

# The Python file is inside the code folder.
# The project root is therefore one level above it.
BASE = Path(__file__).resolve().parents[1]

RAW = BASE / "01_raw_data"
PROCESSED = BASE / "02_processed_data"
FIGURES = BASE / "04_figures"
TABLES = BASE / "05_tables"
MODELS = BASE / "06_models"
LOGS = BASE / "07_logs_and_notes"

for folder in [PROCESSED, FIGURES, TABLES, MODELS, LOGS]:
    folder.mkdir(parents=True, exist_ok=True)

print("Project root:", BASE)
print("Raw-data folder:", RAW)


# 02_load_and_validate.py
DATA_FILE = RAW / "oasis2_longitudinal_demographic_data.xlsx"
assert DATA_FILE.exists(), f"Data file not found: {DATA_FILE}"

# pandas reads the cached displayed values of the Visit formulas in the workbook.
df_raw = pd.read_excel(DATA_FILE)
print("Raw shape:", df_raw.shape)
print(df_raw.head())
print(df_raw.columns.tolist())
df_raw.info()

required = [
    "Subject ID", "MRI ID", "Group", "Visit", "MR Delay", "M/F",
    "Hand", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"
]

df = df_raw.copy()
df.columns = [str(c).strip() for c in df.columns]
missing_required = sorted(set(required) - set(df.columns))
assert not missing_required, f"Missing columns: {missing_required}"

# Make the fields used by the analysis explicit and robust to spreadsheet typing.
df["Visit"] = pd.to_numeric(df["Visit"], errors="coerce")
df["MR Delay"] = pd.to_numeric(df["MR Delay"], errors="coerce")
for col in ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Enforce a stable row order before filtering and splitting.
# This prevents changes in source-file row order from changing
# the train/test assignment.
df = (
    df.sort_values(
        ["Subject ID", "Visit", "MRI ID"],
        kind="mergesort"
    )
    .reset_index(drop=True)
)

print("Groups:")
print(df["Group"].value_counts(dropna=False))
print("Unique subjects:", df["Subject ID"].nunique())
print("Visits:")
print(df["Visit"].value_counts(dropna=False).sort_index())
print("Missing values:")
print(df.isna().sum().sort_values(ascending=False))

# The supplied workbook should have 373 data rows and 15 source columns.
assert df.shape[1] == 15
assert df["Subject ID"].notna().all()
assert df["Group"].notna().all()


# 03_prepare_binary_baseline.py
# Converted participants are excluded from this binary analysis.
df_binary = df[df["Group"].isin(["Demented", "Nondemented"])].copy()
df_binary["Target"] = df_binary["Group"].map({
    "Nondemented": 0,
    "Demented": 1,
})

assert df_binary["Target"].notna().all()
print("Binary shape:", df_binary.shape)
print(df_binary["Group"].value_counts())
print(df_binary["Target"].value_counts().sort_index())

# Use Visit 1 as the baseline record. This creates one row per subject.
df_base = df_binary[df_binary["Visit"] == 1].copy()

# Enforce a stable subject order before train_test_split.
df_base = (
    df_base.sort_values(
        "Subject ID",
        kind="mergesort"
    )
    .reset_index(drop=True)
)

subject_counts = df_base["Subject ID"].value_counts()
assert subject_counts.max() == 1, "More than one baseline row exists for a subject"
assert df_base["Subject ID"].nunique() == len(df_base), "Duplicate subject rows remain"

df_base["target"] = df_base["Group"].map({
    "Nondemented": 0,
    "Demented": 1,
})
assert df_base["target"].notna().all()

print("Baseline shape:", df_base.shape)
print("Baseline class counts:")
print(df_base["Group"].value_counts())

feature_columns = [
    "M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"
]
target_column = "target"
group_column = "Subject ID"

# Leakage guard. CDR is not a model input because the label is derived from CDR.
assert "CDR" not in feature_columns
assert "Group" not in feature_columns
assert "Subject ID" not in feature_columns
assert "MRI ID" not in feature_columns
assert "Visit" not in feature_columns
assert "MR Delay" not in feature_columns
assert "Hand" not in feature_columns

X = df_base[feature_columns].copy()
y = df_base[target_column].copy()
groups = df_base[group_column].copy()
print("X shape:", X.shape)
print("Target counts:", y.value_counts().sort_index().to_dict())
print("Features:", feature_columns)

processed_columns = [group_column, "MRI ID", "Group", "Visit"] + feature_columns + [target_column]
df_base[processed_columns].to_csv(PROCESSED / "oasis2_binary_baseline.csv", index=False)
with open(PROCESSED / "feature_definition.json", "w", encoding="utf-8") as f:
    json.dump({
        "target": "Group mapped to Nondemented=0, Demented=1",
        "excluded_group": "Converted",
        "timepoint": "Visit 1",
        "features": feature_columns,
        "excluded_from_features": [
            "CDR", "Group", "Subject ID", "MRI ID", "Visit", "MR Delay", "Hand"
        ],
    }, f, indent=2)


# 04_descriptive_analysis.py
numeric_features = ["Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]

summary_table = df_base.groupby("Group")[numeric_features].agg(
    ["count", "mean", "std", "median", "min", "max"]
)
summary_table.to_csv(TABLES / "table_descriptive_statistics.csv")

missing_table = pd.DataFrame({
    "missing_count": df_base[feature_columns].isna().sum(),
    "missing_percent": df_base[feature_columns].isna().mean() * 100,
}).sort_values("missing_count", ascending=False)
missing_table.to_csv(TABLES / "table_missing_values.csv")

# Class-count figure.
plt.figure(figsize=(7, 5))
sns.countplot(data=df_base, x="Group", order=["Nondemented", "Demented"])
plt.title("Baseline class counts")
plt.xlabel("Group")
plt.ylabel("Number of subjects")
plt.tight_layout()
plt.savefig(FIGURES / "fig_class_counts.png", dpi=300, bbox_inches="tight")
plt.close()

# Numeric feature boxplots by diagnostic group.
long_df = df_base.melt(
    id_vars=["Group"],
    value_vars=numeric_features,
    var_name="Feature",
    value_name="Value",
)
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()
for ax, feature in zip(axes, numeric_features):
    sns.boxplot(data=long_df[long_df["Feature"] == feature], x="Group", y="Value", ax=ax)
    ax.set_title(feature)
    ax.tick_params(axis="x", rotation=20)
for ax in axes[len(numeric_features):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(FIGURES / "fig_feature_boxplots.png", dpi=300, bbox_inches="tight")
plt.close()

# Correlation matrix for baseline numeric variables and target.
plt.figure(figsize=(9, 7))
corr = df_base[numeric_features + ["target"]].corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation matrix for baseline numeric variables")
plt.tight_layout()
plt.savefig(FIGURES / "fig_correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("Descriptive tables and figures saved")   


# 05_split_and_group_checks.py
train_df, test_df = train_test_split(
    df_base,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df_base["target"],
)

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()
y_train = train_df["target"].copy()
y_test = test_df["target"].copy()
groups_train = train_df["Subject ID"].copy()
groups_test = test_df["Subject ID"].copy()

print("Train rows:", len(train_df))
print("Test rows:", len(test_df))
print("Train classes:", y_train.value_counts().to_dict())
print("Test classes:", y_test.value_counts().to_dict())

# Confirm the held-out subjects are not present in the training partition.
overlap = set(groups_train).intersection(set(groups_test))
print("Subjects in both train and test:", overlap)
assert len(overlap) == 0, "Subject leakage detected"
print("Leakage check passed")

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# Confirm that every CV fold also keeps subjects disjoint.
for fold, (fold_train_idx, fold_valid_idx) in enumerate(
    cv.split(X_train, y_train, groups=groups_train), start=1
):
    fold_groups_train = groups_train.iloc[fold_train_idx]
    fold_groups_valid = groups_train.iloc[fold_valid_idx]
    assert set(fold_groups_train).isdisjoint(set(fold_groups_valid))
    print(
        f"Fold {fold}: train={len(fold_train_idx)}, "
        f"validation={len(fold_valid_idx)}"
    )
print("All group-aware CV checks passed")


# 06_preprocess_tune_and_save.py
categorical_features = ["M/F"]
numeric_features_for_model = [
    "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features_for_model),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ]),
    "Random Forest": Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
    ]),
    "SVC": Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", SVC(probability=True, random_state=RANDOM_STATE)),
    ]),
}

param_distributions = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__class_weight": ["balanced"],
    },
    "Random Forest": {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 3, 5, 10],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": ["balanced"],
    },
    "SVC": {
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
        "model__class_weight": ["balanced"],
    },
}

search_results = {}
for name, pipeline in models.items():
    print(f"Starting: {name}")

    n_iter = min(
        12,
        len(list(ParameterGrid(param_distributions[name])))
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions[name],
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train, groups=groups_train)
    search_results[name] = search
    print("Best CV ROC AUC:", search.best_score_)
    print("Best parameters:", search.best_params_)

# Save the tuning summary.
tuning_table = pd.DataFrame([
    {
        "model": name,
        "best_cv_roc_auc": search.best_score_,
        "best_parameters": json.dumps(search.best_params_, default=str),
    }
    for name, search in search_results.items()
])
tuning_table.to_csv(TABLES / "table_tuning_results.csv", index=False)

# Save each fitted, preprocessing-plus-model pipeline.
for name, search in search_results.items():
    safe_name = name.lower().replace(" ", "_")
    output_file = MODELS / f"{safe_name}_best_pipeline.joblib"
    joblib.dump(search.best_estimator_, output_file)
    print("Saved:", output_file)


# 07_evaluate_test_set.py
def evaluate_model(name, fitted_search, X_test, y_test):
    model = fitted_search.best_estimator_
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_weighted": f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    ),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    total_observations = tn + fp + fn + tp

    assert total_observations == len(y_test), (
    "Confusion-matrix total does not equal test-set size"
    )

    derived_accuracy = (tn + tp) / total_observations

    assert np.isclose(
        metrics["accuracy"],
        derived_accuracy
    ), (
        f"Accuracy mismatch for {name}: "
        f"reported={metrics['accuracy']}, "
        f"derived={derived_accuracy}"
    )

    return metrics, y_pred, y_prob

all_metrics = []
predictions = {}
for name, search in search_results.items():
    metrics, y_pred, y_prob = evaluate_model(name, search, X_test, y_test)
    all_metrics.append(metrics)
    predictions[name] = {"y_pred": y_pred, "y_prob": y_prob}
    print("\n" + "=" * 70)
    print(name)
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Nondemented", "Demented"],
        zero_division=0,
    ))

results_table = pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False)
results_table.to_csv(TABLES / "table_test_set_performance.csv", index=False)
print(results_table.to_string(index=False))

# Confusion matrices.
for name, values in predictions.items():
    safe_name = name.lower().replace(" ", "_")
    cm = confusion_matrix(y_test, values["y_pred"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Nondemented", "Demented"],
        yticklabels=["Nondemented", "Demented"],
    )
    plt.title(f"Confusion matrix: {name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(FIGURES / f"fig_confusion_matrix_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

# ROC and precision-recall curves on the same held-out test set.
fig, ax = plt.subplots(figsize=(7, 6))
for name, values in predictions.items():
    RocCurveDisplay.from_predictions(y_test, values["y_prob"], name=name, ax=ax)
ax.set_title("ROC curves on held-out test set")
plt.tight_layout()
plt.savefig(FIGURES / "fig_roc_curves_test_set.png", dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
for name, values in predictions.items():
    PrecisionRecallDisplay.from_predictions(y_test, values["y_prob"], name=name, ax=ax)
ax.set_title("Precision-recall curves on held-out test set")
plt.tight_layout()
plt.savefig(FIGURES / "fig_precision_recall_curves_test_set.png", dpi=300, bbox_inches="tight")
plt.close()

print("Evaluation tables and figures saved")


# 08_delong_pairwise_auc_check.py
# Pairwise ROC-AUC comparison on the held-out test-set probabilities.
from itertools import combinations
from scipy import stats

def compute_midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov

def delong_roc_test(y_true, score_a, score_b):
    """
    Return AUC A, AUC B, z statistic, and two-sided p-value.
    """
    y_true = np.asarray(y_true).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "The test set must contain both classes"
        )

    if not (len(y_true) == len(score_a) == len(score_b)):
        raise ValueError(
            "Labels and scores must have equal lengths"
        )

    positive_count = int(np.sum(y_true == 1))

    # Important: order observations by the true labels,
    # not by either model's prediction scores.
    order = np.argsort(-y_true)

    predictions = np.vstack(
        [score_a, score_b]
    )[:, order]

    aucs, covariance = fast_delong(
        predictions,
        positive_count
    )

    contrast = np.array([[1.0, -1.0]])

    variance = (
        contrast @ covariance @ contrast.T
    ).item()

    variance = max(
        variance,
        np.finfo(float).eps
    )

    z_value = (
        aucs[0] - aucs[1]
    ) / np.sqrt(variance)

    p_value = 2 * stats.norm.sf(abs(z_value))

    return aucs[0], aucs[1], z_value, p_value

from sklearn.metrics import roc_auc_score

pairwise_rows = []

model_scores = {
    name: values["y_prob"]
    for name, values in predictions.items()
}

for name_a, name_b in combinations(model_scores.keys(), 2):
    score_a = model_scores[name_a]
    score_b = model_scores[name_b]

    auc_a, auc_b, z_value, p_value = delong_roc_test(
        y_test,
        score_a,
        score_b
    )

    assert np.isclose(
        auc_a,
        roc_auc_score(y_test, score_a)
    ), f"AUC mismatch for {name_a}"

    assert np.isclose(
        auc_b,
        roc_auc_score(y_test, score_b)
    ), f"AUC mismatch for {name_b}"

    pairwise_rows.append({
        "model_a": name_a,
        "model_b": name_b,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_difference": auc_a - auc_b,
        "z_value": z_value,
        "p_value_raw": p_value,
    })

print("DeLong AUC consistency check passed")

pairwise_table = pd.DataFrame(pairwise_rows)

pairwise_table.to_csv(
    TABLES / "table_pairwise_delong_auc.csv",
    index=False
)

print(pairwise_table.to_string(index=False))


# 09_shap_explainability.py

def extract_positive_class_shap(shap_result):
    """
    Return a two-dimensional array with shape:
    n_test_subjects x n_transformed_features
    for the positive class, Demented = 1.
    """

    values = (
        shap_result.values
        if hasattr(shap_result, "values")
        else shap_result
    )

    if isinstance(values, list):
        # Older SHAP versions may return one array per class.
        if len(values) == 2:
            values = values[1]
        else:
            values = values[-1]

    values = np.asarray(values)

    if values.ndim == 3:
        # Typical shape:
        # observations x features x classes
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

    if values.ndim != 2:
        raise ValueError(
            f"Expected two-dimensional SHAP values, got {values.shape}"
        )

    return values

shap_outputs = {}
shap_rows = []
for name, search in search_results.items():
    model_pipeline = search.best_estimator_
    fitted_preprocessor = model_pipeline.named_steps["preprocess"]
    model = model_pipeline.named_steps["model"]

    X_train_transformed = fitted_preprocessor.transform(X_train)
    X_test_transformed = fitted_preprocessor.transform(X_test)
    feature_names = fitted_preprocessor.get_feature_names_out()

    X_train_dense = (
        X_train_transformed.toarray()
        if hasattr(X_train_transformed, "toarray")
        else X_train_transformed
    )
    X_test_dense = (
        X_test_transformed.toarray()
        if hasattr(X_test_transformed, "toarray")
        else X_test_transformed
    )

    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
    else:
        masker = shap.maskers.Independent(
            X_train_dense,
            max_samples=len(X_train)
        )
        explainer = shap.Explainer(
            model.predict_proba, 
            masker
        )

    shap_values = explainer(X_test_dense)
    positive_class_values = (
        extract_positive_class_shap(shap_values)
    )

    if positive_class_values.shape[1] != len(feature_names):
        raise ValueError(
            "The number of SHAP columns does not match "
            "the number of transformed feature names."
        )

    subject_ids = test_df["Subject ID"].astype(str).to_numpy()

    if len(subject_ids) != positive_class_values.shape[0]:
        raise ValueError(
            "The number of subject IDs does not match "
            "the number of SHAP rows."
        )

    for row_number, subject_id in enumerate(subject_ids):

        for feature_number, feature_name in enumerate(feature_names):

            shap_value = float(
                positive_class_values[
                    row_number,
                    feature_number
                ]
            )

            shap_rows.append({
                "model": name,
                "subject_id": subject_id,
                "test_row": row_number,
                "transformed_feature": feature_name,
                "shap_value": shap_value,
                "abs_shap_value": abs(shap_value),
            })

    shap_values.feature_names = list(feature_names)

    # Keep the complete SHAP object for the waterfall plots.
    shap_outputs[name] = (shap_values, feature_names)

    # Use the positive-class explanation for the summary plot.
    # This converts a binary multi-output explanation to a
    # two-dimensional feature-by-observation explanation.
    if len(shap_values.shape) == 3:
        summary_values = shap_values[:, :, 1]
    else:
        summary_values = shap_values

    summary_values.feature_names = list(feature_names)

    plt.close("all")
 
    shap.summary_plot(
        summary_values,
        X_test_dense,
        feature_names=feature_names,
        show=False,
    )

    fig = plt.gcf()
    fig.suptitle(
        f"SHAP summary: {name}",
        y=0.99,
        fontsize=13,
    )

    fig.tight_layout(
        rect=[0, 0.08, 1, 0.94]
    )

    safe_name = name.lower().replace(" ", "_")

    fig.savefig(
        FIGURES / f"fig_shap_summary_{safe_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# Generate one local waterfall plot for each model
for name, (shap_values, feature_names) in shap_outputs.items():

    if len(shap_values.shape) == 3:
        # Multi-output binary classification
        local_explanation = shap_values[0, :, 1]
    else:
        # Single-output explanation
        local_explanation = shap_values[0]

    plt.close("all")

    shap.plots.waterfall(
        local_explanation,
        max_display=10,
        show=False,
    )

    fig = plt.gcf()

    fig.suptitle(
        f"Illustrative local SHAP explanation: {name}\n"
        "Example held-out test subject; "
        "positive class: Demented = 1",
        fontsize=12,
        y=0.99,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.92]
    )

    safe_name = name.lower().replace(" ", "_")

    fig.savefig(
        FIGURES / f"fig_shap_waterfall_{safe_name}_example.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
print("SHAP figures saved")

shap_values_table = pd.DataFrame(shap_rows)

shap_values_file = (
    TABLES / "table_shap_values_long.csv"
)

shap_values_table.to_csv(
    shap_values_file,
    index=False
)

assert not shap_values_table.empty

assert shap_values_table["shap_value"].notna().all()

assert shap_values_table["abs_shap_value"].notna().all()

assert np.isclose(
    shap_values_table["abs_shap_value"],
    shap_values_table["shap_value"].abs()
).all()

assert set(
    shap_values_table["model"]
) == set(search_results.keys())

assert set(
    shap_values_table["transformed_feature"]
) == set(feature_names)

print(
    "Saved numerical SHAP values:",
    shap_values_file
)

print(
    "Rows saved:",
    len(shap_values_table)
)

# Calculate mean absolute SHAP importance
# for every transformed feature within each model.

shap_importance_transformed = (
    shap_values_table
    .groupby(
        [
            "model",
            "transformed_feature",
        ],
        as_index=False
    )
    .agg(
        mean_abs_shap=(
            "abs_shap_value",
            "mean"
        ),
        n_test_subjects=(
            "subject_id",
            "nunique"
        ),
    )
)

shap_importance_transformed = (
    shap_importance_transformed
    .sort_values(
        [
            "model",
            "mean_abs_shap",
        ],
        ascending=[
            True,
            False,
        ]
    )
)

shap_importance_file = (
    TABLES
    / "table_shap_feature_importance_transformed.csv"
)

shap_importance_transformed.to_csv(
    shap_importance_file,
    index=False
)

print(
    "Saved transformed-feature SHAP importance:",
    shap_importance_file
)

original_predictors = [
    "M/F",
    "Age",
    "EDUC",
    "SES",
    "MMSE",
    "eTIV",
    "nWBV",
    "ASF",
]

def map_transformed_to_original(
    transformed_feature,
    original_predictors
):
    """
    Map a transformed feature name back to its
    original predictor name.
    """

    feature_name = str(transformed_feature)

    # Remove preprocessing prefixes such as:
    # num__ and cat__
    if "__" in feature_name:
        feature_name = feature_name.split(
            "__",
            1
        )[1]

    # Match the longest original name first.
    # This prevents partial matches.
    for original_feature in sorted(
        original_predictors,
        key=len,
        reverse=True
    ):
        if (
            feature_name == original_feature
            or feature_name.startswith(
                f"{original_feature}_"
            )
        ):
            return original_feature

    raise ValueError(
        "Could not map transformed feature "
        f"'{transformed_feature}' to an original predictor."
    )

shap_values_with_original = (
    shap_values_table.copy()
)

shap_values_with_original[
    "original_feature"
] = (
    shap_values_with_original[
        "transformed_feature"
    ]
    .map(
        lambda feature_name:
        map_transformed_to_original(
            feature_name,
            original_predictors
        )
    )
)

subject_level_original_shap = (
    shap_values_with_original
    .groupby(
        [
            "model",
            "subject_id",
            "original_feature",
        ],
        as_index=False
    )
    .agg(
        absolute_shap_sum=(
            "abs_shap_value",
            "sum"
        )
    )
)

shap_importance_original = (
    subject_level_original_shap
    .groupby(
        [
            "model",
            "original_feature",
        ],
        as_index=False
    )
    .agg(
        mean_abs_shap=(
            "absolute_shap_sum",
            "mean"
        ),
        n_test_subjects=(
            "subject_id",
            "nunique"
        ),
    )
)

shap_importance_original = (
    shap_importance_original
    .sort_values(
        [
            "model",
            "mean_abs_shap",
        ],
        ascending=[
            True,
            False,
        ]
    )
)

# Number 4: rank original predictors within each model.
shap_importance_original[
    "rank_within_model"
] = (
    shap_importance_original
    .groupby("model")["mean_abs_shap"]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)


# Re-sort using the new rank column.
shap_importance_original = (
    shap_importance_original
    .sort_values(
        [
            "model",
            "rank_within_model",
            "original_feature",
        ],
        ascending=[
            True,
            True,
            True,
        ]
    )
)

shap_importance_original_file = (
    TABLES
    / "table_shap_feature_importance_original.csv"
)

shap_importance_original.to_csv(
    shap_importance_original_file,
    index=False
)

print(
    "Saved original-predictor SHAP importance:",
    shap_importance_original_file
)

# Create a cross-model ranking and a consistency table.
rank_wide = (
    shap_importance_original
    .pivot(
        index="original_feature",
        columns="model",
        values="rank_within_model"
    )
    .reset_index()
)

model_rank_columns = [
    model_name
    for model_name in search_results.keys()
    if model_name in rank_wide.columns
]

if set(model_rank_columns) != set(
    search_results.keys()
):
    raise ValueError(
        "The consensus table does not contain "
        "all expected models."
    )

rank_wide["mean_within_model_rank"] = (
    rank_wide[model_rank_columns]
    .mean(axis=1)
)

rank_wide["rank_standard_deviation"] = (
    rank_wide[model_rank_columns]
    .std(
        axis=1,
        ddof=0
    )
)

rank_wide["top_rank_count"] = (
    rank_wide[model_rank_columns]
    .eq(1)
    .sum(axis=1)
)

rank_wide["consensus_rank"] = (
    rank_wide["mean_within_model_rank"]
    .rank(
        method="min",
        ascending=True
    )
    .astype(int)
)

shap_consensus_table = (
    rank_wide
    .sort_values(
        [
            "consensus_rank",
            "mean_within_model_rank",
            "original_feature",
        ],
        ascending=[
            True,
            True,
            True,
        ]
    )
)

shap_consensus_file = (
    TABLES
    / "table_shap_feature_ranking_consensus.csv"
)

shap_consensus_table.to_csv(
    shap_consensus_file,
    index=False
)

print(
    "Saved cross-model SHAP ranking:",
    shap_consensus_file
)


# 10_final_manifest.py
# Save exact held-out test predictions and probabilities.
prediction_table = pd.DataFrame({
    "Subject ID": test_df["Subject ID"].astype(str).to_numpy(),
    "true_target": y_test.to_numpy(),
})

for name, values in predictions.items():
    safe_name = name.lower().replace(" ", "_")

    prediction_table[f"{safe_name}_prediction"] = values["y_pred"]
    prediction_table[f"{safe_name}_probability"] = values["y_prob"]

prediction_file = TABLES / "table_test_predictions.csv"
prediction_table.to_csv(prediction_file, index=False)


# Collect fingerprint data
package_names = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "shap",
    "joblib",
    "matplotlib",
    "seaborn",
    "openpyxl",
]

package_versions = {
    package: version(package)
    for package in package_names
}

script_file = Path(__file__).resolve()

fingerprint_payload = {
    "source_file_sha256": sha256_file(DATA_FILE),
    "script_file_sha256": sha256_file(script_file),
    "prediction_file_sha256": sha256_file(prediction_file),
    "python_version": platform.python_version(),
    "package_versions": package_versions,
    "random_state": RANDOM_STATE,
    "train_subject_ids": sorted(
        train_df["Subject ID"].astype(str).tolist()
    ),
    "test_subject_ids": sorted(
        test_df["Subject ID"].astype(str).tolist()
    ),
    "best_parameters": {
        name: search.best_params_
        for name, search in search_results.items()
    },
}


# Convert the fingerprint data into one fingerprint value
fingerprint_text = json.dumps(
    fingerprint_payload,
    sort_keys=True,
    separators=(",", ":"),
    default=str,
)

run_fingerprint = hashlib.sha256(
    fingerprint_text.encode("utf-8")
).hexdigest()


# Create the existing manifest
manifest = {
    "python_version": platform.python_version(),
    "random_state": RANDOM_STATE,
    "data_file": str(DATA_FILE.relative_to(BASE)),
    "raw_shape": list(df_raw.shape),
    "binary_shape": list(df_binary.shape),
    "baseline_shape": list(df_base.shape),
    "feature_columns": feature_columns,
    "target_mapping": {"Nondemented": 0, "Demented": 1},
    "excluded_group": "Converted",
    "baseline_visit": 1,
    "test_size": 0.20,
    "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
    "group_column": "Subject ID",
    "models": list(search_results.keys()),

    "run_fingerprint": {
        "algorithm": "SHA-256",
        "value": run_fingerprint,
    },

    "fingerprint_inputs": fingerprint_payload,

    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
}
with open(LOGS / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

print("Project location:", BASE)
print("Saved figures:")
for file in sorted(FIGURES.glob("*.png")):
    print(" ", file.relative_to(BASE))
print("Saved tables:")
for file in sorted(TABLES.glob("*.csv")):
    print(" ", file.relative_to(BASE))
print("Saved models:")
for file in sorted(MODELS.glob("*.joblib")):
    print(" ", file.relative_to(BASE))