from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str


MODEL_FILES = (
    "logistic_regression_best_pipeline.joblib",
    "random_forest_best_pipeline.joblib",
    "svc_best_pipeline.joblib",
)


TABLE_FILES = (
    "table_descriptive_statistics.csv",
    "table_missing_values.csv",
    "table_pairwise_delong_auc.csv",
    "table_shap_feature_importance_original.csv",
    "table_shap_feature_importance_transformed.csv",
    "table_shap_feature_ranking_consensus.csv",
    "table_shap_values_long.csv",
    "table_test_predictions.csv",
    "table_test_set_performance.csv",
    "table_tuning_results.csv",
)


def project_root() -> Path:
    """Return the dissertation project root."""
    return Path(__file__).resolve().parents[1]


def models_directory() -> Path:
    """Return the model directory."""
    return project_root() / "06_models"


def tables_directory() -> Path:
    """Return the table directory."""
    return project_root() / "05_tables"


def validate_required_files(
    directory: Path,
    filenames: Iterable[str],
    category: str,
) -> list[ValidationResult]:
    """Check that expected files exist and are non-empty."""
    results: list[ValidationResult] = []

    for filename in filenames:
        path = directory / filename

        if not path.exists():
            results.append(
                ValidationResult(
                    name=f"{category}: {filename}",
                    passed=False,
                    message=f"Missing file: {path}",
                )
            )

        elif path.stat().st_size == 0:
            results.append(
                ValidationResult(
                    name=f"{category}: {filename}",
                    passed=False,
                    message="File exists but is empty.",
                )
            )

        else:
            results.append(
                ValidationResult(
                    name=f"{category}: {filename}",
                    passed=True,
                    message="File exists and is non-empty.",
                )
            )

    return results


def validate_csv_files() -> list[ValidationResult]:
    """Check that expected CSV files can be read."""
    results: list[ValidationResult] = []

    for filename in TABLE_FILES:
        path = tables_directory() / filename

        if not path.exists():
            continue

        try:
            dataframe = pd.read_csv(path)

            if dataframe.empty:
                results.append(
                    ValidationResult(
                        name=f"CSV readable: {filename}",
                        passed=False,
                        message=(
                            "CSV loaded successfully but contains "
                            "no rows."
                        ),
                    )
                )

            else:
                results.append(
                    ValidationResult(
                        name=f"CSV readable: {filename}",
                        passed=True,
                        message=(
                            f"Loaded {dataframe.shape[0]} rows and "
                            f"{dataframe.shape[1]} columns."
                        ),
                    )
                )

        except Exception as error:
            results.append(
                ValidationResult(
                    name=f"CSV readable: {filename}",
                    passed=False,
                    message=f"Could not read CSV: {error}",
                )
            )

    return results


def validate_model_files() -> list[ValidationResult]:
    """Check that saved model files can be loaded."""
    results: list[ValidationResult] = []

    for filename in MODEL_FILES:
        path = models_directory() / filename

        if not path.exists():
            continue

        try:
            model = joblib.load(path)

            results.append(
                ValidationResult(
                    name=f"Model loadable: {filename}",
                    passed=True,
                    message=(
                        f"Loaded object of type "
                        f"{type(model).__name__}."
                    ),
                )
            )

        except Exception as error:
            results.append(
                ValidationResult(
                    name=f"Model loadable: {filename}",
                    passed=False,
                    message=f"Could not load model: {error}",
                )
            )

    return results


def run_validation() -> list[ValidationResult]:
    """Run all dashboard artefact checks."""
    results: list[ValidationResult] = []

    results.extend(
        validate_required_files(
            models_directory(),
            MODEL_FILES,
            "Model",
        )
    )

    results.extend(
        validate_required_files(
            tables_directory(),
            TABLE_FILES,
            "Table",
        )
    )

    results.extend(validate_csv_files())
    results.extend(validate_model_files())

    return results


def print_validation_report() -> int:
    """Print the validation report."""
    results = run_validation()

    print("=" * 72)
    print("OASIS-2 dashboard artefact validation")
    print("=" * 72)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.message}")

    passed_count = sum(result.passed for result in results)
    failed_count = len(results) - passed_count

    print("=" * 72)
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("=" * 72)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(print_validation_report())
