# OASIS-2 Dementia Comparative Analysis

## Purpose

This project compares three machine-learning models for binary dementia classification using the OASIS-2 longitudinal demographic dataset.

The models are:

- Logistic Regression
- Random Forest
- Support Vector Classifier (SVC)

The analysis uses Visit 1 baseline records and evaluates all models on the same held-out test set.

This is a machine-learning comparison study and research demonstration. It is not a clinical diagnostic, triage or treatment system.

## Project access

The project is available through two equal access routes:

1. The GitHub repository.
2. The university OneDrive project folder.

Both locations should contain the same reviewed project files and documentation. Neither route is a substitute for the other.

The GitHub and OneDrive links are also provided in the dissertation appendix.

## Access route 1: GitHub

GitHub repository:

```text
https://github.com/Rossyclaire/Streamlit-Dashboard-for-Leakage-free-OASIS-2-SHAP-Explanation
```

Markers can open the repository link to inspect the source code, dashboard files, saved analytical outputs, README documentation, requirements file and master log.

### Downloading the repository

Install Git if it is not already installed. Then open a terminal in VS Code or use a system terminal and run:

```powershell
git clone https://github.com/Rossyclaire/Streamlit-Dashboard-for-Leakage-free-OASIS-2-SHAP-Explanation.git
```

Move into the downloaded repository folder:

```powershell
cd Streamlit-Dashboard-for-Leakage-free-OASIS-2-SHAP-Explanation
```

The repository folder name is based on the GitHub repository name. It may not be the same as the original local project-folder name.

## Access route 2: University OneDrive

The same reviewed project is available in the author's university OneDrive folder.

University OneDrive project link:

```text
[PASTE UNIVERSITY ONEDRIVE SHARING LINK HERE]
```

Markers or supervisors can use the OneDrive link to open or download the project files if they prefer OneDrive or experience difficulty using GitHub.

The OneDrive project is stored under:

```text
My files > Dissertation Projects > OASIS2_Dementia_Dissertation
```

The OneDrive folder should contain the same reviewed materials as the GitHub repository, including:

- The main OASIS-2 analysis script.
- The dashboard source files.
- The README documentation.
- The requirements file.
- Saved analytical tables.
- Saved figures.
- Saved model pipelines.
- Processed-data files required by the dashboard.
- The master log and notes documentation.
- The synthetic dashboard test CSV, if included.

The OneDrive link may require authentication with an authorised university account. Access permissions should be tested before final dissertation submission.

## Keeping GitHub and OneDrive consistent

GitHub and university OneDrive are equal access routes to the project.

Before final submission, confirm that both locations contain:

- The same dashboard files.
- The same README.
- The same requirements file.
- The same saved models, tables and figures.
- The same processed-data files required by the dashboard.
- The same master log and notes documentation.
- The same synthetic test data, if included.

If a change is made after the project is uploaded, update both locations or clearly record which location contains the latest version.

Do not include university account passwords, access tokens or other credentials in this README, GitHub repository or OneDrive folder.

## Environment

The verified analysis and dashboard environment uses:

- Python 3.12.10
- Streamlit 1.61.1
- Plotly 6.9.0

The remaining package versions are listed in `requirements.txt`.

## Installing the project

### Windows

From the downloaded project folder, create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

### macOS or Linux

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

The `requirements.txt` file includes the verified dashboard dependencies:

```text
streamlit==1.61.1
plotly==6.9.0
```

## Launching the dashboard

The dashboard must be started through Streamlit. Individual dashboard page files should not be launched directly with Python.

From the project root, run:

### Windows

```powershell
streamlit run dashboard\app.py
```

### macOS or Linux

```bash
streamlit run dashboard/app.py
```

Streamlit will display a local web address, normally:

```text
http://localhost:8501
```

Open that address in a web browser.

## Project structure

The recommended project structure is:

```text
OASIS2_Dementia_Dissertation/
│
├── 01_raw_data/
│   └── oasis2_longitudinal_demographic_data.xlsx
│
├── 02_processed_data/
│   ├── oasis2_binary_baseline.csv
│   └── feature_definition.json
│
├── 04_figures/
│
├── 05_tables/
│
├── 06_models/
│
├── 07_logs_and_notes/
│   └── run_manifest.json
│
├── code/
│   └── oasis2_binary_baseline.py
│
├── dashboard/
│   ├── app.py
│   ├── loaders.py
│   ├── ng97_mapping.py
│   ├── validators.py
│   └── pages/
│       ├── 1_Classifier_Comparison.py
│       ├── 2_SHAP_Visualisations.py
│       ├── 3_Patient_Risk_Scoring.py
│       ├── 4_Result_Tables.py
│       └── 5_Existing_Figures.py
│
├── README.md
└── requirements.txt
```

The following items should not be committed to GitHub or uploaded for marker review:

```text
.venv/
__pycache__/
*.pyc
```

The original OASIS-2 workbook should only be stored or shared where permitted by the applicable dataset access conditions.

## Input data

To rerun the complete analysis, place the original OASIS-2 workbook at:

```text
01_raw_data/oasis2_longitudinal_demographic_data.xlsx
```

The expected source columns are:

```text
Subject ID
MRI ID
Group
Visit
MR Delay
M/F
Hand
Age
EDUC
SES
MMSE
CDR
eTIV
nWBV
ASF
```

The analysis stops if a required source column is missing.

The raw OASIS-2 workbook is required to rerun the analysis. It is not required to inspect the dashboard when the saved models, tables, figures and processed background data are available.

## Analysis scope

The analysis:

1. Loads and validates the OASIS-2 workbook.
2. Cleans column names and converts numeric fields.
3. Sorts records into a stable order.
4. Excludes participants in the Converted group.
5. Retains Visit 1 as the baseline record.
6. Creates a binary target:
   - Nondemented = 0
   - Demented = 1
7. Excludes variables that could cause diagnostic-label leakage.
8. Creates a stratified training and held-out test split.
9. Checks for subject overlap between the partitions.
10. Uses grouped stratified cross-validation during model tuning.
11. Tunes Logistic Regression, Random Forest and SVC.
12. Evaluates the best fitted version of each model on the held-out test set.
13. Generates descriptive tables, performance tables, figures, SHAP outputs and pairwise DeLong comparisons.
14. Saves a run manifest and SHA-256 fingerprint for reproducibility.

## Data preparation

The raw workbook is loaded into a separate dataframe called `df_raw`.

The working data is processed as follows:

- Column names are converted to strings and stripped of surrounding spaces.
- Visit and MR Delay are converted to numeric values.
- Age, EDUC, SES, MMSE, CDR, eTIV, nWBV and ASF are converted to numeric values.
- Records are sorted by Subject ID, Visit and MRI ID.
- The sorted index is reset before filtering and splitting.

Stable sorting helps ensure that changes in the original workbook order do not silently alter the train/test assignment.

## Binary outcome definition

Participants in the Converted group are excluded from the binary classification analysis.

The remaining groups are mapped as follows:

```text
Nondemented = 0
Demented = 1
```

Only Visit 1 records are retained for the baseline analysis.

The baseline data is checked to confirm that each subject contributes no more than one baseline record.

## Model predictors

The eight model predictors are:

```text
M/F
Age
EDUC
SES
MMSE
eTIV
nWBV
ASF
```

The following variables are excluded from the model inputs:

```text
CDR
Group
Subject ID
MRI ID
Visit
MR Delay
Hand
```

CDR is excluded because the diagnostic grouping is derived from clinically related CDR information. Including CDR as a predictor would create a risk of diagnostic-label leakage.

Subject ID is retained only for grouping and overlap checks. Group defines the target and is not used as a predictor.

## Data splitting

The baseline data is divided using a stratified 80:20 split:

```python
test_size = 0.20
random_state = 42
```

The held-out test set contains 28 records, while the training set contains 108 records.

The split is stratified using the binary target. The script checks that no subject appears in both the training and held-out test partitions.

The held-out test set is not used for hyperparameter selection.

## Preprocessing

Preprocessing is included inside each model pipeline.

### Numeric variables

Numeric variables are processed using:

1. Median imputation.
2. Standard scaling.

### Categorical variables

The `M/F` variable is processed using:

1. Most-frequent-value imputation.
2. One-hot encoding.
3. Ignoring unknown categories during transformation.

Keeping preprocessing inside each pipeline ensures that imputation, scaling and encoding are fitted within the training and cross-validation workflow.

## Cross-validation

Model tuning uses five-fold `StratifiedGroupKFold` with:

```text
n_splits = 5
shuffle = True
random_state = 42
```

The groups are based on Subject ID.

The script checks that subjects in each training fold and validation fold are disjoint before fitting the search procedure.

## Models

### Logistic Regression

The Logistic Regression model uses:

```text
max_iter = 5000
random_state = 42
class_weight = "balanced"
```

The regularisation parameter `C` is tuned.

### Random Forest

The Random Forest model uses:

```text
random_state = 42
n_jobs = 1
class_weight = "balanced"
```

The tuned parameters include:

```text
n_estimators
max_depth
min_samples_leaf
max_features
class_weight
```

One worker is used to support reproducibility and avoid unnecessary parallel execution.

### Support Vector Classifier

The SVC model uses:

```text
probability = True
random_state = 42
class_weight = "balanced"
```

The tuned parameters include:

```text
C
kernel
gamma
class_weight
```

Probability estimates are enabled because ROC-AUC and precision-recall AUC are calculated from predicted probabilities.

## Hyperparameter tuning

The analysis uses `RandomizedSearchCV` with:

```text
scoring = "roc_auc"
n_jobs = 1
random_state = 42
refit = True
return_train_score = True
```

The search is limited to a maximum of 12 iterations.

`ParameterGrid` is used to count finite parameter combinations before each search. This prevents the search from requesting more iterations than exist in a small finite grid.

`ParameterGrid` does not replace `RandomizedSearchCV`, alter the model or change the scoring method.

## Evaluation metrics

Each best fitted pipeline is evaluated on the held-out test set.

The saved performance table includes:

- Accuracy
- Precision
- Recall
- Weighted F1 score
- ROC-AUC
- Precision-recall AUC
- True negatives
- False positives
- False negatives
- True positives

The script checks that the confusion-matrix counts are internally consistent with the test-set size and accuracy.

The results are reported as benchmark results for one held-out split, not as definitive clinical performance estimates.

## Repeating the main analysis

The original OASIS-2 workbook must be available in the expected raw-data location before rerunning the analysis.

From the project root, activate the virtual environment and run:

### Windows

```powershell
python code\oasis2_binary_baseline.py
```

### macOS or Linux

```bash
python3 code/oasis2_binary_baseline.py
```

The script creates or updates the processed data, tables, figures, fitted models and run-manifest outputs.

Rerunning the analysis may overwrite files with the same names. A separate timestamped archive or run identifier is required to preserve a complete history of multiple runs.

## SHAP explainability

The analysis generates:

- Global SHAP summary figures.
- Local SHAP waterfall examples.
- Long-format numerical SHAP values.
- Transformed-feature mean absolute SHAP importance.
- Original-predictor aggregated SHAP importance.
- Within-model feature rankings.
- Cross-model ranking-consensus results.

SHAP values describe model behaviour for the selected dataset and fitted model. They do not establish causality, diagnostic necessity, treatment relevance or clinical importance.

The dashboard can also calculate an interactive patient-specific SHAP explanation. This explanation is recalculated using the saved fitted pipeline and processed baseline observations as the SHAP background dataset.

The interactive explanation is separate from the saved SHAP tables and figures generated by the main analysis script.

## Streamlit dashboard pages

The dashboard reads saved models, tables and figures. It does not retrain the models or alter the original held-out evaluation.

### Classifier Comparison

This page provides:

- Interactive comparison of the three classifiers.
- Selection of accuracy, precision, recall, weighted F1, ROC-AUC or PR-AUC.
- Display of the held-out performance table.
- Selectable confusion matrices.
- Pairwise DeLong AUC comparisons.
- Download controls for the performance and DeLong tables.

The displayed confusion-matrix rows are explicitly ordered so that the class labels correspond to the matrix values.

### SHAP Visualisations

This page provides:

- Consensus feature-ranking visualisation.
- Model-specific transformed-feature importance.
- Model-specific original-feature importance.
- SHAP-value distribution plots.
- Display and download of the long-format SHAP table.

### Patient Risk Scoring

This page provides:

- Manual patient entry.
- Input validation for all eight predictors.
- An exploratory model probability.
- An exploratory threshold decision.
- Patient-specific local SHAP contributions.
- A conceptual SHAP-to-NICE-NG97 crosswalk.
- CSV batch scoring.
- Selected-row SHAP explanation for uploaded CSV data.
- Downloadable patient scoring and explanation tables.

The threshold is an exploratory display setting. It is not a clinically validated cutoff and is not a NICE NG97 threshold.

### Result Tables

This page provides access to the available saved CSV result tables and download controls.

### Existing Figures

This page displays supported saved image figures and provides download controls. PDF files can be downloaded but are not displayed as interactive images.

## NICE NG97 conceptual crosswalk

The dashboard includes a conceptual mapping between model predictors, SHAP contributions and selected NICE NG97-related domains.

This crosswalk is contextual only. It does not mean that NICE recommends any model predictor as a diagnostic feature.

The crosswalk does not establish:

- Clinical validity.
- Diagnostic accuracy in NHS practice.
- Clinical utility.
- Calibration.
- External validation.
- NHS deployment readiness.
- Causality.
- Treatment relevance.

The dashboard must not be used to diagnose, triage or treat a patient.

## Dashboard validation

The `validators.py` module checks that:

- Expected model files exist.
- Expected table files exist.
- Expected files are non-empty.
- CSV tables can be read.
- Saved model files can be loaded.

Run artefact validation from the project root with:

### Windows

```powershell
python dashboard\validators.py
```

### macOS or Linux

```bash
python3 dashboard/validators.py
```

This validates the presence, readability and loadability of dashboard artefacts. It does not prove clinical validity or replace independent software testing.

## Synthetic dashboard test data

The project can be tested using synthetic records containing the eight required predictor columns.

These records are not real patient data:

```csv
M/F,Age,EDUC,SES,MMSE,eTIV,nWBV,ASF
F,60,6,1,30,1105.65,0.644399,0.875539
M,65,12,2,27,1300.00,0.700000,1.050000
F,72,16,3,24,1500.00,0.750000,1.200000
M,81,18,4,18,1750.00,0.790000,1.400000
F,98,23,5,4,2004.48,0.836842,1.587298
```

Save the file as:

```text
batch_scoring_test.csv
```

To test the dashboard:

1. Open Patient Risk Scoring.
2. Select CSV batch scoring.
3. Upload `batch_scoring_test.csv`.
4. Select Score uploaded patients.
5. Confirm that five scored rows appear.
6. Select one uploaded row.
7. Select Generate SHAP to NG97 explanation.

The synthetic records must not be interpreted as real patient cases or clinical examples.

## Dashboard test evidence

Representative test evidence was retained for:

- Classifier comparison.
- Confusion-matrix display.
- Patient scoring.
- Invalid MMSE input.
- Invalid eTIV input.
- Patient-specific SHAP explanation.
- NICE NG97 conceptual crosswalk.
- Result-table display.
- Existing-figure display.

All manual patient input fields were tested for validation. The retained screenshots demonstrate representative invalid-input cases.

## Saved outputs

The final analytical output set contains:

- 14 saved PNG figures.
- 10 CSV result tables.
- Three fitted model pipelines.
- Two processed-data files.
- One run manifest.

The CSV result tables include:

- Descriptive statistics.
- Missing-value audit.
- Hyperparameter tuning results.
- Held-out test-set performance.
- Held-out test predictions.
- Pairwise DeLong comparisons.
- Transformed-feature SHAP importance.
- Original-predictor SHAP importance.
- Cross-model SHAP ranking consensus.
- Long-format numerical SHAP values.

## Limitations

The project has the following limitations:

- The analysis uses a small baseline cohort after filtering.
- The held-out test set contains 28 records.
- The binary analysis excludes Converted participants.
- Evaluation is based on one held-out split.
- Repeated or nested validation could provide a more stable estimate of generalisation.
- External validation has not been performed.
- Model calibration has not been established.
- The exploratory threshold is not clinically validated.
- SHAP values do not establish causal effects.
- The raw DeLong p-values are not multiplicity-adjusted.
- The dashboard has not demonstrated prospective clinical utility, subgroup fairness, workflow compatibility or information-governance readiness.
- The NICE NG97 view is a conceptual crosswalk, not a clinical guideline benchmark.
- The dashboard is a research demonstration interface and must not be used for diagnosis, triage or treatment.

## Reproducibility

The project records:

- Fixed random states.
- The Python environment and package versions.
- Saved fitted model pipelines.
- Generated tables and figures.
- Exact held-out predictions.
- SHA-256 file hashes.
- A run fingerprint in `07_logs_and_notes/run_manifest.json`.

The main analysis and dashboard use fixed output filenames. Rerunning the analysis may overwrite existing files. A separate timestamped archive or run identifier is required to preserve a complete history of multiple runs.

## Repository hygiene

Do not commit or share the following items unless they are explicitly required and permitted:

- The `.venv` directory.
- `__pycache__` folders.
- `.pyc` files.
- Passwords, access tokens or other credentials.
- Temporary files.
- Unauthorised copies of restricted raw data.

The repository and OneDrive folder should retain the source code, README, requirements file, saved analytical outputs and dashboard artefacts required for marker review.
