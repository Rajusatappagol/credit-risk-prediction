# Credit Risk Prediction & Explainability

An educational machine-learning dashboard for exploring the UCI Statlog German Credit dataset. It trains Logistic Regression and Random Forest pipelines, compares both on a fixed stratified holdout, and offers a hypothetical applicant prediction workflow with global feature-importance analysis.

This is not a lending product. The models and scores are not validated for real-world credit decisions.

## Project structure

```text
credit-risk-prediction/
├── app.py                         # Streamlit dashboard
├── data/raw/german.data           # Original 1,000-row UCI data file
├── models/                        # Serialized preprocessing/model pipelines
├── reports/
│   ├── evaluation_results.json    # Metrics, confusion matrices, ROC points
│   ├── model_comparison.csv       # Holdout metric summary
│   ├── feature_importance.csv     # Random Forest importance by source feature
│   └── figures/                   # Generated evaluation/explanation charts
├── src/
│   ├── data_preprocessing.py      # Data loading and target mapping
│   ├── modeling.py                # Pipeline, training, inference, and metrics
│   ├── train.py                   # Model training command
│   ├── evaluate.py                # Evaluation and chart generation command
│   └── explain.py                 # Global feature-importance command
├── tests/                         # pytest workflow and dashboard tests
├── requirements.txt
└── README.md
```

## Dataset

The project uses the original whitespace-separated `german.data` file from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). It contains 1,000 rows, 20 input attributes, and one target. The original target is `1 = Good` and `2 = Bad`; the project consistently maps it to `0 = Good` and `1 = Bad`. The checked-in data has 700 good and 300 bad records and no missing values.

Three measurements are processed as continuous numeric features: duration, credit amount, and age. Other integer-coded attributes are treated as categorical inputs, not continuous measurements. Missing values are imputed within the training pipeline; unseen categories are safely ignored by the one-hot encoder. No records are discarded by the loader.

## Windows setup

Open PowerShell in the project directory (`credit-risk-prediction`):

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the virtual environment already exists, activate it with the `Activate.ps1` command and install/update requirements.

## Train, evaluate, explain

Run commands from the project directory with the virtual environment activated:

```powershell
python -m src.data_preprocessing
python -m src.train
python -m src.evaluate
python -m src.explain
```

Training saves both complete preprocessing/model pipelines to `models/`. Evaluation uses a reproducible 80/20 stratified split (`random_state=42`) and writes authentic holdout metrics, ROC curve points, confusion matrices, and charts to `reports/`. Explainability aggregates Random Forest impurity-based importances from encoded columns back to the original input fields.

Run training before evaluation if model files are missing or the feature-processing implementation changes. Evaluation metrics are not tuned on the holdout set. The default class threshold is 0.5 and has not been optimized for a decision policy.

## Run the dashboard

```powershell
streamlit run app.py
```

The sidebar navigates between Overview, Applicant Information, Model Performance, Explainability, and Project Information. Applicant form values are converted to the exact model feature schema and passed through a saved pipeline. Readable dropdown text maps to the original dataset category values; submitted values are not exported or written to project files.

## Tests and syntax checks

```powershell
python -m pytest -q
python -m compileall -q app.py src tests
```

Tests cover data dimensions and target mapping, missing files and values, semantic feature typing, stratified splitting, train/inference schema, serialization, unseen categories, class-1 probability selection, metric calculations, feature importance, dashboard navigation, and a real pipeline prediction.

## Current holdout results

These results were regenerated from the current pipelines and the fixed 200-record holdout. The test fold contains 160 good-credit and 40 bad-credit records. Positive class for precision, recall, and F1 is bad credit (`1`).

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Confusion matrix (actual rows: good, bad; predicted columns: good, bad) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.765 | 0.578 | 0.800 | 0.671 | 0.814 | `[[105, 35], [12, 48]]` |
| Random Forest | 0.760 | 0.591 | 0.650 | 0.619 | 0.793 | `[[113, 27], [21, 39]]` |

These are one-split estimates on a small historical sample, not guarantees of future performance. The UCI dataset documents an asymmetric classification cost matrix; this project reports standard metrics and does not apply that matrix to tune its threshold or make decisions.

## Limitations and responsible use

The dataset is small, historical, and potentially biased; its class balance and population may not reflect present-day applicants. False-positive and false-negative errors have different consequences. The model's predicted class probability is not calibrated as a real-world probability of default. Generalization and subgroup fairness have not been established. The original inputs include sensitive personal status/sex and foreign-worker fields, and other features may act as proxies. Global feature importance is not causal and cannot explain the direction or rationale of an individual prediction.

Use this project only for education and experimentation. Do not use it to approve, reject, price, or otherwise make real credit decisions.
