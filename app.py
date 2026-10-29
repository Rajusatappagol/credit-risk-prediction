import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.data_preprocessing import COLUMNS, TARGET_COLUMN, load_data
from src.explain import aggregate_feature_importance
from src.modeling import (
    MODEL_FILES,
    PROJECT_ROOT,
    bad_credit_probability,
    load_model,
)

st.set_page_config(
    page_title="Credit Risk Prediction & Explainability",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "german.data"
EVALUATION_PATH = REPORT_DIR / "evaluation_results.json"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"

FIELD_INFO = {
    "checking_account": ("Checking account balance", "Current checking-account status in the source dataset."),
    "duration": ("Credit duration (months)", "Planned duration of the credit in months."),
    "credit_history": ("Credit history", "History of prior credits and repayment status."),
    "purpose": ("Credit purpose", "Purpose for which the credit was requested."),
    "credit_amount": ("Credit amount (DM)", "Requested credit amount, recorded in Deutsche Marks in the source data."),
    "savings": ("Savings account / bonds", "Savings balance category in the source dataset."),
    "employment": ("Length of current employment", "Duration of present employment."),
    "installment_rate": ("Installment rate (% of income)", "Installment rate as a percentage of disposable income, grouped into levels 1 to 4."),
    "personal_status_sex": ("Personal status and sex", "Combined personal-status and sex category as supplied in the historical dataset."),
    "other_debtors": ("Other debtors / guarantors", "Whether another person is a co-applicant or guarantor."),
    "residence_since": ("Years at current residence", "Duration at the current residence, grouped into levels 1 to 4."),
    "property": ("Most valuable property", "Category describing the applicant's most valuable property."),
    "age": ("Age (years)", "Applicant age in years as recorded in the historical data."),
    "other_installment_plans": ("Other installment plans", "Other installment-plan provider, if any."),
    "housing": ("Housing situation", "Housing category in the source dataset."),
    "existing_credits": ("Existing credits at this bank", "Number of existing credits at the bank, grouped into levels 1 to 4."),
    "job": ("Employment category", "Occupational category in the source dataset."),
    "dependents": ("People financially liable", "Number of people for whom the applicant provides maintenance, grouped into levels 1 or 2."),
    "telephone": ("Telephone status", "Whether a telephone is registered in the applicant's name."),
    "foreign_worker": ("Foreign-worker indicator", "Historical dataset indicator; sensitive and potentially inappropriate for real lending use."),
}

CATEGORY_LABELS = {
    "checking_account": {
        "A11": "Overdraft / balance below 0 DM",
        "A12": "Balance from 0 to below 200 DM",
        "A13": "Balance of 200 DM or more",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits or all credits paid back",
        "A31": "All credits at this bank paid back duly",
        "A32": "Existing credits paid back duly so far",
        "A33": "Past delays in payment",
        "A34": "Critical account or other credits elsewhere",
    },
    "purpose": {
        "A40": "New car",
        "A41": "Used car",
        "A42": "Furniture or equipment",
        "A43": "Radio or television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Other purpose",
    },
    "savings": {
        "A61": "Below 100 DM",
        "A62": "100 to below 500 DM",
        "A63": "500 to below 1,000 DM",
        "A64": "1,000 DM or more",
        "A65": "Unknown or no savings account",
    },
    "employment": {
        "A71": "Unemployed",
        "A72": "Employed for less than 1 year",
        "A73": "Employed for 1 to less than 4 years",
        "A74": "Employed for 4 to less than 7 years",
        "A75": "Employed for 7 years or more",
    },
    "personal_status_sex": {
        "A91": "Male: divorced or separated",
        "A92": "Female: divorced, separated, or married",
        "A93": "Male: single",
        "A94": "Male: married or widowed",
        "A95": "Female: single",
    },
    "other_debtors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "installment_rate": {
        1: "1 — 35% or more",
        2: "2 — 25% to less than 35%",
        3: "3 — 20% to less than 25%",
        4: "4 — less than 20%",
    },
    "residence_since": {
        1: "1 — less than 1 year",
        2: "2 — 1 to less than 4 years",
        3: "3 — 4 to less than 7 years",
        4: "4 — 7 years or more",
    },
    "property": {
        "A121": "Real estate",
        "A122": "Building-society savings or life insurance",
        "A123": "Car or other property",
        "A124": "Unknown or no property",
    },
    "other_installment_plans": {
        "A141": "Bank",
        "A142": "Retail store",
        "A143": "None",
    },
    "housing": {"A151": "Rent", "A152": "Own", "A153": "Free housing"},
    "existing_credits": {
        1: "1 existing credit",
        2: "2 existing credits",
        3: "3 existing credits",
        4: "4 or more existing credits",
    },
    "job": {
        "A171": "Unemployed / unskilled non-resident",
        "A172": "Unskilled resident worker",
        "A173": "Skilled employee or official",
        "A174": "Management, self-employed, or highly qualified",
    },
    "dependents": {1: "1 person", 2: "2 or more people"},
    "telephone": {"A191": "No telephone registered", "A192": "Telephone registered"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}

GROUPS = {
    "Credit profile": [
        "checking_account",
        "credit_history",
        "existing_credits",
        "other_installment_plans",
    ],
    "Loan details": ["duration", "purpose", "credit_amount", "installment_rate"],
    "Employment and finances": ["savings", "employment", "job", "telephone", "foreign_worker"],
    "Personal and housing": [
        "personal_status_sex",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "housing",
        "dependents",
    ],
}

FEATURE_NAMES = {column: values[0] for column, values in FIELD_INFO.items()}

st.markdown(
    """
    <style>
    :root { --ink: #172b31; --muted: #64757b; --teal: #147d73; --mint: #e8f4f1; --line: #dce5e3; --amber: #b45d2a; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    [data-testid="stForm"] { border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.25rem; }
    [data-testid="stAlert"] { border-radius: 6px; }
    .eyebrow { color: var(--teal); font-weight: 700; font-size: .78rem; text-transform: uppercase; }
    .muted-copy { color: var(--muted); }
    @media (max-width: 700px) { .block-container { padding: 1rem .8rem 2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _modified_ns(path):
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


@st.cache_data(show_spinner=False)
def cached_dataset(path, modified_ns):
    return load_data(path)


@st.cache_resource(show_spinner=False)
def cached_model(path, modified_ns):
    return load_model(path)


@st.cache_data(show_spinner=False)
def cached_json(path, modified_ns):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def cached_csv(path, modified_ns):
    return pd.read_csv(path)


def _evaluation_is_current(report):
    if not report:
        return False
    for model_name, filename in MODEL_FILES.items():
        artifact = MODEL_DIR / filename
        signature = report.get("model_artifacts", {}).get(model_name, {})
        if not artifact.is_file():
            return False
        current = artifact.stat()
        if signature.get("size") != current.st_size or signature.get("modified_ns") != current.st_mtime_ns:
            return False
    return True


def _show_header(title, subtitle):
    st.markdown('<p class="eyebrow">Credit analytics workspace</p>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="muted-copy">{subtitle}</p>', unsafe_allow_html=True)


def _metric_row(metrics, labels):
    columns = st.columns(len(labels))
    for column, (key, label) in zip(columns, labels):
        value = metrics.get(key)
        column.metric(label, "N/A" if value is None else f"{value:.3f}")


def _model_performance_chart(report):
    records = []
    for model_name, values in report["models"].items():
        for false_positive, true_positive in zip(
            values["roc_curve"]["false_positive_rate"],
            values["roc_curve"]["true_positive_rate"],
        ):
            records.append(
                {
                    "False positive rate": false_positive,
                    "True positive rate": true_positive,
                    "Model": model_name,
                }
            )
    curve = alt.Chart(pd.DataFrame(records)).mark_line().encode(
        x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=alt.Scale(range=["#147d73", "#b45d2a"])),
        tooltip=["Model:N", "False positive rate:Q", "True positive rate:Q"],
    )
    baseline = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    reference = alt.Chart(baseline).mark_line(strokeDash=[5, 4], color="#8d999c").encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
    )
    return (curve + reference).properties(height=340).interactive()


def _confusion_chart(model_name, matrix):
    records = []
    labels = ["Good credit", "Bad credit"]
    for row in range(2):
        for column in range(2):
            records.append(
                {
                    "Actual": labels[row],
                    "Predicted": labels[column],
                    "Records": matrix[row][column],
                }
            )
    cells = alt.Chart(pd.DataFrame(records)).encode(
        x=alt.X("Predicted:N", sort=labels, title="Predicted class"),
        y=alt.Y("Actual:N", sort=labels, title="Actual class"),
        color=alt.Color("Records:Q", scale=alt.Scale(scheme="tealblues")),
        tooltip=["Actual:N", "Predicted:N", "Records:Q"],
    )
    return (cells.mark_rect() + cells.mark_text(color="#172b31", fontSize=16).encode(text="Records:Q")).properties(
        title=model_name, height=220
    )


def _render_overview(data, report, selected_model, model_error):
    _show_header(
        "Credit Risk Prediction & Explainability",
        "An educational analysis of the UCI German Credit dataset, with transparent model evaluation and applicant-level demonstrations.",
    )
    st.info("Educational demonstration only. This model is not validated for lending decisions and must not be used to approve or reject applicants.")
    st.subheader("Historical dataset profile")
    good_count = int((data[TARGET_COLUMN] == 0).sum()) if data is not None else None
    bad_count = int((data[TARGET_COLUMN] == 1).sum()) if data is not None else None
    feature_count = len(data.columns) - 1 if data is not None else len(COLUMNS) - 1
    columns = st.columns(4)
    columns[0].metric("Dataset records", f"{len(data):,}" if data is not None else "Unavailable")
    columns[1].metric("Input features", feature_count)
    columns[2].metric("Good-credit records", f"{good_count:,}" if good_count is not None else "N/A")
    columns[3].metric("Bad-credit records", f"{bad_count:,}" if bad_count is not None else "N/A")
    st.caption("These counts describe historical labels in the dataset; they are not applicant predictions.")
    st.divider()
    st.subheader("Selected model snapshot")
    st.write(f"Current model: **{selected_model}**")
    if model_error:
        st.error(f"The selected model could not be loaded: {model_error}")
    elif report and _evaluation_is_current(report):
        _metric_row(
            report["models"].get(selected_model, {}),
            [("Accuracy", "Accuracy"), ("F1-score", "F1-score"), ("ROC-AUC", "ROC-AUC")],
        )
        st.caption(f"Held-out evaluation on {report['test_records']} records. Positive class: bad credit (1).")
    else:
        st.warning("Current evaluation results are unavailable or out of date. Run `python -m src.evaluate` after training.")


def _render_applicant(data, model, model_error, selected_model):
    _show_header("Applicant information", "Enter a hypothetical profile to see how the selected model classifies the historical dataset label.")
    st.warning("Sensitive attributes are included because they are part of the original dataset schema. Their presence is for educational analysis only.")
    if data is None:
        st.error("Applicant defaults cannot be loaded because the dataset is unavailable. Restore `data/raw/german.data` and reload the page.")
        return
    if model is None:
        st.error(f"The selected model is unavailable: {model_error or 'model file is missing'}")
        st.code("python -m src.train", language="powershell")
        return

    user_input = {}
    with st.form("applicant_form"):
        for group_name, feature_names in GROUPS.items():
            st.markdown(f"#### {group_name}")
            columns = st.columns(2)
            for index, feature in enumerate(feature_names):
                label, description = FIELD_INFO[feature]
                target_column = columns[index % 2]
                if feature in CATEGORY_LABELS:
                    observed_values = set(data[feature].dropna().unique())
                    options = [
                        option
                        for option in CATEGORY_LABELS[feature].items()
                        if option[0] in observed_values
                    ]
                    observed = data[feature].mode(dropna=True)
                    default_value = observed.iloc[0] if not observed.empty else options[0][0]
                    selected_index = next(
                        (position for position, option in enumerate(options) if option[0] == default_value),
                        0,
                    )
                    value, _ = target_column.selectbox(
                        label,
                        options=options,
                        index=selected_index,
                        format_func=lambda option: option[1],
                        help=description,
                        key=f"applicant_{feature}",
                    )
                else:
                    series = data[feature]
                    minimum = int(series.min())
                    maximum = int(series.max())
                    default = int(round(series.median()))
                    step = 50 if feature == "credit_amount" else 1
                    value = target_column.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=step,
                        help=description,
                        key=f"applicant_{feature}",
                    )
                user_input[feature] = value
        st.caption("Dropdown wording is for readability; the original dataset codes are passed unchanged to the model.")
        submitted = st.form_submit_button("Predict credit risk", type="primary", width="stretch")

    if submitted:
        try:
            input_frame = pd.DataFrame([[user_input[column] for column in COLUMNS[:-1]]], columns=COLUMNS[:-1])
            prediction = int(model.predict(input_frame)[0])
            bad_probability = float(bad_credit_probability(model, input_frame)[0])
            result_column, probability_column = st.columns([1, 2])
            result_label = "Bad-credit class" if prediction == 1 else "Good-credit class"
            result_column.metric("Model prediction", result_label)
            probability_column.metric("Estimated bad-credit class probability", f"{bad_probability:.1%}")
            if prediction == 1:
                st.warning("The model assigned this hypothetical record to the bad-credit category in the historical training labels.")
            else:
                st.success("The model assigned this hypothetical record to the good-credit category in the historical training labels.")
            st.caption(f"Model: {selected_model}. This is a model output, not a guaranteed future outcome or a loan decision. Probabilities are not calibrated for real-world default risk.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")


def _render_performance(report):
    _show_header("Model performance", "Fixed stratified holdout evaluation. Both models use the same split; the positive class is bad credit (1).")
    if not report:
        st.warning("Evaluation results are not available. Train and evaluate the models to generate current results.")
        st.code("python -m src.train\npython -m src.evaluate", language="powershell")
        return
    if not _evaluation_is_current(report):
        st.warning("The saved evaluation report does not match the current model artifacts. Regenerate it before interpreting scores.")
        st.code("python -m src.evaluate", language="powershell")
        return

    metric_rows = []
    for model_name, metrics in report["models"].items():
        metric_rows.append({"Model": model_name, **{key: metrics[key] for key in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}})
    metrics_frame = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(metrics_frame.style.format("{:.3f}"), width="stretch")
    st.caption(f"Test fold: {report['test_records']} records. Precision, recall, and F1 use bad credit as the positive class. No test-fold threshold tuning was performed.")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("ROC curves")
        st.altair_chart(_model_performance_chart(report), width="stretch")
    with right:
        st.subheader("Confusion matrices")
        for model_name, metrics in report["models"].items():
            st.altair_chart(_confusion_chart(model_name, metrics["Confusion Matrix"]), width="stretch")
    st.caption("The 0.5 classification threshold is used by default. False-positive and false-negative costs differ by context; these metrics alone do not define a responsible lending policy.")


def _render_explainability():
    _show_header("Explainability", "Global Random Forest feature importance aggregated back to the original dataset fields.")
    model_path = MODEL_DIR / MODEL_FILES["Random Forest"]
    if not model_path.is_file():
        st.warning("The Random Forest pipeline is missing. Run `python -m src.train` first.")
        return
    if not IMPORTANCE_PATH.is_file() or _modified_ns(IMPORTANCE_PATH) < _modified_ns(model_path):
        st.warning("Feature-importance results are missing or older than the current Random Forest. Run `python -m src.explain`.")
        return
    try:
        importance = cached_csv(str(IMPORTANCE_PATH), _modified_ns(IMPORTANCE_PATH))
        current_model = cached_model(str(model_path), _modified_ns(model_path))
        expected = aggregate_feature_importance(current_model)
        if not importance["Feature"].equals(expected["Feature"]):
            importance = expected
        importance["Readable feature"] = importance["Feature"].map(FEATURE_NAMES).fillna(importance["Feature"])
        chart = alt.Chart(importance.head(15).sort_values("Importance")).mark_bar(color="#147d73").encode(
            x=alt.X("Importance:Q", title="Aggregated impurity importance"),
            y=alt.Y("Readable feature:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("Readable feature:N", title="Feature"), alt.Tooltip("Importance:Q", format=".4f")],
        ).properties(height=440).interactive()
        st.altair_chart(chart, width="stretch")
        st.caption("Method: Random Forest impurity-based feature importance; one-hot category contributions are summed by their original input feature.")
        st.warning("Importance is a global association measure, not a causal explanation. It does not show whether a feature raises or lowers a prediction, can favor features with many possible splits, and may conceal subgroup behavior.")
    except Exception as error:
        st.error(f"Could not load feature-importance results: {error}")


def _render_project_information():
    _show_header("Project information", "Scope, methods, and important limits of this educational demonstration.")
    st.subheader("Objective and data")
    st.write("Compare Logistic Regression and Random Forest classifiers on the UCI German Credit dataset. The source target is 1 = good and 2 = bad; this project maps it to 0 = good and 1 = bad.")
    st.markdown("Dataset source: [UCI Machine Learning Repository: Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)")
    st.write("The original dataset contains 1,000 historical records, 20 input fields, and a 700/300 good/bad class split. Monetary amounts are denominated in historical Deutsche Marks.")
    st.subheader("Models and evaluation")
    st.write("Logistic Regression uses standardized continuous features, one-hot categorical fields, and class weighting. Random Forest uses the same preprocessing schema with class weighting. Imputation and encoding are fitted inside each pipeline on training records only.")
    st.write("A reproducible, stratified 80/20 split with random state 42 is used for the reported holdout metrics. Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curves are generated by `python -m src.evaluate`.")
    st.subheader("Technologies")
    st.write("Python, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib, Altair, and Streamlit.")
    st.subheader("Limitations and responsible use")
    st.markdown("""
    - This is a small, historical dataset and may not represent current applicants, economic conditions, or lending practices.
    - The class distribution is imbalanced; false-positive and false-negative trade-offs should be reviewed separately.
    - Predicted scores are not calibrated probabilities of default and have not been validated for production use.
    - Generalization, subgroup performance, and fairness have not been established.
    - The dataset includes personal status/sex and foreign-worker fields; other variables may act as sensitive proxies.
    - Global feature importance does not establish causation or explain any individual prediction.
    - Do not use this application to make or automate real credit, approval, rejection, pricing, or eligibility decisions.
    """)


st.sidebar.markdown("### Credit analytics")
st.sidebar.caption("German Credit educational demo")
page = st.sidebar.radio(
    "Workspace",
    ["Overview", "Applicant Information", "Model Performance", "Explainability", "Project Information"],
)
selected_model = st.sidebar.selectbox("Prediction model", list(MODEL_FILES))
st.sidebar.divider()
st.sidebar.caption("Historical analysis only. Not a lending decision tool.")

dataset_error = None
try:
    data = cached_dataset(str(DATA_PATH), _modified_ns(DATA_PATH))
except Exception as error:
    data = None
    dataset_error = str(error)
    st.sidebar.error(f"Dataset unavailable: {dataset_error}")

report = None
if EVALUATION_PATH.is_file():
    try:
        report = cached_json(str(EVALUATION_PATH), _modified_ns(EVALUATION_PATH))
    except Exception as error:
        st.sidebar.error(f"Evaluation report could not be read: {error}")

selected_model_path = MODEL_DIR / MODEL_FILES[selected_model]
model = None
model_error = None
if selected_model_path.is_file():
    try:
        model = cached_model(str(selected_model_path), _modified_ns(selected_model_path))
    except Exception as error:
        model_error = str(error)
else:
    model_error = f"Model file not found: {selected_model_path}"

if page == "Overview":
    _render_overview(data, report, selected_model, model_error)
elif page == "Applicant Information":
    _render_applicant(data, model, model_error, selected_model)
elif page == "Model Performance":
    _render_performance(report)
elif page == "Explainability":
    _render_explainability()
else:
    _render_project_information()