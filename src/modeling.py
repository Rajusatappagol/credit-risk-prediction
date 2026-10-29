from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import COLUMNS, TARGET_COLUMN, load_data

NUMERICAL_FEATURES = ["duration", "credit_amount", "age"]
CATEGORICAL_FEATURES = [
    column for column in COLUMNS[:-1] if column not in NUMERICAL_FEATURES
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_pipeline(classifier):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def make_classifiers():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def split_dataset(data=None, test_size=0.2, random_state=42):
    data = load_data() if data is None else data
    if data.columns.tolist() != COLUMNS:
        raise ValueError("Dataset columns do not match the German Credit schema.")
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN].astype(int)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def train_models(data=None):
    X_train, _, y_train, _ = split_dataset(data)
    fitted_models = {}
    for model_name, classifier in make_classifiers().items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline
    return fitted_models


def train_and_save(data=None, output_dir=None):
    output_dir = Path(output_dir or PROJECT_ROOT / "models")
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted_models = train_models(data)
    for model_name, model in fitted_models.items():
        joblib.dump(model, output_dir / MODEL_FILES[model_name])
    return fitted_models


def load_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or "preprocessor" not in model.named_steps:
        raise ValueError(f"Model artifact is not a supported preprocessing pipeline: {path}")
    if list(model.feature_names_in_) != COLUMNS[:-1]:
        raise ValueError(f"Model feature schema does not match the dataset: {path}")
    return model


def bad_credit_probability(model, features):
    classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not contain class 1 (bad credit).")
    return np.asarray(model.predict_proba(features)[:, classes.index(1)], dtype=float)


def calculate_metrics(target, predictions, bad_probabilities):
    return {
        "Accuracy": float(accuracy_score(target, predictions)),
        "Precision": float(precision_score(target, predictions, zero_division=0)),
        "Recall": float(recall_score(target, predictions, zero_division=0)),
        "F1-score": float(f1_score(target, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(target, bad_probabilities)),
        "Confusion Matrix": confusion_matrix(
            target, predictions, labels=[0, 1]
        ).tolist(),
    }