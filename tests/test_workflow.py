from pathlib import Path

import numpy as np
import pytest

from src.data_preprocessing import COLUMNS, load_data
from src.modeling import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    bad_credit_probability,
    calculate_metrics,
    load_model,
    split_dataset,
    train_and_save,
)
from src.explain import aggregate_feature_importance


def test_dataset_schema_and_target_mapping():
    data = load_data()

    assert data.shape == (1000, 21)
    assert data.columns.tolist() == COLUMNS
    assert data["credit_risk"].value_counts().to_dict() == {0: 700, 1: 300}
    assert data.isna().sum().sum() == 0


def test_missing_dataset_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="data file not found"):
        load_data(tmp_path / "missing.data")


def test_original_target_labels_map_to_project_classes(tmp_path):
    source_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "german.data"
    sample_path = tmp_path / "sample.data"
    source_rows = source_path.read_text(encoding="utf-8").splitlines(keepends=True)[:2]
    sample_path.write_text("".join(source_rows), encoding="utf-8")

    assert load_data(sample_path)["credit_risk"].tolist() == [0, 1]


def test_semantic_feature_types_cover_schema():
    assert len(NUMERICAL_FEATURES) == 3
    assert len(CATEGORICAL_FEATURES) == 17
    assert set(NUMERICAL_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES) == set(COLUMNS[:-1])


def test_training_serialization_schema_and_unseen_categories(tmp_path):
    data = load_data()
    models = train_and_save(data, tmp_path)
    model = load_model(tmp_path / "logistic_regression.joblib")
    sample = data.drop(columns="credit_risk").iloc[[0]].copy()
    sample.loc[:, "purpose"] = "UNSEEN_CATEGORY"

    prediction = model.predict(sample)
    probability = bad_credit_probability(model, sample)
    class_one_index = list(model.named_steps["classifier"].classes_).index(1)

    assert prediction.shape == (1,)
    assert prediction[0] in (0, 1)
    assert probability.shape == (1,)
    assert 0.0 <= probability[0] <= 1.0
    assert probability[0] == pytest.approx(
        model.predict_proba(sample)[0, class_one_index]
    )
    assert list(models["Logistic Regression"].feature_names_in_) == COLUMNS[:-1]
    assert (tmp_path / "random_forest.joblib").is_file()
    importance = aggregate_feature_importance(models["Random Forest"])
    assert set(importance["Feature"]) == set(COLUMNS[:-1])
    assert importance["Importance"].sum() == pytest.approx(1.0)


def test_missing_model_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(tmp_path / "missing.joblib")


def test_split_is_stratified_and_preprocessing_handles_missing_values(tmp_path):
    data = load_data()
    X_train, X_test, y_train, y_test = split_dataset(data)
    data["duration"] = data["duration"].astype(float)
    data.loc[X_train.index[0], ["duration", "checking_account"]] = np.nan
    models = train_and_save(data, output_dir=tmp_path)
    incomplete_record = X_test.iloc[[0]].copy()
    incomplete_record["duration"] = incomplete_record["duration"].astype(float)
    incomplete_record.loc[:, "duration"] = np.nan
    incomplete_record.loc[:, "checking_account"] = np.nan
    predictions = models["Logistic Regression"].predict(incomplete_record)

    assert len(X_train) == 800
    assert len(X_test) == 200
    assert y_test.value_counts(normalize=True).sort_index().tolist() == [0.7, 0.3]
    assert predictions.shape == (1,)
    assert y_train.notna().all()


def test_metric_calculation_uses_bad_credit_as_positive_class():
    metrics = calculate_metrics(
        target=[0, 0, 1, 1],
        predictions=[0, 1, 1, 0],
        bad_probabilities=[0.1, 0.7, 0.8, 0.4],
    )

    assert metrics["Accuracy"] == 0.5
    assert metrics["Precision"] == 0.5
    assert metrics["Recall"] == 0.5
    assert metrics["F1-score"] == 0.5
    assert metrics["ROC-AUC"] == 0.75
    assert metrics["Confusion Matrix"] == [[1, 1], [1, 1]]