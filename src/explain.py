from src.data_preprocessing import COLUMNS
from src.modeling import PROJECT_ROOT, load_model

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def aggregate_feature_importance(model):
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise ValueError("The selected model does not provide tree feature importances.")
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    aggregated = {feature: 0.0 for feature in COLUMNS[:-1]}
    for transformed_name, importance in zip(
        transformed_names, classifier.feature_importances_
    ):
        source_name = transformed_name.split("__", 1)[-1]
        feature_name = next(
            (
                column
                for column in COLUMNS[:-1]
                if source_name == column or source_name.startswith(f"{column}_")
            ),
            None,
        )
        if feature_name is not None:
            aggregated[feature_name] += float(importance)
    result = pd.DataFrame(
        {"Feature": list(aggregated), "Importance": list(aggregated.values())}
    ).sort_values("Importance", ascending=False, ignore_index=True)
    if result["Importance"].sum() <= 0:
        raise ValueError("The model did not return usable feature importances.")
    return result


def run_explainability():
    model = load_model(PROJECT_ROOT / "models" / "random_forest.joblib")
    importance = aggregate_feature_importance(model)
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    top_features = importance.head(15).sort_values("Importance")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_features["Feature"], top_features["Importance"], color="#147d73")
    axis.set(
        title="Random Forest global feature importance",
        xlabel="Aggregated impurity importance",
    )
    figure.tight_layout()
    figure.savefig(figures_dir / "feature_importance.png", dpi=160)
    plt.close(figure)
    return importance


def main():
    print(run_explainability().head(15).to_string(index=False))
    print("\nFeature importance saved to reports/.")


if __name__ == "__main__":
    main()