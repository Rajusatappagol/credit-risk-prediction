import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_curve

from src.modeling import (
    MODEL_FILES,
    PROJECT_ROOT,
    bad_credit_probability,
    calculate_metrics,
    load_model,
    split_dataset,
)


def run_evaluation():
    _, X_test, _, y_test = split_dataset()
    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "UCI German Credit",
        "target_positive_class": "Bad credit (1)",
        "test_size": 0.2,
        "random_state": 42,
        "test_records": len(y_test),
        "model_artifacts": {},
        "models": {},
    }
    comparison_rows = []
    roc_figure, roc_axis = plt.subplots(figsize=(8, 6))

    for model_name, filename in MODEL_FILES.items():
        model = load_model(models_dir / filename)
        predictions = model.predict(X_test)
        bad_probabilities = bad_credit_probability(model, X_test)
        metrics = calculate_metrics(y_test, predictions, bad_probabilities)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, bad_probabilities
        )
        report["models"][model_name] = {
            **metrics,
            "confusion_labels": ["Good credit", "Bad credit"],
            "roc_curve": {
                "false_positive_rate": false_positive_rate.tolist(),
                "true_positive_rate": true_positive_rate.tolist(),
            },
        }
        artifact_path = models_dir / filename
        artifact_stat = artifact_path.stat()
        report["model_artifacts"][model_name] = {
            "size": artifact_stat.st_size,
            "modified_ns": artifact_stat.st_mtime_ns,
        }
        comparison_rows.append(
            {key: value for key, value in metrics.items() if key != "Confusion Matrix"}
            | {"Model": model_name}
        )

        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Good", "Bad"],
            yticklabels=["Good", "Bad"],
            xlabel="Predicted class",
            ylabel="Actual class",
            title=f"{model_name} confusion matrix",
        )
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        figure.colorbar(image, ax=axis, label="Records")
        figure.tight_layout()
        figure.savefig(
            figures_dir / f"confusion_matrix_{filename.removesuffix('.joblib')}.png",
            dpi=160,
        )
        plt.close(figure)

        RocCurveDisplay.from_predictions(
            y_test, bad_probabilities, name=model_name, ax=roc_axis
        )

    roc_axis.plot(
        [0, 1], [0, 1], linestyle="--", color="#78838f", label="No-skill baseline"
    )
    roc_axis.set_title("ROC curves on the held-out test set")
    roc_figure.tight_layout()
    roc_figure.savefig(figures_dir / "roc_curve.png", dpi=160)
    plt.close(roc_figure)

    confusion_figure, confusion_axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (model_name, model_results) in zip(
        confusion_axes, report["models"].items()
    ):
        matrix = model_results["Confusion Matrix"]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Good", "Bad"],
            yticklabels=["Good", "Bad"],
            xlabel="Predicted class",
            ylabel="Actual class",
            title=model_name,
        )
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    confusion_figure.tight_layout()
    confusion_figure.savefig(figures_dir / "confusion_matrix.png", dpi=160)
    plt.close(confusion_figure)

    pd.DataFrame(comparison_rows).set_index("Model").to_csv(
        reports_dir / "model_comparison.csv"
    )
    (reports_dir / "evaluation_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    report = run_evaluation()
    for model_name, metrics in report["models"].items():
        print(f"\n{model_name}")
        print(
            pd.Series(
                {
                    key: value
                    for key, value in metrics.items()
                    if key != "Confusion Matrix"
                }
            )
        )
        print("Confusion matrix [good, bad]:")
        print(
            pd.DataFrame(
                metrics["Confusion Matrix"],
                index=["Actual good", "Actual bad"],
                columns=["Predicted good", "Predicted bad"],
            )
        )
    print("\nEvaluation artifacts saved to reports/.")


if __name__ == "__main__":
    main()