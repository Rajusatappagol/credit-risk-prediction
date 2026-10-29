from src.modeling import split_dataset, train_and_save


def main():
    X_train, _, y_train, _ = split_dataset()
    train_and_save()
    print("Trained Logistic Regression and Random Forest pipelines saved to models/.")
    print(f"Training records: {len(X_train)}; bad-credit records: {int(y_train.sum())}")


if __name__ == "__main__":
    main()