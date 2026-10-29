# import pandas as pd
# from ucimlrepo import fetch_ucirepo


# def load_data():
#     # Fetch the German Credit dataset
#     dataset = fetch_ucirepo(id=144)

#     # Load features and target
#     X = dataset.data.features.copy()
#     y = dataset.data.targets.iloc[:, 0].copy()

#     # Convert target:
#     # Original 1 = Good, 2 = Bad
#     # New 0 = Good, 1 = Bad
#     y = y.map({1: 0, 2: 1})

#     # Add target to a working DataFrame
#     df = X.copy()
#     df["credit_risk"] = y

#     return df


# if __name__ == "__main__":
#     df = load_data()

#     print("Dataset shape:", df.shape)
#     print("\nFirst five rows:")
#     print(df.head())

#     print("\nTarget distribution:")
#     print(df["credit_risk"].value_counts())

#     print("\nMissing values:")
#     print(df.isnull().sum())

import pandas as pd
from pathlib import Path

TARGET_COLUMN = "credit_risk"

COLUMNS = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",
]


def load_data(file_path=None):
    if file_path is None:
        file_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "raw"
            / "german.data"
        )
    else:
        file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"German Credit data file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    if df.columns.tolist() != COLUMNS:
        raise ValueError("The German Credit file does not match the expected 21-column schema.")

    original_labels = set(df[TARGET_COLUMN].dropna().unique())
    if not original_labels.issubset({1, 2}) or df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Unexpected original target labels: {sorted(original_labels)}")

    # UCI labels: 1 = Good, 2 = Bad. Project labels: 0 = Good, 1 = Bad.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({1: 0, 2: 1}).astype("int64")

    return df


if __name__ == "__main__":
    df = load_data()

    print("Dataset shape:", df.shape)
    print(df.head())
    print("\nTarget distribution:")
    print(df["credit_risk"].value_counts())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:", df.isnull().sum())
    print("description:", df.describe())
    print("info:", df.info())
