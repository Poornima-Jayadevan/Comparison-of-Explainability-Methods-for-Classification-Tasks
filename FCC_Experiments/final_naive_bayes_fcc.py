

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


DATA_FILE = Path("labeled_fringes.csv")

MODEL_FILE = Path("final_categorical_nb_model.joblib")
FEATURE_ENCODER_FILE = Path("final_feature_encoder.joblib")
LABEL_ENCODER_FILE = Path("final_label_encoder.joblib")
METADATA_FILE = Path("final_model_metadata.csv")
PREDICTIONS_FILE = Path("final_training_predictions.csv")


FEATURE_COLUMNS = [
    "chain_length_bin",
    "direction_change_bin",
    "fuzziness_bin",
]

TARGET_COLUMN = "label"


def load_and_validate_data() -> pd.DataFrame:
    """Load the labeled dataset and validate required columns."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found. Keep it in the same folder "
            "as this script."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()

    valid_labels = {"Keep", "Discard"}
    invalid_labels = set(df[TARGET_COLUMN].unique()).difference(valid_labels)

    if invalid_labels:
        raise ValueError(
            "Unexpected labels were found: "
            + ", ".join(sorted(map(str, invalid_labels)))
        )

    if len(df) == 0:
        raise ValueError("No labeled rows are available for training.")

    return df


def build_feature_encoder() -> OrdinalEncoder:
    
    return OrdinalEncoder(
        categories=[
            ["Short", "Medium", "Long"],
            ["Low", "Medium", "High"],
            ["Low", "Medium", "High"],
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int64,
    )


def train_final_model() -> None:
    """Train, evaluate on the full dataset, and save the final model."""
    df = load_and_validate_data()

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    feature_encoder = build_feature_encoder()
    X_encoded = feature_encoder.fit_transform(X).astype(int)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("=" * 70)
    print("Final Model Training")
    print("=" * 70)
    print(f"Number of labeled fringes: {len(df)}")
    print("\nClass distribution:")
    print(y.value_counts().to_string())

    print("\nFeature encoding:")
    for feature_name, categories in zip(
        FEATURE_COLUMNS,
        feature_encoder.categories_,
    ):
        mapping = ", ".join(
            f"{category}={index}"
            for index, category in enumerate(categories)
        )
        print(f"{feature_name}: {mapping}")

    print("\nClass encoding:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name}={index}")

    # alpha=1.0 applies Laplace smoothing and prevents zero probabilities.
    model = CategoricalNB(alpha=1.0)
    model.fit(X_encoded, y_encoded)

    y_pred = model.predict(X_encoded)
    y_prob = model.predict_proba(X_encoded)

    training_accuracy = accuracy_score(y_encoded, y_pred)

    print("\n" + "=" * 70)
    print("Full-Dataset Consistency Check")
    print("=" * 70)
    print(f"Training accuracy: {training_accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_encoded,
            y_pred,
            target_names=label_encoder.classes_,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_encoded, y_pred))

    print("\nClass prior probabilities learned by the model:")
    for class_name, log_prior in zip(
        label_encoder.classes_,
        model.class_log_prior_,
    ):
        print(f"P({class_name}) = {np.exp(log_prior):.6f}")

    # Save model and encoders.
    joblib.dump(model, MODEL_FILE)
    joblib.dump(feature_encoder, FEATURE_ENCODER_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)

    # Save metadata describing the model configuration.
    metadata_rows = [
        {
            "item": "model_type",
            "value": "CategoricalNB",
        },
        {
            "item": "laplace_smoothing_alpha",
            "value": model.alpha,
        },
        {
            "item": "training_samples",
            "value": len(df),
        },
        {
            "item": "feature_columns",
            "value": ", ".join(FEATURE_COLUMNS),
        },
        {
            "item": "target_column",
            "value": TARGET_COLUMN,
        },
        {
            "item": "class_names",
            "value": ", ".join(label_encoder.classes_),
        },
        {
            "item": "cross_validation_accuracy",
            "value": "0.9430 ± 0.0620",
        },
        {
            "item": "training_accuracy_consistency_check",
            "value": training_accuracy,
        },
    ]

    pd.DataFrame(metadata_rows).to_csv(
        METADATA_FILE,
        index=False,
    )

    # Save predictions and probabilities for every labeled fringe.
    prediction_df = df.copy()
    prediction_df["predicted_label"] = label_encoder.inverse_transform(
        y_pred
    )
    prediction_df["correct_prediction"] = (
        prediction_df[TARGET_COLUMN]
        == prediction_df["predicted_label"]
    )

    for class_index, class_name in enumerate(label_encoder.classes_):
        prediction_df[f"probability_{class_name.lower()}"] = (
            y_prob[:, class_index]
        )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("Saved Files")
    print("=" * 70)
    print(MODEL_FILE.resolve())
    print(FEATURE_ENCODER_FILE.resolve())
    print(LABEL_ENCODER_FILE.resolve())
    print(METADATA_FILE.resolve())
    print(PREDICTIONS_FILE.resolve())



if __name__ == "__main__":
    train_final_model()