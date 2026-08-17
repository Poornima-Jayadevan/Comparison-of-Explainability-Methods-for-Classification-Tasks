from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


DATA_FILE = Path("labeled_fringes.csv")

FEATURE_COLUMNS = [
    "chain_length_bin",
    "direction_change_bin",
    "fuzziness_bin",
]

TARGET_COLUMN = "label"

RANDOM_STATE = 42
N_SPLITS = 5


def load_data() -> pd.DataFrame:
    """Load and validate the labeled FCC dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found. Keep it in the same folder "
            "as this script."
        )

    df = pd.read_csv(DATA_FILE)

    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()

    valid_labels = {"Keep", "Discard"}
    invalid_labels = set(df[TARGET_COLUMN].unique()).difference(valid_labels)

    if invalid_labels:
        raise ValueError(
            "Unexpected labels found: "
            + ", ".join(sorted(map(str, invalid_labels)))
        )

    return df


def build_encoder() -> ColumnTransformer:
    """
    One-hot encode the three categorical feature-state columns.
    """
    encoder = OneHotEncoder(
        categories=[
            ["Short", "Medium", "Long"],
            ["Low", "Medium", "High"],
            ["Low", "Medium", "High"],
        ],
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                encoder,
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model() -> RandomForestClassifier:
    """Create the Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
) -> None:
    """Display and save the overall out-of-fold confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix)

    ax.set_xticks(
        range(len(class_names)),
        labels=class_names,
    )
    ax.set_yticks(
        range(len(class_names)),
        labels=class_names,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(
        "Random Forest — 5-Fold Out-of-Fold Confusion Matrix"
    )

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(
        "rf_5fold_confusion_matrix.png",
        dpi=300,
    )
    plt.show()


def main() -> None:
    df = load_data()

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("=" * 70)
    print("Random Forest — Freeman Chain Code Classification")
    print("=" * 70)
    print(f"Samples: {len(df)}")
    print("\nClass distribution:")
    print(y.value_counts().to_string())

    print("\nClass encoding:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{index} = {class_name}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    prediction_rows = []

    all_actual = []
    all_predicted = []

    for fold_number, (train_index, test_index) in enumerate(
        cv.split(X, y_encoded),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y_encoded[train_index]
        y_test = y_encoded[test_index]

        encoder = build_encoder()
        X_train_encoded = encoder.fit_transform(X_train)
        X_test_encoded = encoder.transform(X_test)

        model = build_model()
        model.fit(X_train_encoded, y_train)

        y_pred = model.predict(X_test_encoded)
        y_probability = model.predict_proba(X_test_encoded)

        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        fold_recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        fold_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        fold_rows.append(
            {
                "fold": fold_number,
                "accuracy": fold_accuracy,
                "precision_weighted": fold_precision,
                "recall_weighted": fold_recall,
                "f1_weighted": fold_f1,
                "test_size": len(test_index),
            }
        )

        predicted_labels = label_encoder.inverse_transform(y_pred)
        actual_labels = label_encoder.inverse_transform(y_test)

        for local_position, dataframe_index in enumerate(test_index):
            row = df.iloc[dataframe_index]

            prediction_row = row.to_dict()
            prediction_row.update(
                {
                    "fold": fold_number,
                    "actual_label": actual_labels[local_position],
                    "predicted_label": predicted_labels[local_position],
                    "correct_prediction": (
                        actual_labels[local_position]
                        == predicted_labels[local_position]
                    ),
                }
            )

            for class_index, class_name in enumerate(
                label_encoder.classes_
            ):
                prediction_row[
                    f"probability_{class_name.lower()}"
                ] = y_probability[local_position, class_index]

            prediction_rows.append(prediction_row)

        all_actual.extend(y_test.tolist())
        all_predicted.extend(y_pred.tolist())

        print("\n" + "=" * 60)
        print(f"Fold {fold_number}")
        print("=" * 60)
        print(f"Test samples: {len(test_index)}")
        print(f"Accuracy:  {fold_accuracy:.4f}")
        print(f"Precision: {fold_precision:.4f}")
        print(f"Recall:    {fold_recall:.4f}")
        print(f"F1-score:  {fold_f1:.4f}")

    metrics_df = pd.DataFrame(fold_rows)

    print("\n" + "=" * 70)
    print("5-Fold Stratified Cross-Validation Summary")
    print("=" * 70)

    for metric in [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]:
        print(
            f"{metric}: "
            f"{metrics_df[metric].mean():.4f} "
            f"± {metrics_df[metric].std(ddof=1):.4f}"
        )

    all_actual = np.asarray(all_actual)
    all_predicted = np.asarray(all_predicted)

    print("\n" + "=" * 70)
    print("Overall Out-of-Fold Classification Report")
    print("=" * 70)
    print(
        classification_report(
            all_actual,
            all_predicted,
            target_names=label_encoder.classes_,
            digits=4,
            zero_division=0,
        )
    )

    overall_matrix = confusion_matrix(
        all_actual,
        all_predicted,
    )

    print("=" * 70)
    print("Overall Out-of-Fold Confusion Matrix")
    print("=" * 70)
    print(overall_matrix)

    metrics_df.to_csv(
        "rf_5fold_metrics.csv",
        index=False,
    )

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df = predictions_df.sort_values("fringe_id")
    predictions_df.to_csv(
        "rf_out_of_fold_predictions.csv",
        index=False,
    )

    save_confusion_matrix(
        overall_matrix,
        list(label_encoder.classes_),
    )

    # Train the final model on all 123 fringes.
    final_encoder = build_encoder()
    X_all_encoded = final_encoder.fit_transform(X)

    final_model = build_model()
    final_model.fit(X_all_encoded, y_encoded)

    final_predictions = final_model.predict(X_all_encoded)
    final_probabilities = final_model.predict_proba(X_all_encoded)

    final_prediction_df = df.copy()
    final_prediction_df["predicted_label"] = (
        label_encoder.inverse_transform(final_predictions)
    )
    final_prediction_df["correct_prediction"] = (
        final_prediction_df[TARGET_COLUMN]
        == final_prediction_df["predicted_label"]
    )

    for class_index, class_name in enumerate(label_encoder.classes_):
        final_prediction_df[
            f"probability_{class_name.lower()}"
        ] = final_probabilities[:, class_index]

    final_prediction_df.to_csv(
        "final_rf_training_predictions.csv",
        index=False,
    )

    joblib.dump(
        final_model,
        "final_random_forest_model.joblib",
    )
    joblib.dump(
        final_encoder,
        "final_rf_feature_encoder.joblib",
    )
    joblib.dump(
        label_encoder,
        "final_rf_label_encoder.joblib",
    )

    encoded_feature_names = final_encoder.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "encoded_feature": encoded_feature_names,
            "random_forest_importance": (
                final_model.feature_importances_
            ),
        }
    ).sort_values(
        "random_forest_importance",
        ascending=False,
    )

    importance_df.to_csv(
        "rf_encoded_feature_importance.csv",
        index=False,
    )

    print("\nSaved files:")
    print("rf_5fold_metrics.csv")
    print("rf_out_of_fold_predictions.csv")
    print("rf_5fold_confusion_matrix.png")
    print("final_random_forest_model.joblib")
    print("final_rf_feature_encoder.joblib")
    print("final_rf_label_encoder.joblib")
    print("final_rf_training_predictions.csv")
    print("rf_encoded_feature_importance.csv")

    print(
        "\nUse the cross-validation results—not the full-dataset "
        "training predictions—as the Random Forest performance estimate."
    )


if __name__ == "__main__":
    main()