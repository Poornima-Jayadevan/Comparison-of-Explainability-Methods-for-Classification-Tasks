from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = Path("labeled_fringes.csv")
MODEL_FILE = Path("final_random_forest_model.joblib")
FEATURE_ENCODER_FILE = Path("final_rf_feature_encoder.joblib")
LABEL_ENCODER_FILE = Path("final_rf_label_encoder.joblib")

ORIGINAL_FEATURE_COLUMNS = [
    "chain_length_bin",
    "direction_change_bin",
    "fuzziness_bin",
]

READABLE_FEATURE_NAMES = [
    "Chain Length",
    "Direction Change",
    "Fuzziness",
]

EXPLAINED_CLASS = "Keep"
RANDOM_STATE = 42
NUM_FEATURES = 3
NUM_SAMPLES = 5000
STABILITY_RUNS = 5
SPARSITY_THRESHOLD = 1e-6


def load_artifacts():
    """Load the labeled data, Random Forest model, and encoders."""
    required_files = [
        DATA_FILE,
        MODEL_FILE,
        FEATURE_ENCODER_FILE,
        LABEL_ENCODER_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_path} was not found. Keep all required files in "
                "the same folder as this script."
            )

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    feature_encoder = joblib.load(FEATURE_ENCODER_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    required_columns = set(
        ORIGINAL_FEATURE_COLUMNS + ["label"]
    )
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.dropna(
        subset=ORIGINAL_FEATURE_COLUMNS + ["label"]
    ).copy()

    X_original = df[ORIGINAL_FEATURE_COLUMNS].copy()
    X_encoded = feature_encoder.transform(X_original)

    encoded_feature_names = list(
        feature_encoder.get_feature_names_out()
    )

    class_names = list(label_encoder.classes_)

    if EXPLAINED_CLASS not in class_names:
        raise ValueError(
            f"Class '{EXPLAINED_CLASS}' was not found. "
            f"Available classes: {class_names}"
        )

    keep_class_index = class_names.index(EXPLAINED_CLASS)

    return (
        df,
        X_original,
        np.asarray(X_encoded, dtype=float),
        encoded_feature_names,
        model,
        feature_encoder,
        label_encoder,
        class_names,
        keep_class_index,
    )


def make_prediction_function(model) -> Callable[[np.ndarray], np.ndarray]:
    """Return Random Forest probability predictions for LIME."""
    def predict_proba(samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=float)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        return model.predict_proba(values)

    return predict_proba


def build_encoded_groups(
    encoded_feature_names: list[str],
) -> dict[str, list[int]]:
    """Map one-hot encoded columns back to the three original features."""
    groups = {
        "Chain Length": [],
        "Direction Change": [],
        "Fuzziness": [],
    }

    for index, name in enumerate(encoded_feature_names):
        if name.startswith("chain_length_bin_"):
            groups["Chain Length"].append(index)
        elif name.startswith("direction_change_bin_"):
            groups["Direction Change"].append(index)
        elif name.startswith("fuzziness_bin_"):
            groups["Fuzziness"].append(index)

    for feature_name, indices in groups.items():
        if not indices:
            raise ValueError(
                f"No encoded columns found for {feature_name}."
            )

    return groups


def create_lime_explainer(
    X_encoded: np.ndarray,
    encoded_feature_names: list[str],
    class_names: list[str],
    random_state: int,
) -> LimeTabularExplainer:
    """Create a LIME explainer for the one-hot encoded Random Forest input."""
    return LimeTabularExplainer(
        training_data=X_encoded,
        feature_names=encoded_feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=False,
        sample_around_instance=True,
        random_state=random_state,
    )


def select_representative_indices(
    df: pd.DataFrame,
    model,
    X_encoded: np.ndarray,
    label_encoder,
) -> dict[str, int]:
    """Select one correct Keep, one correct Discard, and one misclassified case."""
    predicted_encoded = model.predict(X_encoded)
    predicted_labels = label_encoder.inverse_transform(
        predicted_encoded
    )
    actual_labels = df["label"].to_numpy()

    selections: dict[str, int] = {}

    correct_keep = np.flatnonzero(
        (actual_labels == "Keep")
        & (predicted_labels == "Keep")
    )
    correct_discard = np.flatnonzero(
        (actual_labels == "Discard")
        & (predicted_labels == "Discard")
    )
    misclassified = np.flatnonzero(
        actual_labels != predicted_labels
    )

    if correct_keep.size:
        selections["keep"] = int(correct_keep[0])

    if correct_discard.size:
        selections["discard"] = int(correct_discard[0])

    if misclassified.size:
        selections["misclassified"] = int(
            misclassified[0]
        )

    return selections


def encoded_explanation_vector(
    explanation,
    class_index: int,
    number_of_encoded_features: int,
) -> np.ndarray:
    """Convert LIME local explanation into a fixed one-hot contribution vector."""
    vector = np.zeros(
        number_of_encoded_features,
        dtype=float,
    )

    for feature_index, contribution in explanation.local_exp[class_index]:
        vector[int(feature_index)] = float(contribution)

    return vector


def aggregate_lime_vector(
    encoded_vector: np.ndarray,
    groups: dict[str, list[int]],
) -> np.ndarray:
    """Aggregate one-hot LIME contributions into three original FCC features."""
    aggregated = []

    for feature_name in READABLE_FEATURE_NAMES:
        aggregated.append(
            float(np.sum(encoded_vector[groups[feature_name]]))
        )

    return np.asarray(aggregated, dtype=float)


def get_lime_intercept(
    explanation,
    class_index: int,
) -> float:
    """Return the LIME local surrogate intercept for the explained class."""
    intercept = explanation.intercept

    if isinstance(intercept, dict):
        return float(intercept[class_index])

    values = np.asarray(
        intercept,
        dtype=float,
    ).reshape(-1)

    return float(values[class_index])


def readable_state_labels(row: pd.Series) -> list[str]:
    """Create readable local feature-state labels."""
    return [
        f"Length = {row['chain_length_bin']}",
        f"Direction change = {row['direction_change_bin']}",
        f"Fuzziness = {row['fuzziness_bin']}",
    ]


def save_local_plot(
    output_filename: str,
    title: str,
    row: pd.Series,
    contribution_vector: np.ndarray,
    model_probability: float,
    surrogate_probability: float,
) -> None:
    """Save an aggregated local LIME contribution plot."""
    labels = np.asarray(readable_state_labels(row))
    order = np.argsort(np.abs(contribution_vector))

    plt.figure(figsize=(9, 5))
    plt.barh(
        labels[order],
        contribution_vector[order],
    )
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Aggregated LIME contribution to P(Keep)")
    plt.title(
        f"{title}\n"
        f"Model P(Keep) = {model_probability:.4f}, "
        f"LIME surrogate = {surrogate_probability:.4f}"
    )
    plt.tight_layout()
    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def explain_instance(
    explainer: LimeTabularExplainer,
    predict_proba_function: Callable[[np.ndarray], np.ndarray],
    encoded_instance: np.ndarray,
    keep_class_index: int,
):
    """Generate one LIME explanation."""
    return explainer.explain_instance(
        data_row=encoded_instance,
        predict_fn=predict_proba_function,
        labels=(keep_class_index,),
        num_features=len(encoded_instance),
        num_samples=NUM_SAMPLES,
    )


def calculate_stability(
    X_encoded: np.ndarray,
    encoded_feature_names: list[str],
    class_names: list[str],
    predict_proba_function: Callable[[np.ndarray], np.ndarray],
    selected_indices: list[int],
    keep_class_index: int,
    groups: dict[str, list[int]],
) -> tuple[float, float, pd.DataFrame]:
    """Repeat LIME with different seeds and compare aggregated vectors."""
    rows = []
    all_similarities = []

    for sample_index in selected_indices:
        repeated_vectors = []

        for run in range(STABILITY_RUNS):
            explainer = create_lime_explainer(
                X_encoded,
                encoded_feature_names,
                class_names,
                random_state=RANDOM_STATE + run,
            )

            explanation = explain_instance(
                explainer,
                predict_proba_function,
                X_encoded[sample_index],
                keep_class_index,
            )

            encoded_vector = encoded_explanation_vector(
                explanation,
                keep_class_index,
                X_encoded.shape[1],
            )

            aggregated_vector = aggregate_lime_vector(
                encoded_vector,
                groups,
            )

            repeated_vectors.append(aggregated_vector)

        repeated_matrix = np.asarray(
            repeated_vectors,
            dtype=float,
        )

        similarity_matrix = cosine_similarity(
            repeated_matrix
        )

        upper_triangle = similarity_matrix[
            np.triu_indices(
                STABILITY_RUNS,
                k=1,
            )
        ]

        mean_similarity = float(
            np.mean(upper_triangle)
        )

        all_similarities.extend(
            upper_triangle.tolist()
        )

        rows.append(
            {
                "sample_index": sample_index,
                "mean_cosine_similarity": mean_similarity,
            }
        )

    overall_stability = (
        float(np.mean(all_similarities))
        if all_similarities
        else np.nan
    )

    stability_std = (
        float(np.std(all_similarities, ddof=1))
        if len(all_similarities) > 1
        else np.nan
    )

    return overall_stability, stability_std, pd.DataFrame(rows)


def main() -> None:
    (
        df,
        X_original,
        X_encoded,
        encoded_feature_names,
        model,
        feature_encoder,
        label_encoder,
        class_names,
        keep_class_index,
    ) = load_artifacts()

    groups = build_encoded_groups(
        encoded_feature_names
    )

    predict_proba_function = make_prediction_function(
        model
    )

    explainer = create_lime_explainer(
        X_encoded,
        encoded_feature_names,
        class_names,
        random_state=RANDOM_STATE,
    )

    selections = select_representative_indices(
        df,
        model,
        X_encoded,
        label_encoder,
    )

    output_names = {
        "keep": "rf_lime_local_keep.png",
        "discard": "rf_lime_local_discard.png",
        "misclassified": "rf_lime_local_misclassified.png",
    }

    explanation_rows = []

    print("=" * 72)
    print("Random Forest LIME Analysis")
    print("=" * 72)
    print(f"Encoded features: {len(encoded_feature_names)}")
    print(
        f"Aggregated features: {READABLE_FEATURE_NAMES}"
    )
    print(f"Explained class: {EXPLAINED_CLASS}")
    print(f"LIME samples per explanation: {NUM_SAMPLES}")

    for case_type, sample_index in selections.items():
        explanation = explain_instance(
            explainer,
            predict_proba_function,
            X_encoded[sample_index],
            keep_class_index,
        )

        encoded_vector = encoded_explanation_vector(
            explanation,
            keep_class_index,
            X_encoded.shape[1],
        )

        aggregated_vector = aggregate_lime_vector(
            encoded_vector,
            groups,
        )

        intercept = get_lime_intercept(
            explanation,
            keep_class_index,
        )

        surrogate_probability = float(
            intercept + np.sum(encoded_vector)
        )

        model_probability = float(
            predict_proba_function(
                X_encoded[
                    sample_index : sample_index + 1
                ]
            )[0, keep_class_index]
        )

        fidelity_error = abs(
            model_probability - surrogate_probability
        )

        sparsity = float(
            np.mean(
                np.abs(aggregated_vector)
                <= SPARSITY_THRESHOLD
            )
        )

        predicted_label = (
            label_encoder.inverse_transform(
                model.predict(
                    X_encoded[
                        sample_index : sample_index + 1
                    ]
                )
            )[0]
        )

        actual_label = str(
            df.iloc[sample_index]["label"]
        )

        fringe_id = (
            int(df.iloc[sample_index]["fringe_id"])
            if "fringe_id" in df.columns
            else sample_index + 1
        )

        title = (
            f"Random Forest LIME — Fringe {fringe_id}\n"
            f"Actual: {actual_label}, "
            f"Predicted: {predicted_label}"
        )

        save_local_plot(
            output_filename=output_names[case_type],
            title=title,
            row=df.iloc[sample_index],
            contribution_vector=aggregated_vector,
            model_probability=model_probability,
            surrogate_probability=surrogate_probability,
        )

        row_data = {
            "case_type": case_type,
            "sample_index": sample_index,
            "fringe_id": fringe_id,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "model_probability_keep": model_probability,
            "lime_intercept_keep": intercept,
            "lime_surrogate_probability_keep": (
                surrogate_probability
            ),
            "absolute_fidelity_error": fidelity_error,
            "sparsity": sparsity,
            "lime_chain_length": (
                aggregated_vector[0]
            ),
            "lime_direction_change": (
                aggregated_vector[1]
            ),
            "lime_fuzziness": (
                aggregated_vector[2]
            ),
        }

        explanation_rows.append(row_data)

    explanation_df = pd.DataFrame(
        explanation_rows
    )

    explanation_df.to_csv(
        "rf_lime_explanations.csv",
        index=False,
    )

    selected_indices = list(
        selections.values()
    )

    overall_stability, stability_std, stability_df = (
        calculate_stability(
            X_encoded,
            encoded_feature_names,
            class_names,
            predict_proba_function,
            selected_indices,
            keep_class_index,
            groups,
        )
    )

    stability_df.to_csv(
        "rf_lime_stability_details.csv",
        index=False,
    )

    mean_fidelity_error = float(
        explanation_df[
            "absolute_fidelity_error"
        ].mean()
    )

    max_fidelity_error = float(
        explanation_df[
            "absolute_fidelity_error"
        ].max()
    )

    mean_sparsity = float(
        explanation_df["sparsity"].mean()
    )

    metrics_df = pd.DataFrame(
        [
            {
                "method": (
                    "Random Forest LIME Aggregated"
                ),
                "explained_class": EXPLAINED_CLASS,
                "mean_fidelity_error": (
                    mean_fidelity_error
                ),
                "max_fidelity_error": (
                    max_fidelity_error
                ),
                (
                    "mean_stability_cosine_similarity"
                ): overall_stability,
                "stability_cosine_similarity_std": stability_std,
                "mean_sparsity": mean_sparsity,
                "sparsity_threshold": (
                    SPARSITY_THRESHOLD
                ),
                "representative_samples": len(
                    selected_indices
                ),
                "stability_runs": STABILITY_RUNS,
                "lime_num_samples": NUM_SAMPLES,
                "encoded_features": len(
                    encoded_feature_names
                ),
                "aggregated_features": len(
                    READABLE_FEATURE_NAMES
                ),
            }
        ]
    )

    metrics_df.to_csv(
        "rf_lime_metrics.csv",
        index=False,
    )

    print("\nRepresentative explanations:")
    print(
        explanation_df.to_string(index=False)
    )

    print("\nExplanation metrics:")
    print(
        f"Mean fidelity error: "
        f"{mean_fidelity_error:.8f}"
    )
    print(
        f"Maximum fidelity error: "
        f"{max_fidelity_error:.8f}"
    )
    print(
        "Mean stability cosine similarity: "
        f"{overall_stability:.8f}"
    )
    print(
        "Stability cosine similarity standard deviation: "
        f"{stability_std:.8f}"
    )
    print(
        f"Mean sparsity: "
        f"{mean_sparsity:.8f}"
    )

    print("\nSaved files:")
    print("rf_lime_explanations.csv")
    print("rf_lime_metrics.csv")
    print("rf_lime_stability_details.csv")

    for output_name in output_names.values():
        if Path(output_name).exists():
            print(output_name)


if __name__ == "__main__":
    main()