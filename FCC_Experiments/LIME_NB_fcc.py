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
MODEL_FILE = Path("final_categorical_nb_model.joblib")
FEATURE_ENCODER_FILE = Path("final_feature_encoder.joblib")
LABEL_ENCODER_FILE = Path("final_label_encoder.joblib")

FEATURE_COLUMNS = [
    "chain_length_bin",
    "direction_change_bin",
    "fuzziness_bin",
]

CLASS_TO_EXPLAIN = "Keep"

RANDOM_STATE = 42
NUM_FEATURES = 3
NUM_SAMPLES = 5000
STABILITY_RUNS = 5
SPARSITY_THRESHOLD = 1e-6


def load_artifacts():
    """Load the labeled data, model, and encoders."""
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

    missing_columns = set(FEATURE_COLUMNS + ["label"]).difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()

    X_original = df[FEATURE_COLUMNS].copy()
    X_encoded = feature_encoder.transform(X_original).astype(int)

    class_names = list(label_encoder.classes_)
    if CLASS_TO_EXPLAIN not in class_names:
        raise ValueError(
            f"Class '{CLASS_TO_EXPLAIN}' was not found. "
            f"Available classes: {class_names}"
        )

    keep_class_index = class_names.index(CLASS_TO_EXPLAIN)

    return (
        df,
        X_original,
        X_encoded,
        model,
        feature_encoder,
        label_encoder,
        class_names,
        keep_class_index,
    )


def make_probability_function(model) -> Callable[[np.ndarray], np.ndarray]:
    
    max_categories = np.asarray(model.n_categories_, dtype=int) - 1

    def predict_proba(samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=float)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        values = np.rint(values).astype(int)
        values = np.clip(values, 0, max_categories)

        return model.predict_proba(values)

    return predict_proba


def category_names_from_encoder(feature_encoder) -> dict[int, list[str]]:
    """Create LIME category-name mappings for each encoded feature."""
    return {
        feature_index: [
            str(category)
            for category in categories
        ]
        for feature_index, categories in enumerate(
            feature_encoder.categories_
        )
    }


def create_explainer(
    X_encoded: np.ndarray,
    feature_encoder,
    class_names: list[str],
    random_state: int,
) -> LimeTabularExplainer:
    """Create a categorical LIME tabular explainer."""
    categorical_features = list(range(len(FEATURE_COLUMNS)))

    return LimeTabularExplainer(
        training_data=X_encoded.astype(float),
        feature_names=FEATURE_COLUMNS,
        class_names=class_names,
        categorical_features=categorical_features,
        categorical_names=category_names_from_encoder(feature_encoder),
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
    """Select the same case types used in SHAP."""
    predicted_encoded = model.predict(X_encoded)
    predicted_labels = label_encoder.inverse_transform(predicted_encoded)
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
        selections["misclassified"] = int(misclassified[0])

    return selections


def explanation_vector(
    explanation,
    class_index: int,
    number_of_features: int,
) -> np.ndarray:
    """
    Convert LIME's sparse local explanation into a fixed feature vector.
    """
    vector = np.zeros(number_of_features, dtype=float)

    for feature_index, contribution in explanation.local_exp[class_index]:
        vector[int(feature_index)] = float(contribution)

    return vector


def lime_intercept(
    explanation,
    class_index: int,
) -> float:
    """Extract the local linear model intercept for one class."""
    intercept = explanation.intercept

    if isinstance(intercept, dict):
        return float(intercept[class_index])

    values = np.asarray(intercept, dtype=float).reshape(-1)
    return float(values[class_index])


def readable_feature_labels(row: pd.Series) -> list[str]:
    """Create readable state labels for plots."""
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
    """Save a local contribution plot for P(Keep)."""
    labels = np.asarray(readable_feature_labels(row))
    order = np.argsort(np.abs(contribution_vector))

    ordered_labels = labels[order]
    ordered_values = contribution_vector[order]

    plt.figure(figsize=(9, 5))
    plt.barh(
        ordered_labels,
        ordered_values,
    )
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("LIME contribution to P(Keep)")
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
        data_row=encoded_instance.astype(float),
        predict_fn=predict_proba_function,
        labels=(keep_class_index,),
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES,
    )


def calculate_stability(
    X_encoded: np.ndarray,
    feature_encoder,
    class_names: list[str],
    predict_proba_function: Callable[[np.ndarray], np.ndarray],
    selected_indices: list[int],
    keep_class_index: int,
) -> tuple[float, float, pd.DataFrame]:
    """
    Repeat LIME with different random seeds and compare contribution vectors.
    """
    rows = []
    all_similarities: list[float] = []

    for sample_index in selected_indices:
        repeated_vectors = []

        for run in range(STABILITY_RUNS):
            explainer = create_explainer(
                X_encoded,
                feature_encoder,
                class_names,
                random_state=RANDOM_STATE + run,
            )

            explanation = explain_instance(
                explainer,
                predict_proba_function,
                X_encoded[sample_index],
                keep_class_index,
            )

            vector = explanation_vector(
                explanation,
                keep_class_index,
                len(FEATURE_COLUMNS),
            )
            repeated_vectors.append(vector)

        repeated_matrix = np.asarray(repeated_vectors)
        similarity_matrix = cosine_similarity(repeated_matrix)

        pairwise_values = similarity_matrix[
            np.triu_indices(
                STABILITY_RUNS,
                k=1,
            )
        ]

        mean_similarity = float(np.mean(pairwise_values))
        all_similarities.extend(pairwise_values.tolist())

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
        model,
        feature_encoder,
        label_encoder,
        class_names,
        keep_class_index,
    ) = load_artifacts()

    predict_proba_function = make_probability_function(model)

    explainer = create_explainer(
        X_encoded,
        feature_encoder,
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
        "keep": "lime_local_keep.png",
        "discard": "lime_local_discard.png",
        "misclassified": "lime_local_misclassified.png",
    }

    explanation_rows = []
    contribution_vectors = []

    print("=" * 70)
    print("LIME Analysis — Categorical Naive Bayes")
    print("=" * 70)
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Explained class: {CLASS_TO_EXPLAIN}")
    print(f"LIME perturbation samples: {NUM_SAMPLES}")

    for selection_name, sample_index in selections.items():
        explanation = explain_instance(
            explainer,
            predict_proba_function,
            X_encoded[sample_index],
            keep_class_index,
        )

        contributions = explanation_vector(
            explanation,
            keep_class_index,
            len(FEATURE_COLUMNS),
        )
        contribution_vectors.append(contributions)

        intercept = lime_intercept(
            explanation,
            keep_class_index,
        )

        surrogate_probability = float(
            intercept + np.sum(contributions)
        )

        model_probability = float(
            predict_proba_function(
                X_encoded[sample_index : sample_index + 1]
            )[0, keep_class_index]
        )

        fidelity_error = abs(
            model_probability - surrogate_probability
        )

        sparsity = float(
            np.mean(
                np.abs(contributions) <= SPARSITY_THRESHOLD
            )
        )

        predicted_label = label_encoder.inverse_transform(
            model.predict(
                X_encoded[sample_index : sample_index + 1]
            )
        )[0]

        actual_label = str(
            df.iloc[sample_index]["label"]
        )

        fringe_id = (
            int(df.iloc[sample_index]["fringe_id"])
            if "fringe_id" in df.columns
            else sample_index + 1
        )

        title = (
            f"LIME Local Explanation — Fringe {fringe_id}\n"
            f"Actual: {actual_label}, Predicted: {predicted_label}"
        )

        save_local_plot(
            output_filename=output_names[selection_name],
            title=title,
            row=df.iloc[sample_index],
            contribution_vector=contributions,
            model_probability=model_probability,
            surrogate_probability=surrogate_probability,
        )

        row_data = {
            "case_type": selection_name,
            "sample_index": sample_index,
            "fringe_id": fringe_id,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "model_probability_keep": model_probability,
            "lime_intercept_keep": intercept,
            "lime_surrogate_probability_keep": surrogate_probability,
            "absolute_fidelity_error": fidelity_error,
            "sparsity": sparsity,
        }

        for feature_name, contribution in zip(
            FEATURE_COLUMNS,
            contributions,
        ):
            row_data[f"lime_{feature_name}"] = contribution

        explanation_rows.append(row_data)

    explanation_df = pd.DataFrame(explanation_rows)
    explanation_df.to_csv(
        "lime_explanations.csv",
        index=False,
    )

    selected_indices = list(selections.values())

    overall_stability, stability_std, stability_df = calculate_stability(
        X_encoded,
        feature_encoder,
        class_names,
        predict_proba_function,
        selected_indices,
        keep_class_index,
    )

    stability_df.to_csv(
        "lime_stability_details.csv",
        index=False,
    )

    mean_fidelity_error = float(
        explanation_df["absolute_fidelity_error"].mean()
    )
    max_fidelity_error = float(
        explanation_df["absolute_fidelity_error"].max()
    )
    mean_sparsity = float(
        explanation_df["sparsity"].mean()
    )

    metrics_df = pd.DataFrame(
        [
            {
                "method": "LIME Tabular",
                "explained_class": CLASS_TO_EXPLAIN,
                "mean_fidelity_error": mean_fidelity_error,
                "max_fidelity_error": max_fidelity_error,
                "mean_stability_cosine_similarity": overall_stability,
                "stability_cosine_similarity_std": stability_std,
                "mean_sparsity": mean_sparsity,
                "sparsity_threshold": SPARSITY_THRESHOLD,
                "representative_samples": len(selected_indices),
                "stability_runs": STABILITY_RUNS,
                "lime_num_samples": NUM_SAMPLES,
            }
        ]
    )

    metrics_df.to_csv(
        "lime_metrics.csv",
        index=False,
    )

    print("\nRepresentative explanations:")
    print(explanation_df.to_string(index=False))

    print("\nExplanation metrics:")
    print(f"Mean fidelity error: {mean_fidelity_error:.8f}")
    print(f"Maximum fidelity error: {max_fidelity_error:.8f}")
    print(
        "Mean stability cosine similarity: "
        f"{overall_stability:.8f}"
    )
    print(
        "Stability cosine similarity standard deviation: "
        f"{stability_std:.8f}"
    )
    print(f"Mean sparsity: {mean_sparsity:.8f}")

    print("\nSaved files:")
    print("lime_explanations.csv")
    print("lime_metrics.csv")
    print("lime_stability_details.csv")

    for output_name in output_names.values():
        if Path(output_name).exists():
            print(output_name)


if __name__ == "__main__":
    main()