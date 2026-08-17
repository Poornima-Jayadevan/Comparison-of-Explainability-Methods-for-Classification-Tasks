from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = Path("labeled_fringes.csv")
MODEL_FILE = Path("final_random_forest_model.joblib")
FEATURE_ENCODER_FILE = Path("final_rf_feature_encoder.joblib")
LABEL_ENCODER_FILE = Path("final_rf_label_encoder.joblib")
TRAINING_PREDICTIONS_FILE = Path("final_rf_training_predictions.csv")

ORIGINAL_FEATURE_COLUMNS = [
    "chain_length_bin",
    "direction_change_bin",
    "fuzziness_bin",
]

AGGREGATED_FEATURE_NAMES = [
    "Chain Length",
    "Direction Change",
    "Fuzziness",
]

EXPLAINED_CLASS = "Keep"
SPARSITY_THRESHOLD = 1e-6
STABILITY_RUNS = 5


def load_artifacts():
    """Load dataset, trained Random Forest, and encoders."""
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
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
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
        keep_class_index,
    )


def build_aggregation_groups(
    encoded_feature_names: list[str],
) -> dict[str, list[int]]:
    """
    Identify which one-hot columns belong to each original FCC feature.
    """
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
                f"No encoded columns were found for '{feature_name}'. "
                f"Encoded feature names were: {encoded_feature_names}"
            )

    return groups


def get_keep_class_shap_values(
    model,
    X_encoded: np.ndarray,
    keep_class_index: int,
) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values for P(Keep) using TreeExplainer.
    """
    explainer = shap.TreeExplainer(
        model,
        model_output="raw",
    )

    raw_values = explainer.shap_values(X_encoded)

    if isinstance(raw_values, list):
        shap_values = np.asarray(
            raw_values[keep_class_index],
            dtype=float,
        )
    else:
        values = np.asarray(raw_values, dtype=float)

        if values.ndim == 3:
            shap_values = values[:, :, keep_class_index]
        elif values.ndim == 2:
            shap_values = values
        else:
            raise ValueError(
                f"Unexpected SHAP value shape: {values.shape}"
            )

    expected_value = np.asarray(
        explainer.expected_value,
        dtype=float,
    )

    if expected_value.ndim == 0:
        keep_expected_value = float(expected_value)
    else:
        keep_expected_value = float(
            expected_value.reshape(-1)[keep_class_index]
        )

    return shap_values, keep_expected_value


def aggregate_shap_values(
    encoded_shap_values: np.ndarray,
    aggregation_groups: dict[str, list[int]],
) -> np.ndarray:
    """
    Sum one-hot SHAP values belonging to each original FCC feature.
    """
    aggregated_columns = []

    for feature_name in AGGREGATED_FEATURE_NAMES:
        indices = aggregation_groups[feature_name]
        aggregated_columns.append(
            encoded_shap_values[:, indices].sum(axis=1)
        )

    return np.column_stack(aggregated_columns)


def aggregate_encoded_feature_values(
    X_original: pd.DataFrame,
) -> np.ndarray:
    """
    Convert categorical states into ordered numeric values for the summary plot.
    """
    mappings = {
        "chain_length_bin": {
            "Short": 0,
            "Medium": 1,
            "Long": 2,
        },
        "direction_change_bin": {
            "Low": 0,
            "Medium": 1,
            "High": 2,
        },
        "fuzziness_bin": {
            "Low": 0,
            "Medium": 1,
            "High": 2,
        },
    }

    values = []

    for feature_name in ORIGINAL_FEATURE_COLUMNS:
        mapped = X_original[feature_name].map(
            mappings[feature_name]
        )

        if mapped.isna().any():
            invalid = X_original.loc[
                mapped.isna(),
                feature_name,
            ].unique()
            raise ValueError(
                f"Unexpected states in {feature_name}: {invalid}"
            )

        values.append(mapped.to_numpy(dtype=float))

    return np.column_stack(values)


def save_global_importance(
    aggregated_shap_values: np.ndarray,
) -> pd.DataFrame:
    """Save and display aggregated global SHAP importance."""
    importance = np.mean(
        np.abs(aggregated_shap_values),
        axis=0,
    )

    importance_df = pd.DataFrame(
        {
            "feature": AGGREGATED_FEATURE_NAMES,
            "mean_absolute_shap": importance,
        }
    ).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance_df.to_csv(
        "rf_shap_aggregated_global_importance.csv",
        index=False,
    )

    plot_df = importance_df.sort_values(
        "mean_absolute_shap",
        ascending=True,
    )

    plt.figure(figsize=(8, 5))
    plt.barh(
        plot_df["feature"],
        plot_df["mean_absolute_shap"],
    )
    plt.xlabel("Mean absolute aggregated SHAP value for P(Keep)")
    plt.ylabel("Original FCC feature")
    plt.title("Random Forest — Aggregated Global SHAP Importance")
    plt.tight_layout()
    plt.savefig(
        "rf_shap_aggregated_global_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return importance_df


def save_summary_plot(
    aggregated_shap_values: np.ndarray,
    aggregated_feature_values: np.ndarray,
) -> None:
    """Create a three-feature aggregated SHAP summary plot."""
    plt.figure()

    shap.summary_plot(
        aggregated_shap_values,
        aggregated_feature_values,
        feature_names=AGGREGATED_FEATURE_NAMES,
        show=False,
    )

    plt.title(
        "Random Forest — Aggregated SHAP Summary for P(Keep)"
    )
    plt.tight_layout()
    plt.savefig(
        "rf_shap_aggregated_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def format_feature_states(row: pd.Series) -> list[str]:
    """Create readable feature-state labels for local plots."""
    return [
        f"Length = {row['chain_length_bin']}",
        (
            "Direction change = "
            f"{row['direction_change_bin']}"
        ),
        f"Fuzziness = {row['fuzziness_bin']}",
    ]


def save_local_explanation(
    output_filename: str,
    title: str,
    row: pd.Series,
    shap_vector: np.ndarray,
    base_value: float,
    model_probability: float,
) -> None:
    """Save an aggregated local SHAP contribution plot."""
    labels = np.asarray(format_feature_states(row))
    order = np.argsort(np.abs(shap_vector))

    ordered_labels = labels[order]
    ordered_values = shap_vector[order]

    reconstructed_probability = float(
        base_value + shap_vector.sum()
    )

    plt.figure(figsize=(9, 5))
    plt.barh(
        ordered_labels,
        ordered_values,
    )
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Aggregated SHAP contribution to P(Keep)")
    plt.title(
        f"{title}\n"
        f"Base = {base_value:.4f}, "
        f"SHAP reconstruction = {reconstructed_probability:.4f}, "
        f"Model P(Keep) = {model_probability:.4f}"
    )
    plt.tight_layout()
    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def select_representative_indices(
    df: pd.DataFrame,
    model,
    X_encoded: np.ndarray,
    label_encoder,
) -> dict[str, int]:
    """
    Select one correct Keep, one correct Discard, and one misclassified fringe.
    """
    predictions_encoded = model.predict(X_encoded)
    predictions = label_encoder.inverse_transform(
        predictions_encoded
    )
    actual = df["label"].to_numpy()

    selections: dict[str, int] = {}

    correct_keep = np.flatnonzero(
        (actual == "Keep")
        & (predictions == "Keep")
    )
    correct_discard = np.flatnonzero(
        (actual == "Discard")
        & (predictions == "Discard")
    )
    misclassified = np.flatnonzero(
        actual != predictions
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


def calculate_fidelity(
    aggregated_shap_values: np.ndarray,
    base_value: float,
    model_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Calculate SHAP reconstruction fidelity."""
    reconstructed = (
        base_value
        + aggregated_shap_values.sum(axis=1)
    )

    absolute_error = np.abs(
        model_probabilities - reconstructed
    )

    return pd.DataFrame(
        {
            "model_probability_keep": model_probabilities,
            "shap_reconstructed_probability_keep": reconstructed,
            "absolute_fidelity_error": absolute_error,
        }
    )


def calculate_stability(
    model,
    X_encoded: np.ndarray,
    keep_class_index: int,
    aggregation_groups: dict[str, list[int]],
    selected_indices: list[int],
) -> tuple[float, pd.DataFrame]:
    """
    Repeat TreeSHAP and compare aggregated vectors.
    """
    rows = []
    all_pairwise_similarities = []

    for sample_index in selected_indices:
        repeated_vectors = []

        for _ in range(STABILITY_RUNS):
            encoded_values, _ = get_keep_class_shap_values(
                model,
                X_encoded[
                    sample_index : sample_index + 1
                ],
                keep_class_index,
            )

            aggregated = aggregate_shap_values(
                encoded_values,
                aggregation_groups,
            )

            repeated_vectors.append(aggregated[0])

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

        all_pairwise_similarities.extend(
            upper_triangle.tolist()
        )

        rows.append(
            {
                "sample_index": sample_index,
                "mean_cosine_similarity": mean_similarity,
            }
        )

    overall_stability = (
        float(np.mean(all_pairwise_similarities))
        if all_pairwise_similarities
        else np.nan
    )

    stability_std = (
        float(np.std(all_pairwise_similarities, ddof=1))
        if len(all_pairwise_similarities) > 1
        else np.nan
    )

    return overall_stability, stability_std, pd.DataFrame(rows)


def calculate_sparsity(
    aggregated_shap_values: np.ndarray,
    threshold: float,
) -> tuple[float, np.ndarray]:
    """
    Sparsity 
    """
    per_sample = np.mean(
        np.abs(aggregated_shap_values) <= threshold,
        axis=1,
    )

    overall = float(np.mean(per_sample))

    return overall, per_sample


def main() -> None:
    (
        df,
        X_original,
        X_encoded,
        encoded_feature_names,
        model,
        feature_encoder,
        label_encoder,
        keep_class_index,
    ) = load_artifacts()

    aggregation_groups = build_aggregation_groups(
        encoded_feature_names
    )

    print("=" * 72)
    print("Random Forest Aggregated SHAP Analysis")
    print("=" * 72)
    print(f"Samples explained: {len(df)}")
    print(f"Encoded features: {len(encoded_feature_names)}")
    print(
        f"Aggregated features: "
        f"{AGGREGATED_FEATURE_NAMES}"
    )
    print(f"Explained output: P({EXPLAINED_CLASS})")

    print("\nAggregation groups:")
    for feature_name, indices in aggregation_groups.items():
        encoded_names = [
            encoded_feature_names[index]
            for index in indices
        ]
        print(
            f"{feature_name}: "
            + ", ".join(encoded_names)
        )

    encoded_shap_values, base_value = (
        get_keep_class_shap_values(
            model,
            X_encoded,
            keep_class_index,
        )
    )

    aggregated_shap_values = aggregate_shap_values(
        encoded_shap_values,
        aggregation_groups,
    )

    aggregated_feature_values = (
        aggregate_encoded_feature_values(
            X_original
        )
    )

    model_probabilities = model.predict_proba(
        X_encoded
    )[:, keep_class_index]

    fringe_ids = (
        df["fringe_id"].to_numpy()
        if "fringe_id" in df.columns
        else np.arange(1, len(df) + 1)
    )

    values_df = pd.DataFrame(
        aggregated_shap_values,
        columns=[
            "shap_chain_length",
            "shap_direction_change",
            "shap_fuzziness",
        ],
    )

    values_df.insert(
        0,
        "fringe_id",
        fringe_ids,
    )
    values_df["base_value_keep"] = base_value
    values_df["model_probability_keep"] = (
        model_probabilities
    )

    values_df.to_csv(
        "rf_shap_aggregated_values.csv",
        index=False,
    )

    importance_df = save_global_importance(
        aggregated_shap_values
    )

    save_summary_plot(
        aggregated_shap_values,
        aggregated_feature_values,
    )

    selections = select_representative_indices(
        df,
        model,
        X_encoded,
        label_encoder,
    )

    output_names = {
        "keep": "rf_shap_aggregated_local_keep.png",
        "discard": (
            "rf_shap_aggregated_local_discard.png"
        ),
        "misclassified": (
            "rf_shap_aggregated_local_misclassified.png"
        ),
    }

    for case_type, sample_index in selections.items():
        prediction_encoded = model.predict(
            X_encoded[
                sample_index : sample_index + 1
            ]
        )

        predicted_label = (
            label_encoder.inverse_transform(
                prediction_encoded
            )[0]
        )

        actual_label = str(
            df.iloc[sample_index]["label"]
        )

        fringe_id = int(fringe_ids[sample_index])

        title = (
            f"Random Forest Aggregated SHAP — "
            f"Fringe {fringe_id}\n"
            f"Actual: {actual_label}, "
            f"Predicted: {predicted_label}"
        )

        save_local_explanation(
            output_filename=output_names[case_type],
            title=title,
            row=df.iloc[sample_index],
            shap_vector=(
                aggregated_shap_values[sample_index]
            ),
            base_value=base_value,
            model_probability=float(
                model_probabilities[sample_index]
            ),
        )

    fidelity_df = calculate_fidelity(
        aggregated_shap_values,
        base_value,
        model_probabilities,
    )

    mean_fidelity_error = float(
        fidelity_df[
            "absolute_fidelity_error"
        ].mean()
    )
    max_fidelity_error = float(
        fidelity_df[
            "absolute_fidelity_error"
        ].max()
    )

    stability_indices = list(selections.values())

    overall_stability, stability_std, stability_df = (
        calculate_stability(
            model,
            X_encoded,
            keep_class_index,
            aggregation_groups,
            stability_indices,
        )
    )

    overall_sparsity, per_sample_sparsity = (
        calculate_sparsity(
            aggregated_shap_values,
            SPARSITY_THRESHOLD,
        )
    )

    fidelity_df.insert(
        0,
        "fringe_id",
        fringe_ids,
    )
    fidelity_df["sparsity"] = (
        per_sample_sparsity
    )

    fidelity_df.to_csv(
        "rf_shap_aggregated_explanation_details.csv",
        index=False,
    )

    stability_df.to_csv(
        "rf_shap_aggregated_stability_details.csv",
        index=False,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "method": (
                    "Random Forest TreeSHAP Aggregated"
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
                "mean_sparsity": overall_sparsity,
                "sparsity_threshold": (
                    SPARSITY_THRESHOLD
                ),
                "samples_explained": len(df),
                "encoded_features": len(
                    encoded_feature_names
                ),
                "aggregated_features": len(
                    AGGREGATED_FEATURE_NAMES
                ),
                "stability_samples": len(
                    stability_indices
                ),
                "stability_runs": STABILITY_RUNS,
            }
        ]
    )

    metrics_df.to_csv(
        "rf_shap_aggregated_metrics.csv",
        index=False,
    )

    print("\nAggregated global feature importance:")
    print(
        importance_df.to_string(index=False)
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
        f"{overall_sparsity:.8f}"
    )
    print(
        f"TreeSHAP base value P(Keep): "
        f"{base_value:.8f}"
    )

    print("\nSaved files:")
    print(
        "rf_shap_aggregated_global_importance.csv"
    )
    print("rf_shap_aggregated_values.csv")
    print("rf_shap_aggregated_metrics.csv")
    print(
        "rf_shap_aggregated_explanation_details.csv"
    )
    print(
        "rf_shap_aggregated_stability_details.csv"
    )
    print(
        "rf_shap_aggregated_global_bar.png"
    )
    print(
        "rf_shap_aggregated_summary.png"
    )

    for output_name in output_names.values():
        if Path(output_name).exists():
            print(output_name)


if __name__ == "__main__":
    main()