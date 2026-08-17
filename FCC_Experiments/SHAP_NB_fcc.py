from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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

RANDOM_STATE = 42
KEEP_CLASS_NAME = "Keep"
BACKGROUND_SIZE = 30
SHAP_NSAMPLES = 200
STABILITY_RUNS = 5
SPARSITY_THRESHOLD = 1e-6


def load_artifacts():
    for file_path in [DATA_FILE, MODEL_FILE, FEATURE_ENCODER_FILE, LABEL_ENCODER_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_path} was not found. Keep all required files in the same folder."
            )

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    feature_encoder = joblib.load(FEATURE_ENCODER_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    missing = set(FEATURE_COLUMNS + ["label"]).difference(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()
    X_original = df[FEATURE_COLUMNS].copy()
    X_encoded = feature_encoder.transform(X_original).astype(int)

    class_names = list(label_encoder.classes_)
    if KEEP_CLASS_NAME not in class_names:
        raise ValueError(f"Class '{KEEP_CLASS_NAME}' not found. Available: {class_names}")

    keep_class_index = class_names.index(KEEP_CLASS_NAME)
    return df, X_original, X_encoded, model, feature_encoder, label_encoder, keep_class_index


def make_keep_probability_function(model, keep_class_index: int) -> Callable[[np.ndarray], np.ndarray]:
    max_categories = np.asarray(model.n_categories_, dtype=int) - 1

    def predict_keep(encoded_samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(encoded_samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        samples = np.rint(samples).astype(int)
        samples = np.clip(samples, 0, max_categories)
        return model.predict_proba(samples)[:, keep_class_index]

    return predict_keep


def select_background(X_encoded: np.ndarray, size: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if len(X_encoded) <= size:
        return X_encoded.copy()
    indices = rng.choice(len(X_encoded), size=size, replace=False)
    return X_encoded[indices]


def compute_shap_values(prediction_function, background, samples, nsamples):
    explainer = shap.KernelExplainer(prediction_function, background)
    raw_values = explainer.shap_values(samples, nsamples=nsamples, silent=True)

    if isinstance(raw_values, list):
        values = np.asarray(raw_values[0], dtype=float)
    else:
        values = np.asarray(raw_values, dtype=float)

    values = np.squeeze(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    expected = np.asarray(explainer.expected_value, dtype=float).squeeze()
    expected_value = float(np.ravel(expected)[0]) if np.ndim(expected) > 0 else float(expected)
    return values, expected_value


def save_global_importance(shap_values: np.ndarray) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_absolute_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    importance_df.to_csv("shap_global_importance.csv", index=False)

    plot_df = importance_df.sort_values("mean_absolute_shap")
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature"], plot_df["mean_absolute_shap"])
    plt.xlabel("Mean absolute SHAP value for P(Keep)")
    plt.ylabel("Feature")
    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig("shap_global_bar.png", dpi=300)
    plt.show()
    return importance_df


def save_summary_plot(shap_values: np.ndarray, X_encoded: np.ndarray) -> None:
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_encoded,
        feature_names=FEATURE_COLUMNS,
        show=False,
    )
    plt.title("SHAP Summary — Keep-Class Probability")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
    plt.show()


def format_feature_states(row: pd.Series) -> list[str]:
    return [
        f"Length = {row['chain_length_bin']}",
        f"Direction change = {row['direction_change_bin']}",
        f"Fuzziness = {row['fuzziness_bin']}",
    ]


def save_local_explanation(output_filename, title, row, shap_vector, base_value, predicted_probability):
    labels = np.asarray(format_feature_states(row))
    order = np.argsort(np.abs(shap_vector))
    plt.figure(figsize=(9, 5))
    plt.barh(labels[order], shap_vector[order])
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("SHAP contribution to P(Keep)")
    plt.title(
        f"{title}\nBase P(Keep) = {base_value:.4f}, "
        f"Predicted P(Keep) = {predicted_probability:.4f}"
    )
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.show()


def select_representative_indices(df, model, X_encoded, label_encoder):
    predicted = label_encoder.inverse_transform(model.predict(X_encoded))
    actual = df["label"].to_numpy()
    selections = {}

    correct_keep = np.flatnonzero((actual == "Keep") & (predicted == "Keep"))
    correct_discard = np.flatnonzero((actual == "Discard") & (predicted == "Discard"))
    misclassified = np.flatnonzero(actual != predicted)

    if correct_keep.size:
        selections["keep"] = int(correct_keep[0])
    if correct_discard.size:
        selections["discard"] = int(correct_discard[0])
    if misclassified.size:
        selections["misclassified"] = int(misclassified[0])
    return selections


def calculate_fidelity(shap_values, base_value, predicted_probabilities):
    reconstructed = base_value + np.sum(shap_values, axis=1)
    absolute_error = np.abs(predicted_probabilities - reconstructed)
    return pd.DataFrame(
        {
            "model_probability_keep": predicted_probabilities,
            "shap_reconstructed_probability": reconstructed,
            "absolute_fidelity_error": absolute_error,
        }
    )


def calculate_stability(prediction_function, X_encoded, selected_indices):
    rows = []
    all_similarities = []

    for sample_index in selected_indices:
        repeated_vectors = []
        for run in range(STABILITY_RUNS):
            background = select_background(
                X_encoded,
                size=BACKGROUND_SIZE,
                random_state=RANDOM_STATE + run,
            )
            values, _ = compute_shap_values(
                prediction_function,
                background,
                X_encoded[sample_index : sample_index + 1],
                nsamples=SHAP_NSAMPLES,
            )
            repeated_vectors.append(values[0])

        repeated_matrix = np.asarray(repeated_vectors)
        similarity_matrix = cosine_similarity(repeated_matrix)
        upper = similarity_matrix[np.triu_indices(STABILITY_RUNS, k=1)]
        mean_similarity = float(np.mean(upper))
        all_similarities.extend(upper.tolist())
        rows.append(
            {
                "sample_index": sample_index,
                "mean_cosine_similarity": mean_similarity,
            }
        )

    overall = float(np.mean(all_similarities)) if all_similarities else np.nan
    stability_std = (
        float(np.std(all_similarities, ddof=1))
        if len(all_similarities) > 1
        else np.nan
    )
    return overall, stability_std, pd.DataFrame(rows)


def calculate_sparsity(shap_values, threshold):
    per_sample = np.mean(np.abs(shap_values) <= threshold, axis=1)
    return float(np.mean(per_sample)), per_sample


def main() -> None:
    (
        df,
        _,
        X_encoded,
        model,
        _,
        label_encoder,
        keep_class_index,
    ) = load_artifacts()

    prediction_function = make_keep_probability_function(model, keep_class_index)
    background = select_background(X_encoded, BACKGROUND_SIZE, RANDOM_STATE)

    print("=" * 70)
    print("SHAP Analysis — Categorical Naive Bayes")
    print("=" * 70)
    print(f"Samples explained: {len(X_encoded)}")
    print(f"Background size: {len(background)}")
    print("Explained output: P(Keep)")

    shap_values, base_value = compute_shap_values(
        prediction_function,
        background,
        X_encoded,
        nsamples=SHAP_NSAMPLES,
    )
    predicted_probabilities = prediction_function(X_encoded)

    values_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{name}" for name in FEATURE_COLUMNS],
    )
    values_df.insert(
        0,
        "fringe_id",
        df["fringe_id"].to_numpy() if "fringe_id" in df.columns else np.arange(1, len(df) + 1),
    )
    values_df["base_value_keep"] = base_value
    values_df["model_probability_keep"] = predicted_probabilities
    values_df.to_csv("shap_values_keep_class.csv", index=False)

    importance_df = save_global_importance(shap_values)
    save_summary_plot(shap_values, X_encoded)

    selections = select_representative_indices(df, model, X_encoded, label_encoder)
    output_names = {
        "keep": "shap_local_keep.png",
        "discard": "shap_local_discard.png",
        "misclassified": "shap_local_misclassified.png",
    }

    for name, index in selections.items():
        predicted_label = label_encoder.inverse_transform(model.predict(X_encoded[index:index+1]))[0]
        actual_label = df.iloc[index]["label"]
        fringe_id = int(df.iloc[index]["fringe_id"]) if "fringe_id" in df.columns else index + 1
        save_local_explanation(
            output_names[name],
            f"SHAP Local Explanation — Fringe {fringe_id}\nActual: {actual_label}, Predicted: {predicted_label}",
            df.iloc[index],
            shap_values[index],
            base_value,
            predicted_probabilities[index],
        )

    fidelity_df = calculate_fidelity(shap_values, base_value, predicted_probabilities)
    mean_fidelity_error = float(fidelity_df["absolute_fidelity_error"].mean())
    max_fidelity_error = float(fidelity_df["absolute_fidelity_error"].max())

    stability_indices = list(selections.values())
    overall_stability, stability_std, stability_df = calculate_stability(
        prediction_function,
        X_encoded,
        stability_indices,
    )

    overall_sparsity, per_sample_sparsity = calculate_sparsity(
        shap_values,
        SPARSITY_THRESHOLD,
    )

    fidelity_df.insert(
        0,
        "fringe_id",
        df["fringe_id"].to_numpy() if "fringe_id" in df.columns else np.arange(1, len(df) + 1),
    )
    fidelity_df["sparsity"] = per_sample_sparsity
    fidelity_df.to_csv("shap_explanation_details.csv", index=False)
    stability_df.to_csv("shap_stability_details.csv", index=False)

    pd.DataFrame(
        [
            {
                "method": "SHAP KernelExplainer",
                "explained_class": KEEP_CLASS_NAME,
                "mean_fidelity_error": mean_fidelity_error,
                "max_fidelity_error": max_fidelity_error,
                "mean_stability_cosine_similarity": overall_stability,
                "stability_cosine_similarity_std": stability_std,
                "mean_sparsity": overall_sparsity,
                "sparsity_threshold": SPARSITY_THRESHOLD,
                "samples_explained": len(X_encoded),
                "stability_samples": len(stability_indices),
                "stability_runs": STABILITY_RUNS,
                "kernel_nsamples": SHAP_NSAMPLES,
            }
        ]
    ).to_csv("shap_metrics.csv", index=False)

    print("\nGlobal feature importance:")
    print(importance_df.to_string(index=False))
    print("\nExplanation metrics:")
    print(f"Mean fidelity error: {mean_fidelity_error:.8f}")
    print(f"Maximum fidelity error: {max_fidelity_error:.8f}")
    print(f"Mean stability cosine similarity: {overall_stability:.8f}")
    print(f"Stability cosine similarity standard deviation: {stability_std:.8f}")
    print(f"Mean sparsity: {overall_sparsity:.8f}")
    print(f"SHAP base value P(Keep): {base_value:.8f}")

    print("\nSaved files:")
    for name in [
        "shap_global_importance.csv",
        "shap_values_keep_class.csv",
        "shap_metrics.csv",
        "shap_explanation_details.csv",
        "shap_stability_details.csv",
        "shap_global_bar.png",
        "shap_summary.png",
        "shap_local_keep.png",
        "shap_local_discard.png",
        "shap_local_misclassified.png",
    ]:
        if Path(name).exists():
            print(name)


if __name__ == "__main__":
    main()