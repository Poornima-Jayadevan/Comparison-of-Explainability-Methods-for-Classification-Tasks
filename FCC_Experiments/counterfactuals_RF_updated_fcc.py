from __future__ import annotations

from itertools import combinations
from pathlib import Path

import dice_ml
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path("labeled_fringes.csv")
MODEL_FILE = Path("final_random_forest_model.joblib")
FEATURE_ENCODER_FILE = Path("final_rf_feature_encoder.joblib")
LABEL_ENCODER_FILE = Path("final_rf_label_encoder.joblib")

FEATURE_COLUMNS = [
    "chain_length_bin",
    "direction_change_bin",
    "fuzziness_bin",
]

TARGET_COLUMN = "label"
TOTAL_COUNTERFACTUALS = 3
RANDOM_SEED = 42


class RandomForestCategoricalWrapper:
    """
    Wrap the saved feature encoder and Random Forest.
    """

    def __init__(
        self,
        model,
        feature_encoder,
        label_encoder,
    ):
        self.model = model
        self.feature_encoder = feature_encoder
        self.label_encoder = label_encoder

        # DiCE works more reliably with numeric class IDs.
        self.classes_ = np.arange(
            len(label_encoder.classes_)
        )

    def _to_dataframe(
        self,
        samples,
    ) -> pd.DataFrame:
        """Convert incoming samples to a correctly ordered DataFrame."""
        if isinstance(samples, pd.DataFrame):
            return samples[FEATURE_COLUMNS].copy()

        values = np.asarray(
            samples,
            dtype=object,
        )

        if values.ndim == 1:
            values = values.reshape(1, -1)

        return pd.DataFrame(
            values,
            columns=FEATURE_COLUMNS,
        )

    def predict_proba(
        self,
        samples,
    ) -> np.ndarray:
        """Return Random Forest class probabilities."""
        frame = self._to_dataframe(samples)
        encoded = self.feature_encoder.transform(frame)
        return self.model.predict_proba(encoded)

    def predict(
        self,
        samples,
    ) -> np.ndarray:
        """Return numeric class predictions for DiCE."""
        probabilities = self.predict_proba(samples)
        return np.argmax(
            probabilities,
            axis=1,
        )


def load_artifacts():
    """Load dataset, Random Forest, and encoders."""
    required_files = [
        DATA_FILE,
        MODEL_FILE,
        FEATURE_ENCODER_FILE,
        LABEL_ENCODER_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_path} was not found. Keep all required files "
                "in the same folder as this script."
            )

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    feature_encoder = joblib.load(
        FEATURE_ENCODER_FILE
    )
    label_encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )
    missing = required_columns.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()

    wrapper = RandomForestCategoricalWrapper(
        model,
        feature_encoder,
        label_encoder,
    )

    return (
        df,
        wrapper,
        label_encoder,
    )


def class_name(
    label_encoder,
    encoded_class: int,
) -> str:
    """Convert numeric class ID to Keep or Discard."""
    return str(
        label_encoder.inverse_transform(
            np.asarray(
                [int(encoded_class)]
            )
        )[0]
    )


def class_index(
    label_encoder,
    class_label: str,
) -> int:
    """Convert Keep or Discard to numeric class ID."""
    return int(
        label_encoder.transform(
            np.asarray([class_label])
        )[0]
    )


def opposite_class(
    predicted_label: str,
) -> str:
    """Return the opposite binary class."""
    return (
        "Discard"
        if predicted_label == "Keep"
        else "Keep"
    )


def select_representative_indices(
    df: pd.DataFrame,
    wrapper: RandomForestCategoricalWrapper,
    label_encoder,
) -> dict[str, int]:
    """
    Select one correct Keep, one correct Discard, and one misclassified fringe.
    """
    predicted_encoded = wrapper.predict(
        df[FEATURE_COLUMNS]
    )
    predicted_labels = (
        label_encoder.inverse_transform(
            predicted_encoded.astype(int)
        )
    )

    actual_labels = df[
        TARGET_COLUMN
    ].to_numpy()

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
        selections["keep"] = int(
            correct_keep[0]
        )

    if correct_discard.size:
        selections["discard"] = int(
            correct_discard[0]
        )

    if misclassified.size:
        selections["misclassified"] = int(
            misclassified[0]
        )

    return selections


def build_dice_objects(
    df: pd.DataFrame,
    wrapper: RandomForestCategoricalWrapper,
    label_encoder,
):
    """
    Create DiCE data, model, and explainer objects using numeric outcomes.
    """
    dice_data_frame = df[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].copy()

    dice_data_frame[TARGET_COLUMN] = (
        label_encoder.transform(
            dice_data_frame[TARGET_COLUMN]
        ).astype(int)
    )

    data_interface = dice_ml.Data(
        dataframe=dice_data_frame,
        continuous_features=[],
        outcome_name=TARGET_COLUMN,
    )

    model_interface = dice_ml.Model(
        model=wrapper,
        backend="sklearn",
        model_type="classifier",
    )

    explainer = dice_ml.Dice(
        data_interface,
        model_interface,
        method="random",
    )

    return (
        data_interface,
        model_interface,
        explainer,
    )


def extract_counterfactual_dataframe(
    explanation,
) -> pd.DataFrame:
    """Extract generated counterfactuals from DiCE output."""
    if not explanation.cf_examples_list:
        return pd.DataFrame()

    example = explanation.cf_examples_list[0]

    candidates = [
        getattr(
            example,
            "final_cfs_df",
            None,
        ),
        getattr(
            example,
            "final_cfs_df_sparse",
            None,
        ),
    ]

    for candidate in candidates:
        if (
            candidate is not None
            and not candidate.empty
        ):
            return candidate.copy()

    return pd.DataFrame()


def changed_features(
    original: pd.Series,
    counterfactual: pd.Series,
) -> list[str]:
    """Return names of changed original FCC features."""
    return [
        feature
        for feature in FEATURE_COLUMNS
        if (
            original[feature]
            != counterfactual[feature]
        )
    ]


def categorical_hamming_ratio(
    first_row: pd.Series,
    second_row: pd.Series,
) -> float:
    """Return proportion of categorical states that differ."""
    changes = [
        first_row[feature]
        != second_row[feature]
        for feature in FEATURE_COLUMNS
    ]

    return float(np.mean(changes))


def mean_pairwise_diversity(
    counterfactuals: pd.DataFrame,
) -> float:
    """Mean pairwise Hamming distance among counterfactuals."""
    if len(counterfactuals) < 2:
        return 0.0

    distances = []

    for first_index, second_index in combinations(
        range(len(counterfactuals)),
        2,
    ):
        first = counterfactuals.iloc[
            first_index
        ]
        second = counterfactuals.iloc[
            second_index
        ]

        distances.append(
            categorical_hamming_ratio(
                first,
                second,
            )
        )

    return (
        float(np.mean(distances))
        if distances
        else 0.0
    )


def readable_state(
    feature_name: str,
    value: str,
) -> str:
    """Create readable plot labels."""
    readable_names = {
        "chain_length_bin": "Length",
        "direction_change_bin": (
            "Direction change"
        ),
        "fuzziness_bin": "Fuzziness",
    }

    return (
        f"{readable_names[feature_name]} "
        f"= {value}"
    )


def save_counterfactual_plot(
    output_filename: str,
    title: str,
    original: pd.Series,
    counterfactual: pd.Series,
    original_probability_keep: float,
    counterfactual_probability_keep: float,
) -> None:
    """Save a before-and-after plot of original categorical features."""
    original_labels = [
        readable_state(
            feature,
            original[feature],
        )
        for feature in FEATURE_COLUMNS
    ]

    counterfactual_labels = [
        readable_state(
            feature,
            counterfactual[feature],
        )
        for feature in FEATURE_COLUMNS
    ]

    rows = np.arange(
        len(FEATURE_COLUMNS)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        np.zeros(len(rows)),
        rows,
        s=100,
        label="Original",
    )

    ax.scatter(
        np.ones(len(rows)),
        rows,
        s=100,
        label="Counterfactual",
    )

    for row_index, (
        original_text,
        counterfactual_text,
    ) in enumerate(
        zip(
            original_labels,
            counterfactual_labels,
        )
    ):
        ax.text(
            -0.03,
            row_index,
            original_text,
            ha="right",
            va="center",
        )

        ax.text(
            1.03,
            row_index,
            counterfactual_text,
            ha="left",
            va="center",
        )

        if (
            original_text
            != counterfactual_text
        ):
            ax.plot(
                [0, 1],
                [row_index, row_index],
                color="gray",
                linewidth=1.5,
            )

    ax.set_xlim(-0.8, 1.8)

    ax.set_xticks(
        [0, 1],
        ["Original", "Counterfactual"],
    )

    ax.set_yticks([])

    ax.set_title(
        f"{title}\n"
        f"Original P(Keep) = "
        f"{original_probability_keep:.4f}, "
        f"Counterfactual P(Keep) = "
        f"{counterfactual_probability_keep:.4f}"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()



def categorical_change_vector(
    original: pd.Series,
    counterfactual: pd.Series,
) -> np.ndarray:
    """
    Encode a counterfactual explanation as a binary feature-change vector.
    """
    return np.asarray(
        [
            int(original[feature] != counterfactual[feature])
            for feature in FEATURE_COLUMNS
        ],
        dtype=float,
    )


def cosine_similarity_safe(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
) -> float:
    """
    Compute cosine similarity while safely handling zero vectors.
    """
    first_vector = np.asarray(first_vector, dtype=float)
    second_vector = np.asarray(second_vector, dtype=float)

    first_norm = np.linalg.norm(first_vector)
    second_norm = np.linalg.norm(second_vector)

    if first_norm == 0.0 and second_norm == 0.0:
        return 1.0

    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    return float(
        np.dot(first_vector, second_vector)
        / (first_norm * second_norm)
    )


def select_nearest_valid_counterfactual(
    original: pd.Series,
    counterfactuals: pd.DataFrame,
    desired_label: str,
    wrapper,
    label_encoder,
) -> pd.Series | None:
    """
    Select the valid counterfactual with the smallest categorical Hamming distance from the original observation.
    """
    if counterfactuals.empty:
        return None

    counterfactuals = counterfactuals.reset_index(drop=True).copy()

    predicted_classes = wrapper.predict(
        counterfactuals[FEATURE_COLUMNS]
    ).astype(int)

    predicted_labels = label_encoder.inverse_transform(
        predicted_classes
    )

    counterfactuals["verified_prediction"] = predicted_labels

    valid_counterfactuals = counterfactuals[
        counterfactuals["verified_prediction"] == desired_label
    ].copy()

    if valid_counterfactuals.empty:
        return None

    valid_counterfactuals["hamming_distance"] = [
        categorical_hamming_ratio(original, row)
        for _, row in valid_counterfactuals.iterrows()
    ]

    nearest_index = valid_counterfactuals[
        "hamming_distance"
    ].idxmin()

    return valid_counterfactuals.loc[nearest_index]


def compute_dice_counterfactual_stability(
    explainer,
    query_instance: pd.DataFrame,
    original_row: pd.Series,
    desired_class: int,
    desired_label: str,
    wrapper,
    label_encoder,
    n_runs: int = 30,
    total_cfs: int = TOTAL_COUNTERFACTUALS,
    base_seed: int = RANDOM_SEED,
) -> tuple[float, float, float, pd.DataFrame]:
    """
    Evaluate DiCE counterfactual stability across repeated executions.
    """
    change_vectors = []
    run_rows = []

    for run_index in range(n_runs):
        run_seed = base_seed + run_index

        explanation = explainer.generate_counterfactuals(
            query_instance,
            total_CFs=total_cfs,
            desired_class=desired_class,
            features_to_vary="all",
            random_seed=run_seed,
        )

        generated = extract_counterfactual_dataframe(
            explanation
        )

        selected = select_nearest_valid_counterfactual(
            original=original_row,
            counterfactuals=generated,
            desired_label=desired_label,
            wrapper=wrapper,
            label_encoder=label_encoder,
        )

        if selected is None:
            run_rows.append(
                {
                    "run": run_index + 1,
                    "random_seed": run_seed,
                    "successful": False,
                    "change_vector": None,
                    "changed_feature_count": np.nan,
                    "proximity_hamming": np.nan,
                }
            )
            continue

        vector = categorical_change_vector(
            original_row,
            selected,
        )

        change_vectors.append(vector)

        run_rows.append(
            {
                "run": run_index + 1,
                "random_seed": run_seed,
                "successful": True,
                "change_vector": vector.astype(int).tolist(),
                "changed_feature_count": int(vector.sum()),
                "proximity_hamming": categorical_hamming_ratio(
                    original_row,
                    selected,
                ),
            }
        )

    successful_run_ratio = (
        len(change_vectors) / n_runs
        if n_runs > 0
        else 0.0
    )

    if len(change_vectors) < 2:
        stability_mean = np.nan
        stability_std = np.nan
    else:
        similarities = []

        for first_index, second_index in combinations(
            range(len(change_vectors)),
            2,
        ):
            similarities.append(
                cosine_similarity_safe(
                    change_vectors[first_index],
                    change_vectors[second_index],
                )
            )

        stability_mean = float(np.mean(similarities))
        stability_std = float(np.std(similarities))

    return (
        stability_mean,
        stability_std,
        successful_run_ratio,
        pd.DataFrame(run_rows),
    )


def counterfactual_sparsity_score(
    changed_feature_count: int,
) -> float:
    """
    Compute counterfactual sparsity.
    """
    return float(
        1.0
        - (
            changed_feature_count
            / len(FEATURE_COLUMNS)
        )
    )

def main() -> None:
    (
        df,
        wrapper,
        label_encoder,
    ) = load_artifacts()

    _, _, explainer = build_dice_objects(
        df,
        wrapper,
        label_encoder,
    )

    selections = select_representative_indices(
        df,
        wrapper,
        label_encoder,
    )

    output_names = {
        "keep": (
            "rf_counterfactual_local_keep.png"
        ),
        "discard": (
            "rf_counterfactual_local_discard.png"
        ),
        "misclassified": (
            "rf_counterfactual_local_"
            "misclassified.png"
        ),
    }

    explanation_rows = []
    metric_rows = []
    stability_rows = []

    keep_index = class_index(
        label_encoder,
        "Keep",
    )

    print("=" * 72)
    print(
        "Random Forest DiCE "
        "Counterfactual Analysis"
    )
    print("=" * 72)
    print(f"Features: {FEATURE_COLUMNS}")
    print(
        "Counterfactuals requested per "
        f"representative fringe: "
        f"{TOTAL_COUNTERFACTUALS}"
    )

    for case_type, sample_index in (
        selections.items()
    ):
        original_row = df.iloc[
            sample_index
        ]

        query_instance = pd.DataFrame(
            [
                original_row[
                    FEATURE_COLUMNS
                ]
            ]
        )

        predicted_class = int(
            wrapper.predict(
                query_instance
            )[0]
        )

        predicted_label = class_name(
            label_encoder,
            predicted_class,
        )

        desired_label = opposite_class(
            predicted_label
        )

        desired_class = class_index(
            label_encoder,
            desired_label,
        )

        actual_label = str(
            original_row[TARGET_COLUMN]
        )

        original_probabilities = (
            wrapper.predict_proba(
                query_instance
            )[0]
        )

        original_probability_keep = float(
            original_probabilities[
                keep_index
            ]
        )

        explanation = (
            explainer.generate_counterfactuals(
                query_instance,
                total_CFs=(
                    TOTAL_COUNTERFACTUALS
                ),
                desired_class=desired_class,
                features_to_vary="all",
                random_seed=RANDOM_SEED,
            )
        )

        counterfactuals = (
            extract_counterfactual_dataframe(
                explanation
            )
        )

        fringe_id = (
            int(original_row["fringe_id"])
            if "fringe_id" in df.columns
            else sample_index + 1
        )

        if counterfactuals.empty:
            metric_rows.append(
                {
                    "case_type": case_type,
                    "fringe_id": fringe_id,
                    "actual_label": (
                        actual_label
                    ),
                    "original_prediction": (
                        predicted_label
                    ),
                    "desired_class": (
                        desired_label
                    ),
                    (
                        "counterfactuals_"
                        "requested"
                    ): TOTAL_COUNTERFACTUALS,
                    (
                        "counterfactuals_"
                        "generated"
                    ): 0,
                    "validity": 0.0,
                    (
                        "mean_changed_"
                        "feature_count"
                    ): np.nan,
                    (
                        "mean_changed_"
                        "feature_ratio"
                    ): np.nan,
                    "mean_sparsity_score": np.nan,
                    (
                        "mean_proximity_"
                        "hamming"
                    ): np.nan,
                    (
                        "mean_pairwise_"
                        "diversity"
                    ): np.nan,
                }
            )
            continue

        counterfactuals = (
            counterfactuals.reset_index(
                drop=True
            )
        )

        if (
            TARGET_COLUMN
            not in counterfactuals.columns
        ):
            counterfactuals[
                TARGET_COLUMN
            ] = wrapper.predict(
                counterfactuals[
                    FEATURE_COLUMNS
                ]
            ).astype(int)

        predicted_cf_classes = (
            wrapper.predict(
                counterfactuals[
                    FEATURE_COLUMNS
                ]
            ).astype(int)
        )

        predicted_cf_labels = (
            label_encoder.inverse_transform(
                predicted_cf_classes
            )
        )

        counterfactuals[
            "verified_prediction"
        ] = predicted_cf_labels

        cf_probabilities = (
            wrapper.predict_proba(
                counterfactuals[
                    FEATURE_COLUMNS
                ]
            )
        )

        counterfactuals[
            "probability_keep"
        ] = cf_probabilities[
            :,
            keep_index,
        ]

        valid_mask = (
            counterfactuals[
                "verified_prediction"
            ]
            == desired_label
        )

        validity = float(
            np.mean(valid_mask)
        )

        changed_counts = []
        changed_ratios = []
        sparsity_scores = []
        proximity_values = []

        for (
            counterfactual_index,
            counterfactual_row,
        ) in counterfactuals.iterrows():
            changed = changed_features(
                original_row,
                counterfactual_row,
            )

            changed_count = len(changed)

            changed_ratio = (
                changed_count
                / len(FEATURE_COLUMNS)
            )

            sparsity_score = counterfactual_sparsity_score(
                changed_count
            )

            proximity = (
                categorical_hamming_ratio(
                    original_row,
                    counterfactual_row,
                )
            )

            changed_counts.append(
                changed_count
            )
            changed_ratios.append(
                changed_ratio
            )
            sparsity_scores.append(
                sparsity_score
            )
            proximity_values.append(
                proximity
            )

            explanation_row = {
                "case_type": case_type,
                "fringe_id": fringe_id,
                "counterfactual_id": (
                    counterfactual_index + 1
                ),
                "actual_label": (
                    actual_label
                ),
                "original_prediction": (
                    predicted_label
                ),
                "desired_class": (
                    desired_label
                ),
                (
                    "counterfactual_"
                    "prediction"
                ): str(
                    counterfactual_row[
                        "verified_prediction"
                    ]
                ),
                (
                    "valid_"
                    "counterfactual"
                ): bool(
                    counterfactual_row[
                        "verified_prediction"
                    ]
                    == desired_label
                ),
                (
                    "original_probability_"
                    "keep"
                ): original_probability_keep,
                (
                    "counterfactual_"
                    "probability_keep"
                ): float(
                    counterfactual_row[
                        "probability_keep"
                    ]
                ),
                (
                    "changed_feature_"
                    "count"
                ): changed_count,
                (
                    "changed_feature_"
                    "ratio"
                ): changed_ratio,
                "sparsity_score": sparsity_score,
                (
                    "proximity_hamming"
                ): proximity,
                "changed_features": (
                    ", ".join(changed)
                ),
            }

            for feature in FEATURE_COLUMNS:
                explanation_row[
                    f"original_{feature}"
                ] = original_row[
                    feature
                ]

                explanation_row[
                    (
                        f"counterfactual_"
                        f"{feature}"
                    )
                ] = counterfactual_row[
                    feature
                ]

            explanation_rows.append(
                explanation_row
            )

        diversity = (
            mean_pairwise_diversity(
                counterfactuals
            )
        )

        metric_rows.append(
            {
                "case_type": case_type,
                "fringe_id": fringe_id,
                "actual_label": (
                    actual_label
                ),
                "original_prediction": (
                    predicted_label
                ),
                "desired_class": (
                    desired_label
                ),
                (
                    "counterfactuals_"
                    "requested"
                ): TOTAL_COUNTERFACTUALS,
                (
                    "counterfactuals_"
                    "generated"
                ): len(counterfactuals),
                "validity": validity,
                (
                    "mean_changed_"
                    "feature_count"
                ): float(
                    np.mean(
                        changed_counts
                    )
                ),
                (
                    "mean_changed_"
                    "feature_ratio"
                ): float(
                    np.mean(
                        changed_ratios
                    )
                ),
                "mean_sparsity_score": float(
                    np.mean(
                        sparsity_scores
                    )
                ),
                (
                    "mean_proximity_"
                    "hamming"
                ): float(
                    np.mean(
                        proximity_values
                    )
                ),
                (
                    "mean_pairwise_"
                    "diversity"
                ): diversity,
            }
        )

        stability_mean, stability_std, successful_run_ratio, stability_run_df = (
            compute_dice_counterfactual_stability(
                explainer=explainer,
                query_instance=query_instance,
                original_row=original_row,
                desired_class=desired_class,
                desired_label=desired_label,
                wrapper=wrapper,
                label_encoder=label_encoder,
                n_runs=30,
                total_cfs=TOTAL_COUNTERFACTUALS,
                base_seed=RANDOM_SEED,
            )
        )

        stability_rows.append(
            {
                "case_type": case_type,
                "fringe_id": fringe_id,
                "model": wrapper.__class__.__name__,
                "stability_mean_cosine_similarity": stability_mean,
                "stability_standard_deviation": stability_std,
                "successful_run_ratio": successful_run_ratio,
                "successful_runs": int(
                    stability_run_df["successful"].sum()
                ),
                "total_runs": len(stability_run_df),
            }
        )

        stability_run_df.insert(
            0,
            "case_type",
            case_type,
        )
        stability_run_df.insert(
            1,
            "fringe_id",
            fringe_id,
        )

        stability_run_df.to_csv(
            f"{case_type}_counterfactual_stability_runs.csv",
            index=False,
        )

        valid_indices = np.flatnonzero(
            valid_mask.to_numpy()
        )

        selected_counterfactual_index = (
            int(valid_indices[0])
            if valid_indices.size
            else 0
        )

        selected_counterfactual = (
            counterfactuals.iloc[
                selected_counterfactual_index
            ]
        )

        title = (
            f"Random Forest DiCE "
            f"Counterfactual — "
            f"Fringe {fringe_id}\n"
            f"Actual: {actual_label}, "
            f"Original prediction: "
            f"{predicted_label}, "
            f"Desired: {desired_label}"
        )

        save_counterfactual_plot(
            output_filename=(
                output_names[case_type]
            ),
            title=title,
            original=original_row,
            counterfactual=(
                selected_counterfactual
            ),
            original_probability_keep=(
                original_probability_keep
            ),
            counterfactual_probability_keep=(
                float(
                    selected_counterfactual[
                        "probability_keep"
                    ]
                )
            ),
        )

    explanation_df = pd.DataFrame(
        explanation_rows
    )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    stability_df = pd.DataFrame(stability_rows)

    explanation_df.to_csv(
        "rf_counterfactual_explanations.csv",
        index=False,
    )

    metrics_df.to_csv(
        "rf_counterfactual_metrics.csv",
        index=False,
    )

    stability_df.to_csv(
        "rf_counterfactual_stability.csv",
        index=False,
    )

    print("\nCounterfactual stability:")
    if not stability_df.empty:
        print(stability_df.to_string(index=False))

    print("\nCounterfactual metrics:")

    if not metrics_df.empty:
        print(
            metrics_df.to_string(
                index=False
            )
        )

        print("\nOverall averages:")

        metric_names = [
            "validity",
            "mean_changed_feature_count",
            "mean_changed_feature_ratio",
            "mean_sparsity_score",
            "mean_proximity_hamming",
            "mean_pairwise_diversity",
        ]

        for metric_name in metric_names:
            print(
                f"{metric_name}: "
                f"{metrics_df[metric_name].mean():.6f}"
            )


    evaluation_df = metrics_df.merge(
        stability_df[
            [
                "case_type",
                "fringe_id",
                "stability_mean_cosine_similarity",
                "stability_standard_deviation",
                "successful_run_ratio",
            ]
        ],
        on=["case_type", "fringe_id"],
        how="left",
    )

    evaluation_columns = [
        "case_type",
        "fringe_id",
        "actual_label",
        "original_prediction",
        "desired_class",
        "validity",
        "mean_sparsity_score",
        "stability_mean_cosine_similarity",
        "stability_standard_deviation",
        "successful_run_ratio",
        "mean_changed_feature_count",
        "mean_changed_feature_ratio",
        "mean_proximity_hamming",
        "mean_pairwise_diversity",
    ]

    evaluation_df = evaluation_df[
        [
            column
            for column in evaluation_columns
            if column in evaluation_df.columns
        ]
    ]

    evaluation_filename = "rf_counterfactual_evaluation_metrics.csv"
    evaluation_df.to_csv(
        evaluation_filename,
        index=False,
    )

    print("\nCombined counterfactual evaluation:")
    if not evaluation_df.empty:
        print(
            evaluation_df.to_string(
                index=False
            )
        )

    print("\nSaved files:")
    print("rf_counterfactual_evaluation_metrics.csv")
    print(
        "rf_counterfactual_explanations.csv"
    )
    print(
        "rf_counterfactual_metrics.csv"
    )

    for output_name in (
        output_names.values()
    ):
        if Path(output_name).exists():
            print(output_name)


if __name__ == "__main__":
    main()