from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure

BW_FILE = Path("BW_final.npy")
FEATURE_FILE = Path("freeman_features.csv")
OUTPUT_FILE = Path("labeled_fringes.csv")
CROP_PADDING = 15


def load_inputs() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if not BW_FILE.exists():
        raise FileNotFoundError(
            f"{BW_FILE} was not found. Run the Freeman feature script first."
        )
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"{FEATURE_FILE} was not found. Run the Freeman feature script first."
        )

    bw = np.load(BW_FILE).astype(bool)
    features = pd.read_csv(FEATURE_FILE)

    required = {
        "fringe_id",
        "chain_length",
        "chain_length_bin",
        "direction_change_ratio",
        "direction_change_bin",
        "mean_direction_change",
        "fuzziness_bin",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(
            "The feature CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    labeled_image = measure.label(bw, connectivity=2, background=0)
    component_count = int(labeled_image.max())
    if component_count != len(features):
        raise ValueError(
            "Connected-component count does not match feature rows: "
            f"{component_count} components versus {len(features)} rows."
        )

    return bw, features, labeled_image


def load_existing_labels(features: pd.DataFrame) -> pd.DataFrame:
    """
    Load previous labeling progress or create an empty labeling table.
    """
    if OUTPUT_FILE.exists():
        labeled = pd.read_csv(OUTPUT_FILE)

        existing_labels = labeled[["fringe_id", "label"]].copy()

        result = features.merge(
            existing_labels,
            on="fringe_id",
            how="left",
        )
    else:
        result = features.copy()
        result["label"] = pd.Series(
            [None] * len(result),
            dtype="object",
        )

    # Ensure text labels such as Keep and Discard can be stored.
    result["label"] = result["label"].astype("object")

    return result


def get_component_crop(
    labeled_image: np.ndarray,
    component_label: int,
    padding: int = CROP_PADDING,
) -> np.ndarray:
    coordinates = np.argwhere(labeled_image == component_label)
    if coordinates.size == 0:
        raise ValueError(f"Component {component_label} contains no pixels.")

    min_row, min_col = coordinates.min(axis=0)
    max_row, max_col = coordinates.max(axis=0)

    min_row = max(int(min_row) - padding, 0)
    min_col = max(int(min_col) - padding, 0)
    max_row = min(int(max_row) + padding + 1, labeled_image.shape[0])
    max_col = min(int(max_col) + padding + 1, labeled_image.shape[1])

    return (
        labeled_image[min_row:max_row, min_col:max_col] == component_label
    )


def save_progress(data: pd.DataFrame) -> None:
    data.to_csv(OUTPUT_FILE, index=False)


def print_progress(data: pd.DataFrame) -> None:
    counts = data["label"].value_counts(dropna=False)
    keep = int(counts.get("Keep", 0))
    discard = int(counts.get("Discard", 0))
    skip = int(counts.get("Skip", 0))
    unlabeled = int(data["label"].isna().sum())
    print(
        f"Progress — Keep: {keep}, Discard: {discard}, "
        f"Skipped: {skip}, Unlabeled: {unlabeled}"
    )


def display_fringe(
    row: pd.Series,
    component_crop: np.ndarray,
    current_position: int,
    total: int,
) -> str:
    decision = {"value": None}

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(component_crop, cmap="gray", interpolation="nearest")
    ax.set_title(
        f"Fringe {int(row['fringe_id'])} ({current_position}/{total})\n"
        f"Length: {int(row['chain_length'])} [{row['chain_length_bin']}]\n"
        f"Direction-change ratio: {row['direction_change_ratio']:.4f} "
        f"[{row['direction_change_bin']}]\n"
        f"Mean direction change / fuzziness: "
        f"{row['mean_direction_change']:.4f} [{row['fuzziness_bin']}]\n\n"
        "K = Keep    D = Discard    S = Skip    Q = Quit"
    )
    ax.axis("off")
    fig.tight_layout()

    def on_key(event):
        key = (event.key or "").lower()
        mapping = {
            "k": "Keep",
            "d": "Discard",
            "s": "Skip",
            "q": "Quit",
        }
        if key in mapping:
            decision["value"] = mapping[key]
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return decision["value"] or "Quit"


def run_labeling() -> None:
    _, features, labeled_image = load_inputs()
    data = load_existing_labels(features)

    print("\nInteractive FCC fringe labeling")
    print("K = Keep | D = Discard | S = Skip | Q = Quit")
    print_progress(data)

    pending = data.index[~data["label"].isin(["Keep", "Discard"])].tolist()
    if not pending:
        print("All fringes already have Keep/Discard labels.")
        return

    total = len(pending)
    for position, idx in enumerate(pending, start=1):
        row = data.loc[idx]
        fringe_id = int(row["fringe_id"])
        crop = get_component_crop(labeled_image, fringe_id)
        decision = display_fringe(row, crop, position, total)

        if decision == "Quit":
            save_progress(data)
            print("\nLabeling stopped. Progress was saved.")
            print_progress(data)
            return

        data.at[idx, "label"] = decision
        save_progress(data)
        print(f"Fringe {fringe_id}: {decision}")

    print("\nLabeling completed.")
    print_progress(data)
    print(f"Saved final dataset to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    run_labeling()