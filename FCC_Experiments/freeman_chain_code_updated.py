from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from skimage import feature, filters, measure, morphology


# 1 = right-up, 2 = up, 3 = left-up, 4 = left,
# 5 = left-down, 6 = down, 7 = right-down, 8 = right
MOVE_INDEX = np.array(
    [
        [1, 0, -1, -1, -1, 0, 1, 1],   # x / column movement
        [-1, -1, -1, 0, 1, 1, 1, 0],   # y / row movement
    ],
    dtype=int,
)


def _prev_dir(direction: int) -> int:
    return ((direction - 2) % 8) + 1


def _opp_dir(direction: int) -> int:
    return ((direction + 2) % 8) + 1


def _inside(shape: tuple[int, int], row: int, col: int) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _find_first_white(image: np.ndarray) -> tuple[int, int] | None:
    indices = np.flatnonzero(image.ravel(order="F"))

    if indices.size == 0:
        return None

    row, col = np.unravel_index(indices[0], image.shape, order="F")
    return int(row), int(col)


def shut_off_binary_shape_from_its_contour(
    image: np.ndarray,
    boundary_coordinates: np.ndarray,
) -> np.ndarray:
    
    output = image.astype(np.uint8).copy()
    shape_mask = np.zeros_like(output, dtype=np.uint8)

    if boundary_coordinates.size == 0:
        return output.astype(bool)

    rows = boundary_coordinates[:, 1].astype(int)
    cols = boundary_coordinates[:, 0].astype(int)

    row_min = max(int(rows.min()), 0)
    row_max = min(int(rows.max()), output.shape[0] - 1)

    for row in range(row_min, row_max + 1):
        row_cols = cols[rows == row]

        if row_cols.size == 0:
            continue

        col_min = max(int(row_cols.min()), 0)
        col_max = min(int(row_cols.max()), output.shape[1] - 1)
        shape_mask[row, col_min : col_max + 1] = 1

    output = np.clip(output - shape_mask, 0, 1)
    return output.astype(bool)


def freeman_chain_code(
    image: np.ndarray,
    option_display: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[int]],
    list[np.ndarray],
    bool,
]:
    
    working_image = np.asarray(image, dtype=bool).copy()

    if working_image.ndim != 2:
        raise ValueError(
            f"Freeman chain code requires a 2D binary image; "
            f"received shape {working_image.shape}."
        )

    shape = working_image.shape

    # Detect an inverted image by checking whether most border pixels are white.
    contour_sum = (
        working_image[:, 0].sum()
        + working_image[:, -1].sum()
        + working_image[0, :].sum()
        + working_image[-1, :].sum()
        - working_image[0, 0]
        - working_image[0, -1]
        - working_image[-1, 0]
        - working_image[-1, -1]
    )

    border_count = (2 * shape[0]) + (2 * shape[1]) - 4
    invert_img = bool(contour_sum / border_count > 0.5)

    if invert_img:
        working_image = ~working_image

    bound_img = np.zeros_like(working_image, dtype=bool)
    codes: list[list[int]] = []
    boundary_coordinates: list[np.ndarray] = []
    starting_points: list[list[int]] = []

    first_pixel = _find_first_white(working_image)

    while first_pixel is not None:
        row_start, col_start = first_pixel
        starting_points.append([row_start, col_start])
        bound_img[row_start, col_start] = True

        current_row = row_start
        current_col = col_start

        direction = 8
        search_direction = _opp_dir(direction)

        # Temporarily remove the first pixel so that the algorithm can search for the second contour pixel.
        working_image[row_start, col_start] = False

        second_pixel_found = False

        for _ in range(8):
            search_direction = _prev_dir(search_direction)

            next_row = row_start + MOVE_INDEX[1, search_direction - 1]
            next_col = col_start + MOVE_INDEX[0, search_direction - 1]

            if (
                _inside(shape, next_row, next_col)
                and working_image[next_row, next_col]
            ):
                second_pixel_found = True
                break

        working_image[row_start, col_start] = True

        # Handle isolated white pixels.
        if not second_pixel_found:
            codes.append([])
            isolated_coordinate = np.array(
                [[col_start, row_start]],
                dtype=int,
            )
            boundary_coordinates.append(isolated_coordinate)
            working_image[row_start, col_start] = False
            first_pixel = _find_first_white(working_image)
            continue

        region_code = [search_direction]
        bound_img[next_row, next_col] = True
        direction = search_direction

        current_row = next_row
        current_col = next_col

        # Store the traced coordinates directly.
        traced_coordinates = [[col_start, row_start], [current_col, current_row]]

        # Safety guard to prevent an infinite loop on malformed components.
        max_steps = working_image.size * 10
        steps = 0

        while (
            current_row != row_start or current_col != col_start
        ) and steps < max_steps:
            search_direction = _opp_dir(direction)

            # Temporarily switch off the current pixel during neighbour search.
            working_image[current_row, current_col] = False
            next_pixel_found = False

            for _ in range(8):
                search_direction = _prev_dir(search_direction)

                next_row = (
                    current_row + MOVE_INDEX[1, search_direction - 1]
                )
                next_col = (
                    current_col + MOVE_INDEX[0, search_direction - 1]
                )

                if (
                    _inside(shape, next_row, next_col)
                    and working_image[next_row, next_col]
                ):
                    next_pixel_found = True
                    break

            # Restore the current pixel.
            working_image[current_row, current_col] = True

            if not next_pixel_found:
                # The component is open or broken. Preserve the code traced up to this point and stop this contour.
                break

            region_code.append(search_direction)
            direction = search_direction

            bound_img[current_row, current_col] = True
            current_row = next_row
            current_col = next_col
            traced_coordinates.append([current_col, current_row])

            steps += 1

        region_boundary = np.asarray(traced_coordinates, dtype=int)

        # Remove this region so the next connected white region can be traced.
        working_image = shut_off_binary_shape_from_its_contour(
            working_image,
            region_boundary,
        )

        codes.append(region_code)
        boundary_coordinates.append(region_boundary)
        first_pixel = _find_first_white(working_image)

    X0 = (
        np.asarray(starting_points, dtype=int).T
        if starting_points
        else np.empty((2, 0), dtype=int)
    )

    if invert_img:
        bound_img = ~bound_img

    if option_display:
        plt.figure(figsize=(8, 8))
        plt.imshow(bound_img, cmap="gray")
        plt.title("Extracted Freeman boundaries")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return bound_img, X0, codes, boundary_coordinates, invert_img


def discrete_curvature(
    image: np.ndarray,
) -> tuple[list[list[int]], np.ndarray]:
    
    _, X0, codes, _, _ = freeman_chain_code(
        image,
        option_display=False,
    )
    return codes, X0


def _get_mat_variable(
    mat_data: dict,
    variable: str | None,
) -> tuple[str, np.ndarray]:
    
    available = {
        name: value
        for name, value in mat_data.items()
        if not name.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == 2
    }

    if not available:
        raise ValueError("No 2D array was found in the MAT file.")

    if variable is not None:
        if variable not in available:
            names = ", ".join(sorted(available))
            raise KeyError(
                f"Variable '{variable}' was not found. "
                f"Available variables: {names}"
            )
        return variable, np.asarray(available[variable], dtype=float)

    
    selected_name = sorted(available)[0]
    return selected_name, np.asarray(available[selected_name], dtype=float)


def prepare_interferogram(
    mat_path: str | Path,
    variable: str | None = "I_x",
) -> tuple[np.ndarray, str]:
    
    mat_path = Path(mat_path)

    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file does not exist: {mat_path}")

    mat_data = sio.loadmat(mat_path)
    variable_name, interferogram = _get_mat_variable(mat_data, variable)

    interferogram = np.squeeze(interferogram)

    if interferogram.ndim != 2:
        raise ValueError(
            f"Selected variable '{variable_name}' is not a 2D image."
        )

    interferogram = np.nan_to_num(
        interferogram,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return interferogram, variable_name


def _convert_to_unit_range(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)

    minimum = float(np.min(image))
    maximum = float(np.max(image))

    if minimum >= 0.0 and maximum <= 1.0:
        return image.copy()

    if np.isclose(maximum, minimum):
        return np.zeros_like(image, dtype=float)

    return (image - minimum) / (maximum - minimum)


def create_matlab_style_bw(
    interferogram: np.ndarray,
    *,
    apply_smoothing: bool = True,
    gaussian_sigma: float = 2.0,
    binary_threshold: float = 0.5,
    canny_threshold: float = 0.5,
    canny_sigma: float = 2.0,
    annulus_disk_radius: int = 10,
    minimum_component_size: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    
    original = _convert_to_unit_range(interferogram)

    processed = original.copy()

    if apply_smoothing:
        processed = filters.gaussian(
            processed,
            sigma=gaussian_sigma,
            preserve_range=True,
        )

    binary_image = processed > binary_threshold


    high_threshold = float(canny_threshold)
    low_threshold = 0.4 * high_threshold

    BW = feature.canny(
        binary_image.astype(float),
        sigma=canny_sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        use_quantiles=False,
    )

    
    annulus_mask = np.isclose(original, 0.0)

    if annulus_disk_radius > 0:
        annulus_mask = morphology.dilation(
            annulus_mask,
            footprint=morphology.disk(annulus_disk_radius),
        )

    BW = BW.copy()
    BW[annulus_mask] = False

    
    if minimum_component_size > 0:
        # Using max_size avoids the scikit-image min_size deprecation warning.
        BW = morphology.remove_small_objects(
            BW,
            max_size=minimum_component_size - 1,
        )

    intermediate = {
        "original": original,
        "smoothed": processed,
        "binary": binary_image,
        "annulus_mask": annulus_mask,
        "BW": BW,
    }

    return BW.astype(bool), intermediate


def connected_component_information(
    BW: np.ndarray,
) -> tuple[np.ndarray, list[measure._regionprops.RegionProperties]]:
    
    labeled_image = measure.label(
        BW,
        connectivity=2,  # 8-connectivity for a 2D image
        background=0,
    )
    regions = measure.regionprops(labeled_image)
    return labeled_image, regions


def extract_one_chain_code_per_component(
    BW: np.ndarray,
) -> tuple[list[list[int]], np.ndarray]:
    
    labeled_image, regions = connected_component_information(BW)

    component_codes: list[list[int]] = []
    global_starting_points: list[list[int]] = []

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox

        # Add one-pixel padding so contours touching the crop boundary still have a black background around them.
        pad = 1
        crop_min_row = max(min_row - pad, 0)
        crop_min_col = max(min_col - pad, 0)
        crop_max_row = min(max_row + pad, BW.shape[0])
        crop_max_col = min(max_col + pad, BW.shape[1])

        component_crop = (
            labeled_image[
                crop_min_row:crop_max_row,
                crop_min_col:crop_max_col,
            ]
            == region.label
        )

        _, local_X0, local_codes, _, _ = freeman_chain_code(
            component_crop,
            option_display=False,
        )

        # Retain the longest contour as the main chain code of the component.
        if local_codes:
            lengths = [len(code) for code in local_codes]
            selected_index = int(np.argmax(lengths))
            selected_code = local_codes[selected_index]

            if local_X0.shape[1] > selected_index:
                local_row = int(local_X0[0, selected_index])
                local_col = int(local_X0[1, selected_index])
            else:
                # Fallback to the first component pixel.
                local_pixel = np.argwhere(component_crop)[0]
                local_row = int(local_pixel[0])
                local_col = int(local_pixel[1])
        else:
            selected_code = []
            local_pixel = np.argwhere(component_crop)[0]
            local_row = int(local_pixel[0])
            local_col = int(local_pixel[1])

        global_row = crop_min_row + local_row
        global_col = crop_min_col + local_col

        component_codes.append(selected_code)
        global_starting_points.append([global_row, global_col])

    X0 = (
        np.asarray(global_starting_points, dtype=int).T
        if global_starting_points
        else np.empty((2, 0), dtype=int)
    )

    return component_codes, X0



def reconstruct_path_from_code(
    start_row: int,
    start_col: int,
    code: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    
    rows = [int(start_row)]
    cols = [int(start_col)]

    row = int(start_row)
    col = int(start_col)

    for direction in code:
        col += int(MOVE_INDEX[0, direction - 1])
        row += int(MOVE_INDEX[1, direction - 1])

        rows.append(row)
        cols.append(col)

    return np.asarray(rows), np.asarray(cols)


def display_starting_points(
    BW: np.ndarray,
    X0: np.ndarray,
) -> None:
    """Display all Freeman starting points X0 over the BW image."""
    plt.figure(figsize=(8, 8))
    plt.imshow(BW, cmap="gray")

    if X0.size > 0:
        plt.scatter(
            X0[1, :],
            X0[0, :],
            s=14,
            marker="o",
        )

    plt.title("Starting point X0 of each Freeman chain code")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def display_selected_fringe(
    BW: np.ndarray,
    codes: list[list[int]],
    X0: np.ndarray,
    fringe_number: int = 1,
) -> None:
    
    index = fringe_number - 1

    if index < 0 or index >= len(codes):
        raise IndexError(
            f"fringe_number must be between 1 and {len(codes)}."
        )

    rows, cols = reconstruct_path_from_code(
        int(X0[0, index]),
        int(X0[1, index]),
        codes[index],
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(BW, cmap="gray")
    plt.plot(cols, rows, linewidth=1.5)
    plt.scatter(
        [cols[0]],
        [rows[0]],
        s=55,
        marker="o",
        label="X0",
    )
    plt.title(
        f"Reconstructed Freeman path — Fringe {fringe_number}\n"
        f"Chain length: {len(codes[index])}"
    )
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def display_chain_length_distribution(
    codes: list[list[int]],
) -> None:
    """Display a histogram of Freeman chain-code lengths."""
    lengths = [len(code) for code in codes]

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=20)
    plt.xlabel("Chain length")
    plt.ylabel("Number of connected regions")
    plt.title("Distribution of Freeman chain-code lengths")
    plt.tight_layout()
    plt.show()


def display_chain_code_signal(
    codes: list[list[int]],
    fringe_number: int = 1,
    maximum_steps: int | None = 500,
) -> None:
    
    index = fringe_number - 1

    if index < 0 or index >= len(codes):
        raise IndexError(
            f"fringe_number must be between 1 and {len(codes)}."
        )

    code = codes[index]

    if maximum_steps is not None:
        displayed_code = code[:maximum_steps]
    else:
        displayed_code = code

    steps = np.arange(1, len(displayed_code) + 1)

    plt.figure(figsize=(10, 4))
    plt.step(steps, displayed_code, where="mid")
    plt.yticks(range(1, 9))
    plt.xlabel("Chain-code step")
    plt.ylabel("Direction number")
    plt.title(
        f"Freeman chain code — Fringe {fringe_number}"
        + (
            f" (first {len(displayed_code)} steps)"
            if len(displayed_code) < len(code)
            else ""
        )
    )
    plt.tight_layout()
    plt.show()


def save_visual_results(
    BW: np.ndarray,
    codes: list[list[int]],
    X0: np.ndarray,
    fringe_number: int = 1,
) -> None:
    """Save key result figures as PNG files."""
    # Starting points
    plt.figure(figsize=(8, 8))
    plt.imshow(BW, cmap="gray")
    if X0.size > 0:
        plt.scatter(X0[1, :], X0[0, :], s=14, marker="o")
    plt.title("Starting point X0 of each Freeman chain code")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("freeman_starting_points.png", dpi=300)
    plt.close()

    # Selected fringe
    index = fringe_number - 1
    if 0 <= index < len(codes):
        rows, cols = reconstruct_path_from_code(
            int(X0[0, index]),
            int(X0[1, index]),
            codes[index],
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(BW, cmap="gray")
        plt.plot(cols, rows, linewidth=1.5)
        plt.scatter([cols[0]], [rows[0]], s=55, marker="o")
        plt.title(
            f"Reconstructed Freeman path — Fringe {fringe_number}"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"freeman_fringe_{fringe_number}.png",
            dpi=300,
        )
        plt.close()

    # Chain-length histogram
    plt.figure(figsize=(8, 5))
    plt.hist([len(code) for code in codes], bins=20)
    plt.xlabel("Chain length")
    plt.ylabel("Number of connected regions")
    plt.title("Distribution of Freeman chain-code lengths")
    plt.tight_layout()
    plt.savefig("freeman_chain_length_histogram.png", dpi=300)
    plt.close()



def circular_direction_change(direction_a: int, direction_b: int) -> int:
    """Return the smallest circular difference between two directions (0..4)."""
    raw_difference = abs(int(direction_a) - int(direction_b))
    return min(raw_difference, 8 - raw_difference)


def extract_chain_features(code: list[int], fringe_id: int) -> dict[str, float | int]:
    """Convert one Freeman chain code into a compact feature vector."""
    code_array = np.asarray(code, dtype=int)
    chain_length = int(code_array.size)

    features: dict[str, float | int] = {
        "fringe_id": int(fringe_id),
        "chain_length": chain_length,
    }

    for direction in range(1, 9):
        features[f"dir_{direction}_ratio"] = (
            float(np.mean(code_array == direction)) if chain_length else 0.0
        )

    if chain_length < 2:
        changes = np.array([], dtype=int)
    else:
        changes = np.array([
            circular_direction_change(a, b)
            for a, b in zip(code_array[:-1], code_array[1:])
        ], dtype=int)

    if changes.size == 0:
        features.update({
            "direction_change_ratio": 0.0,
            "mean_direction_change": 0.0,
            "max_direction_change": 0,
            "large_change_ratio": 0.0,
        })
    else:
        features.update({
            "direction_change_ratio": float(np.mean(changes > 0)),
            "mean_direction_change": float(np.mean(changes)),
            "max_direction_change": int(np.max(changes)),
            "large_change_ratio": float(np.mean(changes >= 3)),
        })

    return features


def assign_chain_length_bin(chain_length: int) -> str:
    """Assign Short, Medium, or Long chain-length state."""
    if chain_length < 40:
        return "Short"
    if chain_length <= 600:
        return "Medium"
    return "Long"


def assign_direction_change_bin(direction_change_ratio: float) -> str:
    """Assign Low, Medium, or High direction-change state."""
    if direction_change_ratio < 0.42:
        return "Low"
    if direction_change_ratio <= 0.55:
        return "Medium"
    return "High"


def assign_fuzziness_bin(mean_direction_change: float) -> str:
    """Assign Low, Medium, or High fuzziness state."""
    if mean_direction_change < 0.47:
        return "Low"
    if mean_direction_change <= 0.65:
        return "Medium"
    return "High"


def add_feature_bins(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical feature states while keeping raw measurements."""
    output = feature_df.copy()
    output["chain_length_bin"] = output["chain_length"].apply(assign_chain_length_bin)
    output["direction_change_bin"] = output["direction_change_ratio"].apply(assign_direction_change_bin)
    output["fuzziness_bin"] = output["mean_direction_change"].apply(assign_fuzziness_bin)
    return output


def build_feature_dataframe(codes: list[list[int]]) -> pd.DataFrame:
    """Create one feature row per fringe and add categorical bins."""
    rows = [
        extract_chain_features(code, fringe_id=index + 1)
        for index, code in enumerate(codes)
    ]
    feature_df = add_feature_bins(pd.DataFrame(rows))

    columns = [
        "fringe_id",
        "chain_length",
        "chain_length_bin",
        "direction_change_ratio",
        "direction_change_bin",
        "mean_direction_change",
        "fuzziness_bin",
        "max_direction_change",
        "large_change_ratio",
        "dir_1_ratio", "dir_2_ratio", "dir_3_ratio", "dir_4_ratio",
        "dir_5_ratio", "dir_6_ratio", "dir_7_ratio", "dir_8_ratio",
    ]
    return feature_df[columns]


def save_feature_dataset(
    codes: list[list[int]],
    output_csv: str | Path = "freeman_features.csv",
) -> pd.DataFrame:
    """Build and save the Freeman feature dataset."""
    feature_df = build_feature_dataframe(codes)
    feature_df.to_csv(output_csv, index=False)
    return feature_df


def display_feature_summary(feature_df: pd.DataFrame) -> None:
    """Print a preview and summary of raw and binned features."""
    print("\\nFeature dataset preview:")
    print(feature_df.head(10).to_string(index=False))

    summary_columns = [
        "chain_length",
        "direction_change_ratio",
        "mean_direction_change",
        "max_direction_change",
        "large_change_ratio",
    ]
    print("\\nFeature summary:")
    print(feature_df[summary_columns].describe().to_string())

    print("\\nChain-length bin counts:")
    print(feature_df["chain_length_bin"].value_counts().to_string())
    print("\\nDirection-change bin counts:")
    print(feature_df["direction_change_bin"].value_counts().to_string())
    print("\\nFuzziness bin counts:")
    print(feature_df["fuzziness_bin"].value_counts().to_string())


def save_chain_codes_to_csv(
    codes: list[list[int]],
    X0: np.ndarray,
    output_csv: str | Path,
) -> None:
    """Save one Freeman chain code per row."""
    rows = []

    for index, code in enumerate(codes):
        start_row = (
            int(X0[0, index])
            if X0.shape[1] > index
            else None
        )
        start_col = (
            int(X0[1, index])
            if X0.shape[1] > index
            else None
        )

        rows.append(
            {
                "fringe_id": index + 1,
                "x0_row_zero_based": start_row,
                "x0_col_zero_based": start_col,
                "chain_length": len(code),
                "freeman_code": " ".join(map(str, code)),
            }
        )

    pd.DataFrame(rows).to_csv(output_csv, index=False)


def display_preprocessing(
    intermediate: dict[str, np.ndarray],
) -> None:
    """Display each preprocessing result in a separate figure."""
    figures = [
        ("original", "Original interferogram"),
        ("smoothed", "Gaussian-smoothed interferogram"),
        ("binary", "Binary image after threshold 0.5"),
        ("annulus_mask", "Dilated annulus mask"),
        ("BW", "Final BW input to Freeman chain code"),
    ]

    for key, title in figures:
        plt.figure(figsize=(8, 8))
        plt.imshow(intermediate[key], cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def main() -> None:
    
    mat_file = Path("2K_Horizontal.mat")

    
    variable_name = "I_x"

    interferogram, selected_variable = prepare_interferogram(
        mat_file,
        variable=variable_name,
    )

    print(f"Loaded MAT variable: {selected_variable}")
    print(f"Interferogram shape: {interferogram.shape}")
    print(
        "Interferogram range after NaN replacement: "
        f"{interferogram.min():.6f} to {interferogram.max():.6f}"
    )

    BW, intermediate = create_matlab_style_bw(
        interferogram,
        apply_smoothing=True,
        gaussian_sigma=2.0,
        binary_threshold=0.5,
        canny_threshold=0.5,
        canny_sigma=2.0,
        annulus_disk_radius=10,
        minimum_component_size=0,
    )

    labeled_image, regions = connected_component_information(BW)

    print(
        "Number of 8-connected regions before Freeman tracing: "
        f"{len(regions)}"
    )

    # process each connected region separately.
    freeman_codes, X0 = extract_one_chain_code_per_component(BW)

    print(
        "Number of chain codes returned: "
        f"{len(freeman_codes)}"
    )

    for index, code in enumerate(freeman_codes[:10]):
        if X0.shape[1] <= index:
            break

        print(
            f"Fringe {index + 1}: "
            f"X0(row,col)=({X0[0, index]}, {X0[1, index]}), "
            f"length={len(code)}, "
            f"first 40={code[:40]}"
        )

    output_csv = Path("freeman_chain_codes.csv")
    save_chain_codes_to_csv(
        freeman_codes,
        X0,
        output_csv,
    )
    print(f"Saved chain codes to: {output_csv.resolve()}")

    feature_csv = Path("freeman_features.csv")
    feature_df = save_feature_dataset(
        freeman_codes,
        feature_csv,
    )
    print(f"Saved feature dataset to: {feature_csv.resolve()}")
    display_feature_summary(feature_df)

    # Save BW 
    np.save("BW_final.npy", BW)
    plt.imsave("BW_final.png", BW, cmap="gray")
    print("Saved final BW as BW_final.npy and BW_final.png")

    # Select which fringe to inspect visually.
    selected_fringe = 3

    # Display preprocessing and Freeman-chain-code results.
    display_preprocessing(intermediate)
    display_starting_points(BW, X0)
    display_selected_fringe(
        BW,
        freeman_codes,
        X0,
        fringe_number=selected_fringe,
    )
    display_chain_length_distribution(freeman_codes)
    display_chain_code_signal(
        freeman_codes,
        fringe_number=selected_fringe,
        maximum_steps=500,
    )

    # Save the main visual results as high-resolution PNG files.
    save_visual_results(
        BW,
        freeman_codes,
        X0,
        fringe_number=selected_fringe,
    )
    print(
        "Saved visual results: freeman_starting_points.png, "
        f"freeman_fringe_{selected_fringe}.png, and "
        "freeman_chain_length_histogram.png"
    )


if __name__ == "__main__":
    main()