
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Fix for older SHAP versions with newer NumPy
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore

# ==========================================================
# save SHAP plots as PNGs + show images side-by-side
# ==========================================================
def save_shap_summary(shap_2d, X, feature_names, out_png, title=None):
    shap.summary_plot(shap_2d, X, feature_names=feature_names, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def save_shap_dependence(shap_2d, X, feature_names, out_png, title=None, highlight_idx=None):
    shap.dependence_plot(
        "x1",
        shap_2d,
        X,
        feature_names=feature_names,
        interaction_index="x2",
        show=False
    )
    if highlight_idx is not None:
        plt.scatter(
            [X[highlight_idx, 0]],
            [shap_2d[highlight_idx, 0]],
            s=220, marker="X",
            color="red", edgecolor="black", zorder=10
        )
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def save_shap_waterfall(shap_2d, X, feature_names, base_1, idx, out_png, title=None):
    exp = shap.Explanation(
        values=shap_2d[idx],
        base_values=base_1,
        data=X[idx],
        feature_names=feature_names
    )
    shap.plots.waterfall(exp, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def show_images_row(pngs, titles, big_title, out_png, figsize=(24, 6)):
    fig, axes = plt.subplots(1, len(pngs), figsize=figsize)
    if len(pngs) == 1:
        axes = [axes]
    for ax, p, t in zip(axes, pngs, titles):
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(t, fontsize=12)
    fig.suptitle(big_title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

def nearest_train_index(Xtr, pt):
    d = np.linalg.norm(Xtr - pt.reshape(1, -1), axis=1)
    return int(np.argmin(d))

# ==========================================================
# Decision regions + SHAP arrows (1x3)
# ==========================================================
def plot_decision_regions_with_shap_arrows_1x3(
    rf, df, feature_names, shap_2d,
    instances,   
    out_png="rf_combined_shap_arrows_1x3.png"
):
    x_min, x_max = df["x1"].min()-1, df["x1"].max()+1
    y_min, y_max = df["x2"].min()-1, df["x2"].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    x_min, x_max = df["x1"].min() - 1, df["x1"].max() + 1
    y_min, y_max = df["x2"].min() - 1, df["x2"].max() + 1

    max_abs = float(np.max(np.abs(shap_2d)))
    if max_abs < 1e-12:
        max_abs = 1.0

    def arrow_len(v, max_abs, scale=2.0):
        if max_abs < 1e-12:
            return 0.0
        return scale * (float(v) / max_abs)

    plt.figure(figsize=(18, 6))

    for i, inst in enumerate(instances, 1):
        name = inst["name"]
        pt = inst["point"]
        tr_idx = inst["train_idx"]
        color = inst["color"]

        svals = shap_2d[tr_idx]
        dx = arrow_len(svals[0], max_abs=max_abs, scale=2.0)
        dy = arrow_len(svals[1], max_abs=max_abs, scale=2.0)

        plt.subplot(1, 3, i)

        # Background only for the central square
        plt.contourf(xx, yy, Z, alpha=0.3)

        # All points (including outliers)
        plt.scatter(df["x1"], df["x2"], c=df["label"], s=12, alpha=0.6)

        # True boundary
        plt.axline((0, 0), slope=1, linestyle="--", color="black")

        # Explained instance
        plt.scatter(pt[0], pt[1], color="red", marker="X", s=160,
                    edgecolor="black", label="Explained Instance", zorder=5)

        # SHAP arrows
        plt.arrow(pt[0], pt[1], dx, 0, head_width=0.50, length_includes_head=True,
                  color=color, zorder=6)
        plt.arrow(pt[0], pt[1], 0, dy, head_width=0.50, length_includes_head=True,
                  color=color, zorder=6)

        # Wide axes so outliers show (
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.title(name)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()

    plt.suptitle("RF (Combined) — Decision Regions + SHAP Arrows (Clean vs Boundary vs Outlier)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

# =============================
# Load model + data
# =============================
rf = joblib.load("rf_combined_model.pkl")
X_train = joblib.load("X_train_combined.pkl")
feature_names = joblib.load("feature_names_combined.pkl")
df = pd.read_csv("combined.csv")

X_train = np.asarray(X_train, dtype=np.float64)

# =============================
# Pick instances (Clean, Boundary, Outlier) from df
# =============================
df["distance"] = np.abs(df["x1"] - df["x2"]) / np.sqrt(2)
region_lower = df["region"].astype(str).str.lower()

clean_df = df[region_lower.str.contains("clean")]
boundary_df = df[region_lower.str.contains("bound")]
outlier_df = df[region_lower.str.contains("out")]

if len(clean_df) == 0:
    raise ValueError("No clean rows found in df['region'].")
if len(boundary_df) == 0:
    raise ValueError("No boundary rows found in df['region'].")

# Clean: farthest from boundary (within clean region)
clean_row = clean_df.sort_values("distance", ascending=False).iloc[0]
# Boundary: closest to boundary (within boundary region)
boundary_row = boundary_df.sort_values("distance", ascending=True).iloc[0]

# Outlier: prefer region='outlier', else fallback farthest-from-mean
if len(outlier_df) > 0:
    mean_point = df[feature_names].mean().values
    d = np.linalg.norm(outlier_df[feature_names].values - mean_point.reshape(1, -1), axis=1)
    outlier_row = outlier_df.iloc[int(np.argmax(d))]
else:
    mean_point = df[feature_names].mean().values
    outlier_idx = np.linalg.norm(df[feature_names].values - mean_point.reshape(1, -1), axis=1).argmax()
    outlier_row = df.iloc[int(outlier_idx)]

clean_point = clean_row[feature_names].values.astype(np.float64)
boundary_point = boundary_row[feature_names].values.astype(np.float64)
outlier_point = outlier_row[feature_names].values.astype(np.float64)

clean_tr_idx = nearest_train_index(X_train, clean_point)
boundary_tr_idx = nearest_train_index(X_train, boundary_point)
outlier_tr_idx = nearest_train_index(X_train, outlier_point)

print("Clean:", clean_point, "train idx:", clean_tr_idx, "label:", int(clean_row["label"]))
print("Boundary:", boundary_point, "train idx:", boundary_tr_idx, "label:", int(boundary_row["label"]))
print("Outlier:", outlier_point, "train idx:", outlier_tr_idx, "label:", int(outlier_row["label"]))

# =============================
# Compute SHAP
# =============================
explainer = shap.TreeExplainer(rf)
sv = explainer(X_train)

vals = sv.values
base = sv.base_values

if vals.ndim == 2:
    shap_2d = vals
    base_1 = float(base) if np.ndim(base) == 0 else float(np.mean(base))
elif vals.ndim == 3 and vals.shape[2] >= 2:
    shap_2d = vals[:, :, 1]
    if np.ndim(base) == 2:
        base_1 = float(np.mean(base[:, 1]))
    elif np.ndim(base) == 1 and base.shape[0] >= 2:
        base_1 = float(base[1])
    else:
        base_1 = float(np.mean(base))
else:
    raise ValueError(f"Unexpected SHAP shape: {vals.shape}")

# =============================
# SHAP comparison table (3 instances)
# =============================
shap_clean = shap_2d[clean_tr_idx]
shap_boundary = shap_2d[boundary_tr_idx]
shap_outlier = shap_2d[outlier_tr_idx]

comparison_df = pd.DataFrame({
    "Feature": feature_names,
    "Clean-region SHAP": [float(shap_clean[i]) for i in range(len(feature_names))],
    "Tight-boundary SHAP": [float(shap_boundary[i]) for i in range(len(feature_names))],
    "Outlier SHAP": [float(shap_outlier[i]) for i in range(len(feature_names))],
    "Abs Clean": [abs(float(shap_clean[i])) for i in range(len(feature_names))],
    "Abs Boundary": [abs(float(shap_boundary[i])) for i in range(len(feature_names))],
    "Abs Outlier": [abs(float(shap_outlier[i])) for i in range(len(feature_names))],
})

print("\n=== SHAP Comparison Table (Clean vs Boundary vs Outlier) ===")
print(comparison_df)

comparison_df.to_csv("shap_comparison_combined_clean_boundary_outlier.csv", index=False)
print("Saved table: shap_comparison_combined_clean_boundary_outlier.csv")

# ==========================================================
# Decision Regions + SHAP Arrows
# ==========================================================
instances = [
    {"name": "Clean-region", "point": clean_point, "train_idx": clean_tr_idx, "color": "blue"},
    {"name": "Tight-boundary", "point": boundary_point, "train_idx": boundary_tr_idx, "color": "green"},
    {"name": "Outlier", "point": outlier_point, "train_idx": outlier_tr_idx, "color": "purple"},
]

plot_decision_regions_with_shap_arrows_1x3(
    rf=rf,
    df=df,
    feature_names=feature_names,
    shap_2d=shap_2d,
    instances=instances,
    out_png="rf_combined_shap_arrows_1x3.png"
)

# ==========================================================
# SUMMARY [global]
# ==========================================================
sum_clean_png = "tmp_sum_clean.png"
sum_bound_png = "tmp_sum_boundary.png"
sum_out_png = "tmp_sum_outlier.png"

save_shap_summary(shap_2d, X_train, feature_names, sum_clean_png, title="Summary (Clean)")
save_shap_summary(shap_2d, X_train, feature_names, sum_bound_png, title="Summary (Boundary)")
save_shap_summary(shap_2d, X_train, feature_names, sum_out_png, title="Summary (Outlier)")

show_images_row(
    [sum_clean_png, sum_bound_png, sum_out_png],
    ["Clean", "Boundary", "Outlier"],
    "SHAP Summary — Clean vs Boundary vs Outlier (side-by-side)",
    out_png="shap_summary_combined_1x3.png",
    figsize=(24, 6)
)

# ==========================================================
# DEPENDENCE with highlighted instance
# ==========================================================
dep_clean_png = "tmp_dep_clean.png"
dep_bound_png = "tmp_dep_boundary.png"
dep_out_png = "tmp_dep_outlier.png"

save_shap_dependence(shap_2d, X_train, feature_names, dep_clean_png,
                     title="Dependence (Clean highlighted)", highlight_idx=clean_tr_idx)
save_shap_dependence(shap_2d, X_train, feature_names, dep_bound_png,
                     title="Dependence (Boundary highlighted)", highlight_idx=boundary_tr_idx)
save_shap_dependence(shap_2d, X_train, feature_names, dep_out_png,
                     title="Dependence (Outlier highlighted)", highlight_idx=outlier_tr_idx)

show_images_row(
    [dep_clean_png, dep_bound_png, dep_out_png],
    ["Clean highlighted", "Boundary highlighted", "Outlier highlighted"],
    "SHAP Dependence (x1 colored by x2) — Clean vs Boundary vs Outlier (side-by-side)",
    out_png="shap_dependence_combined_1x3.png",
    figsize=(24, 6)
)

# ==========================================================
# WATERFALL 
# ==========================================================
wf_clean_png = "tmp_wf_clean.png"
wf_bound_png = "tmp_wf_boundary.png"
wf_out_png = "tmp_wf_outlier.png"

save_shap_waterfall(shap_2d, X_train, feature_names, base_1, clean_tr_idx, wf_clean_png,
                    title=f"Waterfall (Clean idx {clean_tr_idx})")
save_shap_waterfall(shap_2d, X_train, feature_names, base_1, boundary_tr_idx, wf_bound_png,
                    title=f"Waterfall (Boundary idx {boundary_tr_idx})")
save_shap_waterfall(shap_2d, X_train, feature_names, base_1, outlier_tr_idx, wf_out_png,
                    title=f"Waterfall (Outlier idx {outlier_tr_idx})")

show_images_row(
    [wf_clean_png, wf_bound_png, wf_out_png],
    ["Clean", "Boundary", "Outlier"],
    "SHAP Waterfall — Clean vs Boundary vs Outlier (side-by-side)",
    out_png="shap_waterfall_combined_1x3.png",
    figsize=(24, 6)
)

# Cleanup temps
for p in [
    sum_clean_png, sum_bound_png, sum_out_png,
    dep_clean_png, dep_bound_png, dep_out_png,
    wf_clean_png, wf_bound_png, wf_out_png
]:
    if os.path.exists(p):
        os.remove(p)

print("\nSaved outputs:")
print(" - rf_combined_shap_arrows_1x3.png")
print(" - shap_summary_combined_1x3.png")
print(" - shap_dependence_combined_1x3.png")
print(" - shap_waterfall_combined_1x3.png")
print(" - shap_comparison_combined_clean_boundary_outlier.csv")

# -------------------------------
# Compute SHAP fidelity for Clean, Boundary, Outlier
# -------------------------------
shap_clean_sum = np.sum(shap_clean)
pred_clean = rf.predict_proba(clean_point.reshape(1, -1))[0, 1]
fidelity_clean = abs(pred_clean - (base_1 + shap_clean_sum))

shap_bound_sum = np.sum(shap_boundary)
pred_bound = rf.predict_proba(boundary_point.reshape(1, -1))[0, 1]
fidelity_bound = abs(pred_bound - (base_1 + shap_bound_sum))

shap_out_sum = np.sum(shap_outlier)
pred_out = rf.predict_proba(outlier_point.reshape(1, -1))[0, 1]
fidelity_out = abs(pred_out - (base_1 + shap_out_sum))

# -------------------------------
# Print fidelity values
# -------------------------------
print("\nSHAP Fidelity Summary:")
print(f"Clean-region instance:")
print(f"  Predicted probability: {pred_clean:.6f}")
print(f"  Base + SHAP sum:      {base_1 + shap_clean_sum:.6f}")
print(f"  Fidelity error:       {fidelity_clean:.6f}\n")

print(f"Tight-boundary instance:")
print(f"  Predicted probability: {pred_bound:.6f}")
print(f"  Base + SHAP sum:      {base_1 + shap_bound_sum:.6f}")
print(f"  Fidelity error:       {fidelity_bound:.6f}\n")

print(f"Outlier instance:")
print(f"  Predicted probability: {pred_out:.6f}")
print(f"  Base + SHAP sum:      {base_1 + shap_out_sum:.6f}")
print(f"  Fidelity error:       {fidelity_out:.6f}")

# -------------------------------
# Plot fidelity bar chart (Clean vs Boundary vs Outlier)
# -------------------------------
instances = ["Clean-region", "Tight-boundary", "Outlier"]
fidelity_errors = [fidelity_clean, fidelity_bound, fidelity_out]
base_plus_shap = [base_1 + shap_clean_sum, base_1 + shap_bound_sum, base_1 + shap_out_sum]
pred_probs = [pred_clean, pred_bound, pred_out]

x = np.arange(len(instances))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))

# Bars
bars1 = ax.bar(x - width, fidelity_errors, width, label="Fidelity Error", color="salmon")
bars2 = ax.bar(x, base_plus_shap, width, label="Base + SHAP sum", color="skyblue")
bars3 = ax.bar(x + width, pred_probs, width, label="Predicted Probability", color="lightgreen")

# Labels, title, legend
ax.set_ylabel("Score / Error")
ax.set_xlabel("Instance Type")
ax.set_title("SHAP Fidelity Comparison — Clean vs Boundary vs Outlier")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.axhline(0, color="black", linewidth=1)
ax.legend()

# Add numeric labels on top of bars
for bar in bars1 + bars2 + bars3:
    height = bar.get_height()
    ax.annotate(f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

# ==========================================================
# SHAP STABILITY - RF Combined Dataset
# Clean vs Boundary vs Outlier
# ==========================================================

from sklearn.metrics.pairwise import cosine_similarity

def get_shap_vector_for_instance(explainer, instance):

    sv_inst = explainer(instance.reshape(1, -1))
    vals_inst = sv_inst.values

    if vals_inst.ndim == 2:
        shap_vec = vals_inst[0]

    elif vals_inst.ndim == 3 and vals_inst.shape[2] >= 2:
        shap_vec = vals_inst[0, :, 1]

    else:
        raise ValueError(f"Unexpected SHAP shape: {vals_inst.shape}")

    return shap_vec


def compute_shap_stability(explainer, instance, n_runs=30):

    shap_vectors = []

    for run in range(n_runs):
        shap_vec = get_shap_vector_for_instance(explainer, instance)
        shap_vectors.append(shap_vec)

    shap_vectors = np.array(shap_vectors)

    similarity_matrix = cosine_similarity(shap_vectors)

    # Remove diagonal values because each run is compared with itself
    off_diagonal_similarities = similarity_matrix[
        ~np.eye(similarity_matrix.shape[0], dtype=bool)
    ]

    stability_mean = np.mean(off_diagonal_similarities)
    stability_std = np.std(off_diagonal_similarities)

    return stability_mean, stability_std


# Compute stability
clean_stability_mean, clean_stability_std = compute_shap_stability(
    explainer,
    clean_point,
    n_runs=30
)

boundary_stability_mean, boundary_stability_std = compute_shap_stability(
    explainer,
    boundary_point,
    n_runs=30
)

outlier_stability_mean, outlier_stability_std = compute_shap_stability(
    explainer,
    outlier_point,
    n_runs=30
)

print("\nSHAP Stability Comparison - RF Combined Dataset:")
print("Clean-region Mean Cosine Similarity:", clean_stability_mean)
print("Clean-region Standard Deviation:", clean_stability_std)

print("Tight-boundary Mean Cosine Similarity:", boundary_stability_mean)
print("Tight-boundary Standard Deviation:", boundary_stability_std)

print("Outlier Mean Cosine Similarity:", outlier_stability_mean)
print("Outlier Standard Deviation:", outlier_stability_std)


# clean table
shap_stability_df = pd.DataFrame({
    "Model": ["Random Forest", "Random Forest", "Random Forest"],
    "Dataset": ["Combined", "Combined", "Combined"],
    "Instance": ["Clean-region", "Tight-boundary", "Outlier"],
    "Mean Cosine Similarity": [
        clean_stability_mean,
        boundary_stability_mean,
        outlier_stability_mean
    ],
    "Standard Deviation": [
        clean_stability_std,
        boundary_stability_std,
        outlier_stability_std
    ]
})

print("\nSHAP Stability Metric Table:")
print(shap_stability_df)

# ==========================================================
# SHAP SPARSITY - RF Combined Dataset
# Clean vs Boundary vs Outlier
# ==========================================================

def compute_shap_sparsity(shap_vector, feature_names, threshold=1e-6):

    shap_vector = np.array(shap_vector)

    non_zero_features = np.sum(np.abs(shap_vector) > threshold)
    total_features = len(feature_names)

    sparsity_score = 1 - (non_zero_features / total_features)

    return {
        "shap_vector": shap_vector,
        "non_zero_features": non_zero_features,
        "total_features": total_features,
        "sparsity_score": sparsity_score
    }


# Compute sparsity for all three instances
clean_sparsity = compute_shap_sparsity(
    shap_clean,
    feature_names
)

boundary_sparsity = compute_shap_sparsity(
    shap_boundary,
    feature_names
)

outlier_sparsity = compute_shap_sparsity(
    shap_outlier,
    feature_names
)

# Create sparsity table
shap_sparsity_df = pd.DataFrame({
    "Model": ["Random Forest", "Random Forest", "Random Forest"],
    "Dataset": ["Combined", "Combined", "Combined"],
    "Instance": ["Clean-region", "Tight-boundary", "Outlier"],
    "SHAP Vector": [
        clean_sparsity["shap_vector"],
        boundary_sparsity["shap_vector"],
        outlier_sparsity["shap_vector"]
    ],
    "Non-zero Features": [
        clean_sparsity["non_zero_features"],
        boundary_sparsity["non_zero_features"],
        outlier_sparsity["non_zero_features"]
    ],
    "Total Features": [
        clean_sparsity["total_features"],
        boundary_sparsity["total_features"],
        outlier_sparsity["total_features"]
    ],
    "Sparsity Score": [
        clean_sparsity["sparsity_score"],
        boundary_sparsity["sparsity_score"],
        outlier_sparsity["sparsity_score"]
    ]
})

print("\nSHAP Sparsity Metric Table:")
print(shap_sparsity_df)

# ==========================================================
# SHAP Sparsity Bar Plot
# ==========================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    shap_sparsity_df["Instance"],
    shap_sparsity_df["Sparsity Score"]
)

plt.ylabel("Sparsity Score")
plt.xlabel("Instance Type")
plt.title("SHAP Sparsity - RF Combined Dataset")
plt.axhline(0, color="black", linewidth=1)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.show()