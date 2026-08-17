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
# save SHAP plots as PNGs + show 2 images side-by-side
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
    # highlight one instance
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

def show_two_images_side_by_side(png_left, png_right, title_left, title_right, big_title, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    img1 = plt.imread(png_left)
    img2 = plt.imread(png_right)

    axes[0].imshow(img1)
    axes[0].axis("off")
    axes[0].set_title(title_left, fontsize=12)

    axes[1].imshow(img2)
    axes[1].axis("off")
    axes[1].set_title(title_right, fontsize=12)

    fig.suptitle(big_title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

def nearest_train_index(Xtr, pt):
    d = np.linalg.norm(Xtr - pt.reshape(1, -1), axis=1)
    return int(np.argmin(d))

# ==========================================================
# Decision regions + SHAP arrows (side-by-side)
# ==========================================================
def arrow_len(v, max_abs, scale=2.0):
    max_abs = float(max_abs)
    if max_abs < 1e-12:
        return 0.0
    return scale * (float(v) / max_abs)

def plot_decision_regions_with_shap_arrows_side_by_side(
    rf, df, feature_names, shap_2d,
    clean_point, bound_point,
    clean_tr_idx, bound_tr_idx,
    out_png="rf_boundary_shap_arrows_side_by_side.png"
):
    xx, yy = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    max_abs = float(np.max(np.abs(shap_2d)))
    if max_abs < 1e-12:
        max_abs = 1.0

    plt.figure(figsize=(14, 6))

    # LEFT: Clean-region
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(df["x1"], df["x2"], c=df["label"], s=12, alpha=0.4)
    plt.axline((0, 0), slope=1, linestyle="--", color="black")

    plt.scatter(
        clean_point[0], clean_point[1],
        color="red", marker="X", s=160,
        edgecolor="black", label="Explained Instance", zorder=5
    )

    shap_clean = shap_2d[clean_tr_idx]
    dx = arrow_len(shap_clean[0], max_abs=max_abs, scale=2.0)
    dy = arrow_len(shap_clean[1], max_abs=max_abs, scale=2.0)

    plt.arrow(clean_point[0], clean_point[1], dx, 0, head_width=0.25, length_includes_head=True,
              color="blue", zorder=6)
    plt.arrow(clean_point[0], clean_point[1], 0, dy, head_width=0.25, length_includes_head=True,
              color="blue", zorder=6)

    plt.annotate(
        f"train idx={clean_tr_idx}\n({clean_point[0]:.2f}, {clean_point[1]:.2f})\n"
        f"SHAP x1={float(shap_clean[0]):+.3f}\nSHAP x2={float(shap_clean[1]):+.3f}",
        (clean_point[0], clean_point[1]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9
    )

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title("Clean-region")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    # RIGHT: Tight-boundary
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(df["x1"], df["x2"], c=df["label"], s=12, alpha=0.4)
    plt.axline((0, 0), slope=1, linestyle="--", color="black")

    plt.scatter(
        bound_point[0], bound_point[1],
        color="red", marker="X", s=160,
        edgecolor="black", label="Explained Instance", zorder=5
    )

    shap_bound = shap_2d[bound_tr_idx]
    dx = arrow_len(shap_bound[0], max_abs=max_abs, scale=2.0)
    dy = arrow_len(shap_bound[1], max_abs=max_abs, scale=2.0)

    plt.arrow(bound_point[0], bound_point[1], dx, 0, head_width=0.25, length_includes_head=True,
              color="green", zorder=6)
    plt.arrow(bound_point[0], bound_point[1], 0, dy, head_width=0.25, length_includes_head=True,
              color="green", zorder=6)

    plt.annotate(
        f"train idx={bound_tr_idx}\n({bound_point[0]:.2f}, {bound_point[1]:.2f})\n"
        f"SHAP x1={float(shap_bound[0]):+.3f}\nSHAP x2={float(shap_bound[1]):+.3f}",
        (bound_point[0], bound_point[1]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9
    )

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title("Tight-boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    plt.suptitle("RF (Clean+Boundary) — Decision Regions + SHAP Arrows (Side-by-side)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

# =============================
# Load model + data
# =============================
rf = joblib.load("rf_boundary_model.pkl")
X_train = joblib.load("X_train_boundary.pkl")
feature_names = joblib.load("feature_names_boundary.pkl")
df = pd.read_csv("clean_plus_tight_boundary.csv")

X_train = np.asarray(X_train, dtype=np.float64)

# =============================
# Pick CLEAN vs BOUNDARY instances from df
# =============================
df["distance"] = np.abs(df["x1"] - df["x2"]) / np.sqrt(2)
region_lower = df["region"].astype(str).str.lower()

clean_df = df[region_lower.str.contains("clean")]
boundary_df = df[region_lower.str.contains("bound")]

if len(clean_df) == 0:
    raise ValueError("No clean region rows found in df['region'].")
if len(boundary_df) == 0:
    raise ValueError("No boundary region rows found in df['region'].")

clean_row = clean_df.sort_values("distance", ascending=False).iloc[0]     # farthest from boundary
bound_row = boundary_df.sort_values("distance", ascending=True).iloc[0]   # closest to boundary

clean_point = clean_row[feature_names].values.astype(np.float64)
bound_point = bound_row[feature_names].values.astype(np.float64)

clean_tr_idx = nearest_train_index(X_train, clean_point)
bound_tr_idx = nearest_train_index(X_train, bound_point)

print("Clean train idx:", clean_tr_idx, "point:", clean_point, "label:", int(clean_row["label"]))
print("Boundary train idx:", bound_tr_idx, "point:", bound_point, "label:", int(bound_row["label"]))

# =============================
# Compute SHAP
# =============================
explainer = shap.TreeExplainer(rf)
sv = explainer(X_train)

vals = sv.values
base = sv.base_values

# Convert to 2D SHAP matrix 
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

# ==========================================================
# Decision Regions + SHAP Arrows 
# ==========================================================
plot_decision_regions_with_shap_arrows_side_by_side(
    rf=rf,
    df=df,
    feature_names=feature_names,
    shap_2d=shap_2d,
    clean_point=clean_point,
    bound_point=bound_point,
    clean_tr_idx=clean_tr_idx,
    bound_tr_idx=bound_tr_idx,
    out_png="rf_boundary_shap_arrows_side_by_side.png"
)

# ==========================================================
# SUMMARY side-by-side (Clean vs Boundary)
# ==========================================================
sum_clean_png = "tmp_sum_clean.png"
sum_bound_png = "tmp_sum_bound.png"

save_shap_summary(shap_2d, X_train, feature_names, sum_clean_png, title="Summary (Clean)")
save_shap_summary(shap_2d, X_train, feature_names, sum_bound_png, title="Summary (Boundary)")

show_two_images_side_by_side(
    sum_clean_png, sum_bound_png,
    "Clean", "Boundary",
    "SHAP Summary — Clean vs Tight-boundary (side-by-side)",
    out_png="shap_summary_clean_vs_boundary.png"
)

# ==========================================================
# DEPENDENCE side-by-side 
# ==========================================================
dep_clean_png = "tmp_dep_clean.png"
dep_bound_png = "tmp_dep_bound.png"

save_shap_dependence(
    shap_2d, X_train, feature_names, dep_clean_png,
    title="Dependence (Clean highlighted)", highlight_idx=clean_tr_idx
)
save_shap_dependence(
    shap_2d, X_train, feature_names, dep_bound_png,
    title="Dependence (Boundary highlighted)", highlight_idx=bound_tr_idx
)

show_two_images_side_by_side(
    dep_clean_png, dep_bound_png,
    "Clean highlighted", "Boundary highlighted",
    "SHAP Dependence (x1 colored by x2) — Clean vs Tight-boundary (side-by-side)",
    out_png="shap_dependence_clean_vs_boundary.png"
)

# ==========================================================
# WATERFALL side-by-side 
# ==========================================================
wf_clean_png = "tmp_wf_clean.png"
wf_bound_png = "tmp_wf_bound.png"

save_shap_waterfall(
    shap_2d, X_train, feature_names, base_1, clean_tr_idx, wf_clean_png,
    title=f"Waterfall (Clean idx {clean_tr_idx})"
)
save_shap_waterfall(
    shap_2d, X_train, feature_names, base_1, bound_tr_idx, wf_bound_png,
    title=f"Waterfall (Boundary idx {bound_tr_idx})"
)

show_two_images_side_by_side(
    wf_clean_png, wf_bound_png,
    "Clean", "Boundary",
    "SHAP Waterfall — Clean vs Tight-boundary (side-by-side)",
    out_png="shap_waterfall_clean_vs_boundary.png"
)

# Cleanup temp files
for p in [sum_clean_png, sum_bound_png, dep_clean_png, dep_bound_png, wf_clean_png, wf_bound_png]:
    if os.path.exists(p):
        os.remove(p)

print("\nSaved figures:")
print(" - rf_boundary_shap_arrows_side_by_side.png")
print(" - shap_summary_clean_vs_boundary.png")
print(" - shap_dependence_clean_vs_boundary.png")
print(" - shap_waterfall_clean_vs_boundary.png")

# ==========================================================
# SHAP Comparison Table (Clean vs Tight-boundary)
# ==========================================================
shap_clean = shap_2d[clean_tr_idx]
shap_bound = shap_2d[bound_tr_idx]

# =============================
# PRINT SHAP VALUES 
# =============================

print("\nCLEAN INSTANCE EXPLANATION")
print(f"Index: {clean_tr_idx}")
print(f"x1 = {clean_point[0]:.3f}, x2 = {clean_point[1]:.3f}")
print(f"SHAP x1 = {float(shap_clean[0]):+.4f}")
print(f"SHAP x2 = {float(shap_clean[1]):+.4f}")

pred_clean = rf.predict_proba(clean_point.reshape(1, -1))[0, 1]
print(f"Predicted probability (Class 1): {pred_clean:.4f}")
print(f"Base + SHAP sum ≈ {base_1 + shap_clean[0] + shap_clean[1]:.4f}")


print("\nBOUNDARY INSTANCE EXPLANATION")
print(f"Index: {bound_tr_idx}")
print(f"x1 = {bound_point[0]:.3f}, x2 = {bound_point[1]:.3f}")
print(f"SHAP x1 = {float(shap_bound[0]):+.4f}")
print(f"SHAP x2 = {float(shap_bound[1]):+.4f}")

pred_bound = rf.predict_proba(bound_point.reshape(1, -1))[0, 1]
print(f"Predicted probability (Class 1): {pred_bound:.4f}")
print(f"Base + SHAP sum ≈ {base_1 + shap_bound[0] + shap_bound[1]:.4f}")


comparison_df = pd.DataFrame({
    "Feature": feature_names,
    "Clean-region SHAP": [float(shap_clean[i]) for i in range(len(feature_names))],
    "Tight-boundary SHAP": [float(shap_bound[i]) for i in range(len(feature_names))],
    "Abs Clean": [abs(float(shap_clean[i])) for i in range(len(feature_names))],
    "Abs Boundary": [abs(float(shap_bound[i])) for i in range(len(feature_names))],
})

print("\n=== SHAP Comparison Table (Clean vs Tight-boundary) ===")
print(comparison_df)

comparison_df.to_csv("shap_comparison_clean_vs_boundary.csv", index=False)
print("Saved table: shap_comparison_clean_vs_boundary.csv")

# -------------------------------
# Compute fidelity for clean and boundary
# -------------------------------
shap_clean_sum = np.sum(shap_clean)
pred_clean = rf.predict_proba(clean_point.reshape(1, -1))[0, 1]
fidelity_clean = abs(pred_clean - (base_1 + shap_clean_sum))

shap_bound_sum = np.sum(shap_bound)
pred_bound = rf.predict_proba(bound_point.reshape(1, -1))[0, 1]
fidelity_bound = abs(pred_bound - (base_1 + shap_bound_sum))

# -------------------------------
# Print table 
# -------------------------------
print("=== SHAP Comparison Table (Clean vs Tight-boundary) ===")
print(comparison_df)

# -------------------------------
# Print fidelity summary below table
# -------------------------------
print("\nSHAP Fidelity Summary:")
print(f"Clean-region instance:")
print(f"  Predicted probability (Class 1): {pred_clean:.6f}")
print(f"  Base + SHAP sum: {base_1 + shap_clean_sum:.6f}")
print(f"  Fidelity error: {fidelity_clean:.6f}\n")

print(f"Tight-boundary instance:")
print(f"  Predicted probability (Class 1): {pred_bound:.6f}")
print(f"  Base + SHAP sum: {base_1 + shap_bound_sum:.6f}")
print(f"  Fidelity error: {fidelity_bound:.6f}")

# -------------------------------
# SHAP fidelity values
# -------------------------------
instances = ["Clean-region", "Tight-boundary"]
fidelity_errors = [fidelity_clean, fidelity_bound]
base_plus_shap = [base_1 + shap_clean_sum, base_1 + shap_bound_sum]
pred_probs = [pred_clean, pred_bound]

x = np.arange(len(instances))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))

# Bars for fidelity error, base+shap, and model prediction
bars1 = ax.bar(x - width, fidelity_errors, width, label="Fidelity Error", color="salmon")
bars2 = ax.bar(x, base_plus_shap, width, label="Base + SHAP sum", color="skyblue")
bars3 = ax.bar(x + width, pred_probs, width, label="Predicted Prob.", color="lightgreen")

# Labels, title, legend
ax.set_ylabel("Score / Error")
ax.set_xlabel("Instance Type")
ax.set_title("SHAP Fidelity Comparison - Clean vs Tight-boundary")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.axhline(0, color="black", linewidth=1)
ax.legend()

# Add numeric labels on top of bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

# ==========================================================
# SHAP STABILITY - Clean vs Tight-boundary
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


# Compute stability for clean and boundary instances
clean_stability_mean, clean_stability_std = compute_shap_stability(
    explainer,
    clean_point,
    n_runs=30
)

boundary_stability_mean, boundary_stability_std = compute_shap_stability(
    explainer,
    bound_point,
    n_runs=30
)

print("\nSHAP Stability Comparison - RF Boundary Dataset:")
print("Clean-region Mean Cosine Similarity:", clean_stability_mean)
print("Clean-region Standard Deviation:", clean_stability_std)

print("Tight-boundary Mean Cosine Similarity:", boundary_stability_mean)
print("Tight-boundary Standard Deviation:", boundary_stability_std)


# clean table
shap_stability_df = pd.DataFrame({
    "Model": ["Random Forest", "Random Forest"],
    "Dataset": ["Boundary", "Boundary"],
    "Instance": ["Clean-region", "Tight-boundary"],
    "Mean Cosine Similarity": [
        clean_stability_mean,
        boundary_stability_mean
    ],
    "Standard Deviation": [
        clean_stability_std,
        boundary_stability_std
    ]
})

print("\nSHAP Stability Metric Table:")
print(shap_stability_df)

# ==========================================================
# SHAP SPARSITY - Clean vs Tight-boundary
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


# Compute sparsity for clean and boundary instances
clean_sparsity = compute_shap_sparsity(
    shap_clean,
    feature_names
)

boundary_sparsity = compute_shap_sparsity(
    shap_bound,
    feature_names
)

# Create sparsity table
shap_sparsity_df = pd.DataFrame({
    "Model": ["Random Forest", "Random Forest"],
    "Dataset": ["Boundary", "Boundary"],
    "Instance": ["Clean-region", "Tight-boundary"],
    "SHAP Vector": [
        clean_sparsity["shap_vector"],
        boundary_sparsity["shap_vector"]
    ],
    "Non-zero Features": [
        clean_sparsity["non_zero_features"],
        boundary_sparsity["non_zero_features"]
    ],
    "Total Features": [
        clean_sparsity["total_features"],
        boundary_sparsity["total_features"]
    ],
    "Sparsity Score": [
        clean_sparsity["sparsity_score"],
        boundary_sparsity["sparsity_score"]
    ]
})

print("\nSHAP Sparsity Metric Table:")
print(shap_sparsity_df)

# ==========================================================
# SHAP Sparsity Bar Plot
# ==========================================================

plt.figure(figsize=(7, 5))

bars = plt.bar(
    shap_sparsity_df["Instance"],
    shap_sparsity_df["Sparsity Score"]
)

plt.ylabel("Sparsity Score")
plt.xlabel("Instance Type")
plt.title("SHAP Sparsity - RF Boundary Dataset")
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