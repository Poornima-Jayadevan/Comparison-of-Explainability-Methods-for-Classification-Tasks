
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
# save SHAP plots 
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

def save_shap_waterfall_local(shap_vals_1d, x_row, feature_names, base_value, out_png, title=None):
    exp = shap.Explanation(
        values=np.asarray(shap_vals_1d, dtype=float),
        base_values=float(base_value),
        data=np.asarray(x_row, dtype=float),
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
# Decision regions + TRUE boundary + SHAP arrows 
# ==========================================================
def arrow_len(v, max_abs, scale=2.0):
    max_abs = float(max_abs)
    if max_abs < 1e-12:
        return 0.0
    return scale * (float(v) / max_abs)

def plot_decision_regions_with_shap_arrows_side_by_side(
    nb, df, feature_names,
    clean_point, bound_point,
    shap_clean_1d, shap_bound_1d,
    out_png="nb_boundary_shap_arrows_side_by_side.png"
):
    
    xx, yy = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
    Z = nb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    max_abs = float(np.max(np.abs(np.vstack([shap_clean_1d, shap_bound_1d]))))
    if max_abs < 1e-12:
        max_abs = 1.0

    plt.figure(figsize=(14, 6))

    # LEFT: Clean-region
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(df["x1"], df["x2"], c=df["label"], s=12, alpha=0.4)
    plt.axline((0, 0), slope=1, linestyle="--", color="black")  # TRUE boundary

    plt.scatter(
        clean_point[0], clean_point[1],
        color="red", marker="X", s=160,
        edgecolor="black", label="Explained Instance", zorder=5
    )

    dx = arrow_len(shap_clean_1d[0], max_abs=max_abs, scale=2.0)
    dy = arrow_len(shap_clean_1d[1], max_abs=max_abs, scale=2.0)

    plt.arrow(clean_point[0], clean_point[1], dx, 0, head_width=0.25, length_includes_head=True,
              color="blue", zorder=6)
    plt.arrow(clean_point[0], clean_point[1], 0, dy, head_width=0.25, length_includes_head=True,
              color="blue", zorder=6)

    plt.annotate(
        f"({clean_point[0]:.2f}, {clean_point[1]:.2f})\n"
        f"SHAP x1={float(shap_clean_1d[0]):+.3f}\nSHAP x2={float(shap_clean_1d[1]):+.3f}",
        (clean_point[0], clean_point[1]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9
    )

    plt.xlim(-5, 5); plt.ylim(-5, 5)
    plt.title("Clean-region")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend()

    # RIGHT: Tight-boundary
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(df["x1"], df["x2"], c=df["label"], s=12, alpha=0.4)
    plt.axline((0, 0), slope=1, linestyle="--", color="black")  # TRUE boundary

    plt.scatter(
        bound_point[0], bound_point[1],
        color="red", marker="X", s=160,
        edgecolor="black", label="Explained Instance", zorder=5
    )

    dx = arrow_len(shap_bound_1d[0], max_abs=max_abs, scale=2.0)
    dy = arrow_len(shap_bound_1d[1], max_abs=max_abs, scale=2.0)

    plt.arrow(bound_point[0], bound_point[1], dx, 0, head_width=0.25, length_includes_head=True,
              color="green", zorder=6)
    plt.arrow(bound_point[0], bound_point[1], 0, dy, head_width=0.25, length_includes_head=True,
              color="green", zorder=6)

    plt.annotate(
        f"({bound_point[0]:.2f}, {bound_point[1]:.2f})\n"
        f"SHAP x1={float(shap_bound_1d[0]):+.3f}\nSHAP x2={float(shap_bound_1d[1]):+.3f}",
        (bound_point[0], bound_point[1]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9
    )

    plt.xlim(-5, 5); plt.ylim(-5, 5)
    plt.title("Tight-boundary")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend()

    plt.suptitle("NB (Clean+Boundary) — Decision Regions + SHAP Arrows (Side-by-side)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

# =============================
# Load model + data
# =============================
nb = joblib.load("nb_boundary_model.pkl")
X_train = joblib.load("X_train_boundary_nb.pkl")
feature_names = joblib.load("feature_names_boundary_nb.pkl")
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

clean_row = clean_df.sort_values("distance", ascending=False).iloc[0]
bound_row = boundary_df.sort_values("distance", ascending=True).iloc[0]

clean_point = clean_row[feature_names].values.astype(np.float64)
bound_point = bound_row[feature_names].values.astype(np.float64)

clean_tr_idx = nearest_train_index(X_train, clean_point)
bound_tr_idx = nearest_train_index(X_train, bound_point)

print("Clean train idx:", clean_tr_idx, "point:", clean_point, "label:", int(clean_row["label"]))
print("Boundary train idx:", bound_tr_idx, "point:", bound_point, "label:", int(bound_row["label"]))

# =============================
# Compute Kernel SHAP for NB 
# =============================
def predict_proba_class1(data):
    proba = nb.predict_proba(np.asarray(data, dtype=np.float64))
    return proba[:, 1]

bg_size = 50
background = shap.kmeans(X_train, bg_size)
explainer = shap.KernelExplainer(predict_proba_class1, background)
base_1 = float(explainer.expected_value)

# Global subset for plots 
explain_n = min(500, X_train.shape[0])
X_explain = X_train[:explain_n]
shap_2d = np.asarray(explainer.shap_values(X_explain, nsamples=200))  # (n, 2)

# Local SHAP for the two specific points 
X_two = np.vstack([clean_point, bound_point])
shap_two = np.asarray(explainer.shap_values(X_two, nsamples=400))  # (2, 2)
shap_clean_local = shap_two[0]
shap_bound_local = shap_two[1]


# ==========================================================
# PRINT LOCAL SHAP VALUES 
# ==========================================================

print("\nCLEAN INSTANCE SHAP EXPLANATION")
print(f"x1 = {clean_point[0]:.3f}, x2 = {clean_point[1]:.3f}")
print(f"SHAP x1 = {float(shap_clean_local[0]):+.4f}")
print(f"SHAP x2 = {float(shap_clean_local[1]):+.4f}")
print(f"Base value = {base_1:.4f}")

pred_clean = nb.predict_proba(clean_point.reshape(1,-1))[0,1]
print(f"Predicted probability (Class 1) = {pred_clean:.4f}")
print(f"Base + SHAP sum ≈ {base_1 + shap_clean_local[0] + shap_clean_local[1]:.4f}")


print("\nBOUNDARY INSTANCE SHAP EXPLANATION")
print(f"x1 = {bound_point[0]:.3f}, x2 = {bound_point[1]:.3f}")
print(f"SHAP x1 = {float(shap_bound_local[0]):+.4f}")
print(f"SHAP x2 = {float(shap_bound_local[1]):+.4f}")
print(f"Base value = {base_1:.4f}")

pred_bound = nb.predict_proba(bound_point.reshape(1,-1))[0,1]
print(f"Predicted probability (Class 1) = {pred_bound:.4f}")
print(f"Base + SHAP sum ≈ {base_1 + shap_bound_local[0] + shap_bound_local[1]:.4f}")


# ==========================================================
# Decision Regions + SHAP Arrows 
# ==========================================================
plot_decision_regions_with_shap_arrows_side_by_side(
    nb=nb,
    df=df,
    feature_names=feature_names,
    clean_point=clean_point,
    bound_point=bound_point,
    shap_clean_1d=shap_clean_local,
    shap_bound_1d=shap_bound_local,
    out_png="nb_boundary_shap_arrows_side_by_side.png"
)

# ==========================================================
# SUMMARY side-by-side 
# ==========================================================
sum_clean_png = "tmp_sum_clean.png"
sum_bound_png = "tmp_sum_bound.png"

save_shap_summary(shap_2d, X_explain, feature_names, sum_clean_png, title="Summary (Clean)")
save_shap_summary(shap_2d, X_explain, feature_names, sum_bound_png, title="Summary (Boundary)")

show_two_images_side_by_side(
    sum_clean_png, sum_bound_png,
    "Clean", "Boundary",
    "NB SHAP Summary — Clean vs Tight-boundary (side-by-side)",
    out_png="nb_shap_summary_clean_vs_boundary.png"
)

# ==========================================================
# DEPENDENCE side-by-side 
# ==========================================================
dep_clean_png = "tmp_dep_clean.png"
dep_bound_png = "tmp_dep_bound.png"

highlight_clean = clean_tr_idx if clean_tr_idx < X_explain.shape[0] else None
highlight_bound = bound_tr_idx if bound_tr_idx < X_explain.shape[0] else None

save_shap_dependence(
    shap_2d, X_explain, feature_names, dep_clean_png,
    title="Dependence (Clean highlighted)", highlight_idx=highlight_clean
)
save_shap_dependence(
    shap_2d, X_explain, feature_names, dep_bound_png,
    title="Dependence (Boundary highlighted)", highlight_idx=highlight_bound
)

show_two_images_side_by_side(
    dep_clean_png, dep_bound_png,
    "Clean highlighted", "Boundary highlighted",
    "NB SHAP Dependence (x1 colored by x2) — Clean vs Tight-boundary (side-by-side)",
    out_png="nb_shap_dependence_clean_vs_boundary.png"
)

# ==========================================================
# WATERFALL side-by-side 
# ==========================================================
wf_clean_png = "tmp_wf_clean.png"
wf_bound_png = "tmp_wf_bound.png"

save_shap_waterfall_local(
    shap_vals_1d=shap_clean_local,
    x_row=clean_point,
    feature_names=feature_names,
    base_value=base_1,
    out_png=wf_clean_png,
    title="Waterfall (Clean point)"
)
save_shap_waterfall_local(
    shap_vals_1d=shap_bound_local,
    x_row=bound_point,
    feature_names=feature_names,
    base_value=base_1,
    out_png=wf_bound_png,
    title="Waterfall (Boundary point)"
)

show_two_images_side_by_side(
    wf_clean_png, wf_bound_png,
    "Clean", "Boundary",
    "NB SHAP Waterfall — Clean vs Tight-boundary (side-by-side)",
    out_png="nb_shap_waterfall_clean_vs_boundary.png"
)

# Cleanup temp files
for p in [sum_clean_png, sum_bound_png, dep_clean_png, dep_bound_png, wf_clean_png, wf_bound_png]:
    if os.path.exists(p):
        os.remove(p)

print("\nSaved figures:")
print(" - nb_boundary_shap_arrows_side_by_side.png")
print(" - nb_shap_summary_clean_vs_boundary.png")
print(" - nb_shap_dependence_clean_vs_boundary.png")
print(" - nb_shap_waterfall_clean_vs_boundary.png")

# ==========================================================
# SHAP Comparison Table 
# ==========================================================
comparison_df = pd.DataFrame({
    "Feature": feature_names,
    "Clean-region SHAP": [float(shap_clean_local[i]) for i in range(len(feature_names))],
    "Tight-boundary SHAP": [float(shap_bound_local[i]) for i in range(len(feature_names))],
    "Abs Clean": [abs(float(shap_clean_local[i])) for i in range(len(feature_names))],
    "Abs Boundary": [abs(float(shap_bound_local[i])) for i in range(len(feature_names))],
})

print("\n=== NB SHAP Comparison Table (Clean vs Tight-boundary) ===")
print(comparison_df)

comparison_df.to_csv("nb_shap_comparison_clean_vs_boundary.csv", index=False)
print("Saved table: nb_shap_comparison_clean_vs_boundary.csv")

# -------------------------------
# SHAP Fidelity for the two instances
# -------------------------------
# Clean
pred_clean = nb.predict_proba(clean_point.reshape(1, -1))[0, 1]
shap_clean_sum = np.sum(shap_clean_local)
fidelity_clean = abs(pred_clean - (base_1 + shap_clean_sum))

# Boundary
pred_bound = nb.predict_proba(bound_point.reshape(1, -1))[0, 1]
shap_bound_sum = np.sum(shap_bound_local)
fidelity_bound = abs(pred_bound - (base_1 + shap_bound_sum))

print("\nSHAP Fidelity Summary:")
print(f"Clean-region instance:")
print(f"  Predicted probability: {pred_clean:.6f}")
print(f"  Base + SHAP sum:      {base_1 + shap_clean_sum:.6f}")
print(f"  Fidelity error:       {fidelity_clean:.6f}\n")

print(f"Tight-boundary instance:")
print(f"  Predicted probability: {pred_bound:.6f}")
print(f"  Base + SHAP sum:      {base_1 + shap_bound_sum:.6f}")
print(f"  Fidelity error:       {fidelity_bound:.6f}")

# -------------------------------
# SHAP Fidelity Bar Chart
# -------------------------------
instances = ["Clean-region", "Tight-boundary"]
fidelity_errors = [fidelity_clean, fidelity_bound]
base_plus_shap = [base_1 + shap_clean_sum, base_1 + shap_bound_sum]


pred_probs = [pred_clean, pred_bound]

x = np.arange(len(instances))
width = 0.25

fig, ax = plt.subplots(figsize=(7, 5))

bars_fid = ax.bar(x - width, fidelity_errors, width, label="Fidelity Error", color="salmon")
bars_base = ax.bar(x, base_plus_shap, width, label="Base + SHAP sum", color="skyblue")
bars_pred = ax.bar(x + width, pred_probs, width, label="Predicted Prob.", color="lightgreen")

ax.set_ylabel("Score / Error")
ax.set_xlabel("Instance Type")
ax.set_title("SHAP Fidelity Comparison — Clean vs Tight-boundary")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.axhline(0, color="black", linewidth=1)
ax.legend()

for bars in [bars_fid, bars_base, bars_pred]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

# ==========================================================
# SHAP STABILITY 
# ==========================================================

from sklearn.metrics.pairwise import cosine_similarity

def compute_kernel_shap_stability(explainer, instance, nsamples=400, n_runs=30):
    

    shap_vectors = []

    for run in range(n_runs):
        shap_vals_run = explainer.shap_values(
            instance.reshape(1, -1),
            nsamples=nsamples
        )

        shap_vec = np.asarray(shap_vals_run).reshape(-1)
        shap_vectors.append(shap_vec)

    shap_vectors = np.array(shap_vectors)

    similarity_matrix = cosine_similarity(shap_vectors)

    # Remove diagonal values
    off_diagonal_similarities = similarity_matrix[
        ~np.eye(similarity_matrix.shape[0], dtype=bool)
    ]

    stability_mean = np.mean(off_diagonal_similarities)
    stability_std = np.std(off_diagonal_similarities)

    return stability_mean, stability_std


clean_stability_mean, clean_stability_std = compute_kernel_shap_stability(
    explainer,
    clean_point,
    nsamples=400,
    n_runs=30
)

boundary_stability_mean, boundary_stability_std = compute_kernel_shap_stability(
    explainer,
    bound_point,
    nsamples=400,
    n_runs=30
)

print("\nSHAP Stability Comparison - NB Boundary Dataset:")
print("Clean-region Mean Cosine Similarity:", clean_stability_mean)
print("Clean-region Standard Deviation:", clean_stability_std)

print("Tight-boundary Mean Cosine Similarity:", boundary_stability_mean)
print("Tight-boundary Standard Deviation:", boundary_stability_std)


shap_stability_df = pd.DataFrame({
    "Model": ["Naive Bayes", "Naive Bayes"],
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
# SHAP SPARSITY 
# ==========================================================

def compute_shap_sparsity(shap_vector, feature_names, threshold=1e-6):

    shap_vector = np.asarray(shap_vector, dtype=float)

    non_zero_features = np.sum(np.abs(shap_vector) > threshold)
    total_features = len(feature_names)

    sparsity_score = 1 - (non_zero_features / total_features)

    return {
        "shap_vector": shap_vector,
        "non_zero_features": non_zero_features,
        "total_features": total_features,
        "sparsity_score": sparsity_score
    }


# Compute sparsity for both local SHAP explanations
clean_sparsity = compute_shap_sparsity(
    shap_clean_local,
    feature_names
)

boundary_sparsity = compute_shap_sparsity(
    shap_bound_local,
    feature_names
)

# Create sparsity table
shap_sparsity_df = pd.DataFrame({
    "Model": ["Naive Bayes", "Naive Bayes"],
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
plt.title("SHAP Sparsity - NB Boundary Dataset")
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