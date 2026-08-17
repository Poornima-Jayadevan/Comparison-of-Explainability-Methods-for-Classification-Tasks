# ------------------------------------------------------------
# SHAP for Gaussian Naive Bayes trained on CLEAN dataset
# Loads:
#   nb_clean_model.pkl
#   X_train_clean_nb.pkl
#   feature_names_clean_nb.pkl
#   df_clean.csv   (for the decision-boundary + true-boundary plot)
#
# Produces:
#   1) SHAP Summary plot
#   2) SHAP Dependence plot (x1 colored by x2)
#   3) SHAP Waterfall plot for one chosen instance (idx=0 by default)
#   4) Decision boundary + TRUE boundary + SHAP arrows + explained instance  ✅ (added)
#
# Notes:
# - For Naive Bayes we use KernelExplainer (model-agnostic).
# - KernelExplainer is slower; we use a small background set.
# - We explain predicted probability of Class 1.
# ------------------------------------------------------------

import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

# --- Fix for older SHAP versions with newer NumPy ---
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore

# =============================
# 1) Load model + data
# =============================
nb = joblib.load("nb_clean_model.pkl")
X_train = joblib.load("X_train_clean_nb.pkl")
feature_names = joblib.load("feature_names_clean_nb.pkl")

X = np.asarray(X_train, dtype=np.float64)

print("X shape:", X.shape)

# =============================
# 2) Choose a small background set (KernelExplainer needs this)
# =============================
bg_size = 50
background = shap.kmeans(X, bg_size)

# Explain a subset (speed)
explain_n = min(500, X.shape[0])
X_explain = X[:explain_n]

# =============================
# 3) SHAP KernelExplainer on P(Class=1)
# =============================
def predict_proba_class1(data):
    proba = nb.predict_proba(np.asarray(data, dtype=np.float64))
    return proba[:, 1]

explainer = shap.KernelExplainer(predict_proba_class1, background)

# nsamples controls runtime/quality tradeoff
shap_vals = explainer.shap_values(X_explain, nsamples=200)

base_value = explainer.expected_value
shap_2d = np.asarray(shap_vals)

print("shap_2d shape:", shap_2d.shape)
print("base_value:", base_value)

# =============================
# 4) Summary plot
# =============================
shap.summary_plot(shap_2d, X_explain, feature_names=feature_names, show=False)
plt.title("NB (Clean) — SHAP Summary (P(Class=1))")
plt.tight_layout()
plt.savefig("nb_clean_shap_summary.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================
# 5) Dependence plot (explicit interaction to avoid errors)
# =============================
shap.dependence_plot(
    "x1",
    shap_2d,
    X_explain,
    feature_names=feature_names,
    interaction_index="x2",
    show=False
)
plt.title("NB (Clean) — SHAP Dependence: x1 (colored by x2)")
plt.tight_layout()
plt.savefig("nb_clean_shap_dependence_x1.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================
# 6) Waterfall plot (local)
# =============================
idx = 0  # choose any idx within X_explain
exp = shap.Explanation(
    values=shap_2d[idx],
    base_values=float(base_value),
    data=X_explain[idx],
    feature_names=feature_names
)

shap.plots.waterfall(exp, show=False)
plt.title(f"NB (Clean) — SHAP Waterfall (instance {idx})")
plt.tight_layout()
plt.savefig("nb_clean_shap_waterfall.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# 7) ADDED: Decision boundary + True boundary + SHAP arrows
#     (matches the style of your LIME visualization)
# ==========================================================
df_clean = pd.read_csv("df_clean.csv")

# --- decision boundary grid (same as your NB training plot) ---
xx, yy = np.meshgrid(
    np.linspace(-5, 5, 400),
    np.linspace(-5, 5, 400)
)
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# SHAP arrows for the explained instance (idx)
# Scale arrows for visibility (normalized like you did in LIME)
max_abs = float(np.max(np.abs(shap_2d)))
if max_abs < 1e-12:
    max_abs = 1.0

def norm_arrow(v, max_abs, scale=2.0):
    return scale * float(v) / float(max_abs)

dx = norm_arrow(shap_2d[idx, 0], max_abs=max_abs, scale=2.0)
dy = norm_arrow(shap_2d[idx, 1], max_abs=max_abs, scale=2.0)

x0, y0 = float(X_explain[idx, 0]), float(X_explain[idx, 1])

# ---- print local SHAP values for the explained instance
shap_x1 = float(shap_2d[idx, 0])
shap_x2 = float(shap_2d[idx, 1])

print(f"\nExplained instance index: {idx}")
print(f"Instance values: x1 = {x0:.3f}, x2 = {y0:.3f}")
print(f"SHAP values: x1 = {shap_x1:+.6f}, x2 = {shap_x2:+.6f}")
print(f"Base value: {float(base_value):.6f}")

pred_proba = nb.predict_proba(X_explain[idx].reshape(1, -1))[0, 1]
print(f"Predicted probability for class 1: {pred_proba:.6f}")
print(f"Base + SHAP sum: {float(base_value) + shap_x1 + shap_x2:.6f}")


plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    df_clean["x1"],
    df_clean["x2"],
    c=df_clean["label"],
    s=15,
    alpha=0.8
)

# True boundary x1 - x2 = 0  (slope 1 line through origin)
plt.axline((0, 0), slope=1, linestyle="--", color="k", label="True Boundary (x1 - x2 = 0)")

# Explained instance (red X)
plt.scatter(x0, y0, color="red", marker="X", s=150, edgecolor="black", label="Explained Instance", zorder=5)

# SHAP arrows (x1 horizontal, x2 vertical)
plt.arrow(x0, y0, dx, 0, head_width=0.18, length_includes_head=True, color="blue", zorder=6)
plt.arrow(x0, y0, 0, dy, head_width=0.18, length_includes_head=True, color="blue", zorder=6)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("SHAP on NB (Clean)")
plt.legend()
plt.tight_layout()
plt.savefig("nb_clean_shap_boundary_with_instance.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ Saved PNGs:")
print(" - nb_clean_shap_summary.png")
print(" - nb_clean_shap_dependence_x1.png")
print(" - nb_clean_shap_waterfall.png")
print(" - nb_clean_shap_boundary_with_instance.png")

# =============================
# 8) SHAP Fidelity - Clean Instance
# =============================
shap_sum = np.sum(shap_2d[idx])  # sum of SHAP values for this instance
pred_prob = nb.predict_proba(X_explain[idx].reshape(1, -1))[0, 1]
base_plus_shap = float(base_value) + shap_sum
fidelity_error = abs(pred_prob - base_plus_shap)

print("\n🔍 SHAP Fidelity - Clean Instance:")
print(f"Predicted probability (model): {pred_prob:.6f}")
print(f"Base + SHAP sum: {base_plus_shap:.6f}")
print(f"Fidelity error: {fidelity_error:.6f}")

# =============================
# 9) Plot SHAP Fidelity
# =============================
import matplotlib.pyplot as plt

instances = ["Clean-instance"]
fidelity_errors = [fidelity_error]
base_shap_sum = [base_plus_shap]
predicted_prob = [pred_prob]

x = np.arange(len(instances))
width = 0.2

fig, ax = plt.subplots(figsize=(6, 5))

# Bars: fidelity error, base+SHAP, predicted probability
bars1 = ax.bar(x - width, fidelity_errors, width, label="Fidelity Error", color="salmon")
bars2 = ax.bar(x, base_shap_sum, width, label="Base + SHAP sum", color="skyblue")
bars3 = ax.bar(x + width, predicted_prob, width, label="Predicted Prob.", color="lightgreen")

# Labels, title, legend
ax.set_ylabel("Score / Error")
ax.set_xlabel("Instance")
ax.set_title("SHAP Fidelity - Clean Instance")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.axhline(0, color="black", linewidth=1)
ax.legend()

# Annotate numeric values
for bar in bars1 + bars2 + bars3:
    height = bar.get_height()
    ax.annotate(f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

# =============================
# 10) SHAP STABILITY - NB Clean Instance
# =============================
# For KernelExplainer, repeated explanations can vary slightly
# because it is model-agnostic and approximation-based.

from sklearn.metrics.pairwise import cosine_similarity

n_runs = 30
shap_vectors = []

for run in range(n_runs):
    shap_vals_run = explainer.shap_values(
        X_explain[idx].reshape(1, -1),
        nsamples=200
    )

    shap_vec = np.asarray(shap_vals_run).reshape(-1)

    shap_vectors.append(shap_vec)

shap_vectors = np.array(shap_vectors)

# Pairwise cosine similarity between repeated SHAP vectors
similarity_matrix = cosine_similarity(shap_vectors)

# Remove diagonal because each run is compared with itself
off_diagonal_similarities = similarity_matrix[
    ~np.eye(similarity_matrix.shape[0], dtype=bool)
]

shap_stability_mean = np.mean(off_diagonal_similarities)
shap_stability_std = np.std(off_diagonal_similarities)

print("\nSHAP Stability - NB Clean Instance:")
print("Mean Cosine Similarity:", shap_stability_mean)
print("Standard Deviation:", shap_stability_std)


# Optional table
shap_stability_df = pd.DataFrame({
    "Model": ["Naive Bayes"],
    "Dataset": ["Clean"],
    "Instance": ["Clean Instance"],
    "Mean Cosine Similarity": [shap_stability_mean],
    "Standard Deviation": [shap_stability_std]
})

print("\nSHAP Stability Metric Table:")
print(shap_stability_df)

# ==========================================================
# 11) SHAP SPARSITY - NB Clean Instance
# ==========================================================

def compute_shap_sparsity(shap_vector, feature_names, threshold=1e-6):
    """
    Sparsity = 1 - (# non-zero SHAP values / total features)

    Higher sparsity:
        fewer contributing features

    Lower sparsity:
        more contributing features
    """

    shap_vector = np.array(shap_vector)

    # Count non-zero SHAP values
    non_zero_features = np.sum(np.abs(shap_vector) > threshold)

    total_features = len(feature_names)

    sparsity_score = 1 - (non_zero_features / total_features)

    return {
        "shap_vector": shap_vector,
        "non_zero_features": non_zero_features,
        "total_features": total_features,
        "sparsity_score": sparsity_score
    }


# Compute sparsity for explained instance
sparsity_result = compute_shap_sparsity(
    shap_2d[idx],
    feature_names
)

# Create table
shap_sparsity_df = pd.DataFrame({
    "Model": ["Naive Bayes"],
    "Dataset": ["Clean"],
    "Instance": [f"Instance {idx}"],
    "SHAP Vector": [sparsity_result["shap_vector"]],
    "Non-zero Features": [sparsity_result["non_zero_features"]],
    "Total Features": [sparsity_result["total_features"]],
    "Sparsity Score": [sparsity_result["sparsity_score"]]
})

print("\nSHAP Sparsity Metric Table:")
print(shap_sparsity_df)

# ==========================================================
# Optional: SHAP Sparsity Plot
# ==========================================================

plt.figure(figsize=(6, 5))

bars = plt.bar(
    shap_sparsity_df["Instance"],
    shap_sparsity_df["Sparsity Score"],
    color="skyblue"
)

plt.ylabel("Sparsity Score")
plt.xlabel("Instance")
plt.title("SHAP Sparsity - NB Clean Dataset")
plt.axhline(0, color="black", linewidth=1)

# Add numeric labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.02,
        f"{height:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.show()