import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Fix for older SHAP versions with newer NumPy
if not hasattr(np, "bool"):
    np.bool = bool  

# =============================
# 1. Load model + data
# =============================
rf = joblib.load("rf_clean_model.pkl")
X_train = joblib.load("X_train_clean.pkl")
feature_names = joblib.load("feature_names_clean.pkl")


df_clean = pd.read_csv("df_clean.csv")

X = np.asarray(X_train, dtype=np.float64)

print("X shape:", X.shape)

# =============================
# 2. Explain using NEW API (robust)
# =============================
explainer = shap.TreeExplainer(rf)
sv = explainer(X)  # Explanation object

vals = sv.values
base = sv.base_values

print("sv.values shape:", np.shape(vals))
print("sv.base_values shape:", np.shape(base))

# =============================
# 3. Convert to 2D SHAP matrix for plotting
# =============================
if vals.ndim == 2:
    shap_2d = vals
    base_1 = float(base) if np.ndim(base) == 0 else float(np.mean(base))

elif vals.ndim == 3 and vals.shape[2] in [2, 3, 4, 5, 10]:  
    shap_2d = vals[:, :, 1]  
    if np.ndim(base) == 2:
        base_1 = float(np.mean(base[:, 1]))
    elif np.ndim(base) == 1 and base.shape[0] >= 2:
        base_1 = float(base[1])
    else:
        base_1 = float(np.mean(base))

elif vals.ndim == 3 and vals.shape[1] == vals.shape[2]:
    raise ValueError(
        f"You computed SHAP INTERACTION values of shape {vals.shape}. "
        "Use explainer(X) (as done here) and do NOT call explainer.shap_interaction_values()."
    )
else:
    raise ValueError(f"Unexpected sv.values shape: {vals.shape}")

print("Using shap_2d shape:", shap_2d.shape)

# =============================
# 4. Summary plot
# =============================
shap.summary_plot(shap_2d, X, feature_names=feature_names, show=False)
plt.title("RF (Clean) - SHAP Summary (Class 1 if applicable)")
plt.tight_layout()
plt.savefig("rf_clean_shap_summary.png", dpi=300)
plt.show()

# =============================
# 5. Dependence plot 
# =============================
shap.dependence_plot(
    "x1",
    shap_2d,
    X,
    feature_names=feature_names,
    interaction_index="x2",
    show=False
)
plt.title("RF (Clean) - SHAP Dependence: x1 (colored by x2)")
plt.tight_layout()
plt.savefig("rf_clean_shap_dependence_x1.png", dpi=300)
plt.show()

# =============================
# 6. Waterfall (local)
# =============================
idx = 0
exp = shap.Explanation(
    values=shap_2d[idx],
    base_values=base_1,
    data=X[idx],
    feature_names=feature_names
)

shap.plots.waterfall(exp, show=False)
plt.title(f"RF (Clean) - SHAP Waterfall (instance {idx})")
plt.tight_layout()
plt.savefig("rf_clean_shap_waterfall.png", dpi=300)
plt.show()

# ==========================================================
# 7. Decision regions + explained instance + SHAP arrows
# ==========================================================

instance = X[idx]
shap_inst = shap_2d[idx]
shap_x1, shap_x2 = float(shap_inst[0]), float(shap_inst[1])

print(f"\nExplained instance index: {idx}")
print(f"Instance values: x1 = {instance[0]:.3f}, x2 = {instance[1]:.3f}")
print(f"SHAP values: x1 = {shap_x1:+.6f}, x2 = {shap_x2:+.6f}")
print(f"Base value: {base_1:.6f}")

pred_proba = rf.predict_proba(instance.reshape(1, -1))[0, 1]
print(f"Predicted probability for class 1: {pred_proba:.6f}")
print(f"Base + SHAP sum: {base_1 + shap_x1 + shap_x2:.6f}")


# decision region grid 
xx, yy = np.meshgrid(
    np.linspace(-5, 5, 400),
    np.linspace(-5, 5, 400)
)
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# normalize arrows so they are visible
max_abs = np.max(np.abs(shap_2d))
if max_abs < 1e-12:
    max_abs = 1.0

def arrow_len(v, scale=2.0):
    return scale * (v / max_abs)

dx = arrow_len(shap_x1, scale=2.0)  # arrow along x1
dy = arrow_len(shap_x2, scale=2.0)  # arrow along x2

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.3)

# scatter full clean dataset for nicer plot
plt.scatter(
    df_clean["x1"], df_clean["x2"],
    c=df_clean["label"],
    s=15, alpha=0.7
)

# true boundary x1=x2
plt.axline((0, 0), slope=1, color="k", linestyle="--")

# explained instance marker
plt.scatter(
    instance[0], instance[1],
    color="red", marker="X",
    s=180, edgecolor="black", linewidth=1.5,
    label="Explained Instance",
    zorder=6
)

# SHAP arrows (x1 horizontal, x2 vertical)
plt.arrow(
    instance[0], instance[1], dx, 0,
    head_width=0.25, length_includes_head=True,
    color="black", zorder=7
)
plt.arrow(
    instance[0], instance[1], 0, dy,
    head_width=0.25, length_includes_head=True,
    color="black", zorder=7
)

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("SHAP on RF (Clean)")
plt.legend()
plt.tight_layout()
plt.savefig("rf_clean_shap_instance_arrows.png", dpi=300)
plt.show()

print("Saved PNGs: rf_clean_shap_summary.png, rf_clean_shap_dependence_x1.png, rf_clean_shap_waterfall.png, rf_clean_shap_instance_arrows.png")

# ==========================================================
# 8. SHAP Fidelity - Clean Instance
# ==========================================================


shap_sum = np.sum(shap_inst)  # sum of SHAP values for this instance
pred_prob = rf.predict_proba(instance.reshape(1, -1))[0, 1]

fidelity_error = abs(pred_prob - (base_1 + shap_sum))

print("\nSHAP Fidelity - Clean Instance:")
print(f"Predicted probability (model): {pred_prob:.6f}")
print(f"Base + SHAP sum: {base_1 + shap_sum:.6f}")
print(f"Fidelity error: {fidelity_error:.6f}")

plt.figure(figsize=(5,5))
plt.bar([0], [fidelity_error], width=0.35, color="salmon", label="Fidelity Error")
plt.bar([0], [base_1 + shap_sum], width=0.35, color="skyblue", label="Base + SHAP sum", alpha=0.6)
plt.xticks([0], [f"Clean Instance"])
plt.ylabel("Score / Probability")
plt.title("SHAP Fidelity - Clean Instance")
plt.legend()
plt.ylim(0,1)
plt.tight_layout()
plt.show()

# ==========================================================
# 9. SHAP STABILITY - Clean Instance
# ==========================================================


from sklearn.metrics.pairwise import cosine_similarity

n_runs = 30
shap_vectors = []

for run in range(n_runs):
    sv_run = explainer(instance.reshape(1, -1))
    vals_run = sv_run.values

    # Convert repeated SHAP output to 2D format
    if vals_run.ndim == 2:
        shap_vec = vals_run[0]

    elif vals_run.ndim == 3 and vals_run.shape[2] in [2, 3, 4, 5, 10]:
        shap_vec = vals_run[0, :, 1]

    else:
        raise ValueError(f"Unexpected SHAP values shape: {vals_run.shape}")

    shap_vectors.append(shap_vec)

shap_vectors = np.array(shap_vectors)

# Pairwise cosine similarity between SHAP vectors
similarity_matrix = cosine_similarity(shap_vectors)

# Remove diagonal values
off_diagonal_similarities = similarity_matrix[
    ~np.eye(similarity_matrix.shape[0], dtype=bool)
]

shap_stability_mean = np.mean(off_diagonal_similarities)
shap_stability_std = np.std(off_diagonal_similarities)

print("\nSHAP Stability - RF Clean Instance:")
print("Mean Cosine Similarity:", shap_stability_mean)
print("Standard Deviation:", shap_stability_std)

shap_stability_df = pd.DataFrame({
    "Model": ["Random Forest"],
    "Dataset": ["Clean"],
    "Instance": ["Clean Instance"],
    "Mean Cosine Similarity": [shap_stability_mean],
    "Standard Deviation": [shap_stability_std]
})

print("\nSHAP Stability Metric Table:")
print(shap_stability_df)

# ==========================================================
# 10. SHAP SPARSITY - RF Clean Instance
# ==========================================================

def compute_shap_sparsity(shap_values_instance, feature_names, threshold=1e-6):
    

    shap_vector = np.array(shap_values_instance)

    non_zero_features = np.sum(np.abs(shap_vector) > threshold)
    total_features = len(shap_vector)

    sparsity_score = 1 - (non_zero_features / total_features)

    return {
        "shap_vector": shap_vector,
        "non_zero_features": non_zero_features,
        "total_features": total_features,
        "sparsity_score": sparsity_score
    }


# Compute sparsity for selected clean instance
shap_sparsity = compute_shap_sparsity(
    shap_inst,
    feature_names
)

shap_sparsity_df = pd.DataFrame({
    "Model": ["Random Forest"],
    "Dataset": ["Clean"],
    "Instance": [f"Instance {idx}"],
    "SHAP Vector": [shap_sparsity["shap_vector"]],
    "Non-zero Features": [shap_sparsity["non_zero_features"]],
    "Total Features": [shap_sparsity["total_features"]],
    "Sparsity Score": [shap_sparsity["sparsity_score"]]
})

print("\nSHAP Sparsity Metric Table:")
print(shap_sparsity_df)

# ==========================================================
# 11. SHAP Sparsity Bar Plot
# ==========================================================

plt.figure(figsize=(6, 5))

bars = plt.bar(
    shap_sparsity_df["Instance"],
    shap_sparsity_df["Sparsity Score"]
)

plt.ylabel("Sparsity Score")
plt.xlabel("Instance")
plt.title("SHAP Sparsity - RF Clean Instance")
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