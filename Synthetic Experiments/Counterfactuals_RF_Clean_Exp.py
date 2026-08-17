

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

# =============================
# Load model + data
# =============================
rf = joblib.load("rf_clean_model.pkl")
X_train = joblib.load("X_train_clean.pkl")
feature_names = joblib.load("feature_names_clean.pkl")
df = pd.read_csv("df_clean.csv")

X_train = np.asarray(X_train, dtype=np.float64)

# =============================
# Pick an instance to explain
# =============================
use_manual_df_index = False
manual_df_index = 0

df["distance"] = np.abs(df["x1"] - df["x2"]) / np.sqrt(2)

if use_manual_df_index:
    row = df.iloc[int(manual_df_index)]
else:
    row = df.sort_values("distance", ascending=True).iloc[0]  # closest to boundary

x = row[feature_names].values.astype(np.float64)
y_true = int(row["label"])

pred = int(rf.predict(x.reshape(1, -1))[0])
proba = rf.predict_proba(x.reshape(1, -1))[0]

print("Original instance x:", x, "label:", y_true)
print("RF prediction:", pred, "proba:", proba)

target = 1 - pred  # flip

# =============================
# Counterfactual search 
# =============================
def find_counterfactual_grid(
    model,
    x0,
    target_class,
    step=0.05,
    max_radius=2.0,
    grid_limit=120000
):
    x0 = np.asarray(x0, dtype=np.float64).reshape(1, -1)

    best_x = None
    best_dist = np.inf

    radii = np.arange(step, max_radius + 1e-12, step)

    for r in radii:
        vals = np.arange(-r, r + 1e-12, step)
        if len(vals) * len(vals) > grid_limit:
            stride = int(np.ceil(np.sqrt((len(vals) * len(vals)) / grid_limit)))
            vals = vals[::stride]

        dx1, dx2 = np.meshgrid(vals, vals)
        offsets = np.c_[dx1.ravel(), dx2.ravel()]

        Xcand = x0 + offsets
        preds = model.predict(Xcand).astype(int)

        mask = preds == int(target_class)
        if not np.any(mask):
            continue

        good = Xcand[mask]
        dists = np.linalg.norm(good - x0, axis=1)
        j = int(np.argmin(dists))

        if dists[j] < best_dist:
            best_dist = float(dists[j])
            best_x = good[j]

        break  # stop at first radius where solution exists

    return best_x, best_dist

x_cf, dist = find_counterfactual_grid(
    model=rf,
    x0=x,
    target_class=target,
    step=0.05,
    max_radius=3.0
)

if x_cf is None:
    raise RuntimeError(
        "No counterfactual found in the search region. "
        "Try increasing max_radius (e.g., 6.0) or using a smaller step (e.g., 0.02)."
    )

pred_cf = int(rf.predict(x_cf.reshape(1, -1))[0])
proba_cf = rf.predict_proba(x_cf.reshape(1, -1))[0]

delta = x_cf - x
print("\n--- Counterfactual found ---")
print("Target class:", target)
print("x_cf:", x_cf)
print("RF prediction(x_cf):", pred_cf, "proba:", proba_cf)
print("Delta (x_cf - x):", delta)
print("L2 distance:", dist)

# =============================
# Plot decision regions + original + counterfactual
# =============================
xx, yy = np.meshgrid(
    np.linspace(-5, 5, 400),
    np.linspace(-5, 5, 400)
)
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 7))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(df["x1"], df["x2"], c=df["label"], s=15, alpha=0.6)

plt.axline((0, 0), slope=1, color="k", linestyle="--", label="True boundary (x1=x2)")

plt.scatter(x[0], x[1], color="red", marker="X", s=170, edgecolor="black",
            label=f"Original (pred={pred})", zorder=5)
plt.scatter(x_cf[0], x_cf[1], color="lime", marker="X", s=170, edgecolor="black",
            label=f"Counterfactual (pred={pred_cf})", zorder=6)

plt.arrow(
    x[0], x[1],
    x_cf[0] - x[0], x_cf[1] - x[1],
    length_includes_head=True,
    head_width=0.18
)

plt.title("RF (Clean) — Counterfactual Explanation")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.tight_layout()
plt.savefig("rf_clean_counterfactual.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved: rf_clean_counterfactual.png")

# ==========================================================
# Feature change bar plot
# ==========================================================
plt.figure(figsize=(6, 4))
plt.bar(feature_names, delta)
plt.axhline(0, color="black", linewidth=1)
plt.title("Counterfactual Feature Changes (x_cf - x)")
plt.ylabel("Change required")
plt.tight_layout()
plt.savefig("rf_clean_counterfactual_feature_changes.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: rf_clean_counterfactual_feature_changes.png")

# ==========================================================
# Probability shift plot (P(Class 1))
# ==========================================================
p1_orig = float(proba[1]) if len(proba) > 1 else float(proba[0])
p1_cf = float(proba_cf[1]) if len(proba_cf) > 1 else float(proba_cf[0])

plt.figure(figsize=(6, 4))
plt.bar(["Original", "Counterfactual"], [p1_orig, p1_cf])
plt.ylim(0, 1)
plt.title("Prediction Probability Shift (P(Class 1))")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig("rf_clean_counterfactual_probability_shift.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: rf_clean_counterfactual_probability_shift.png")

# -------------------------------
# Compute Counterfactual Fidelity
# -------------------------------
pred_orig = rf.predict_proba(x.reshape(1,-1))[0,1]
pred_cf = rf.predict_proba(x_cf.reshape(1,-1))[0,1]

# Treat delta as "contribution" for simple fidelity check
fidelity_cf = abs(pred_cf - (pred_orig + np.sum(delta)))

print("\nCounterfactual Fidelity:")
print(f"Original prediction: {pred_orig:.6f}")
print(f"Counterfactual prediction: {pred_cf:.6f}")
print(f"Sum of feature changes (delta): {np.sum(delta):.6f}")
print(f"Fidelity error: {fidelity_cf:.6f}")

# ==========================================================
# Counterfactual Stability - RF Clean Dataset
# ==========================================================

from sklearn.metrics.pairwise import cosine_similarity

n_runs = 30
cf_rows = []
cf_vectors = []

for run in range(n_runs):
    x_cf_run, dist_run = find_counterfactual_grid(
        model=rf,
        x0=x,
        target_class=target,
        step=0.05,
        max_radius=3.0
    )

    if x_cf_run is None:
        continue

    delta_run = x_cf_run - x
    cf_vectors.append(delta_run)

    cf_rows.append({
        "run": run + 1,
        "x1_change": delta_run[0],
        "x2_change": delta_run[1],
        "l2_distance": dist_run
    })

cf_stability_df = pd.DataFrame(cf_rows)

print("\nCounterfactual repeated results:")
print(cf_stability_df.head())

cf_vectors = np.array(cf_vectors)

# Pairwise cosine similarity between repeated counterfactual delta vectors
similarity_matrix = cosine_similarity(cf_vectors)

# Remove diagonal values
off_diagonal_similarities = similarity_matrix[
    ~np.eye(similarity_matrix.shape[0], dtype=bool)
]

cf_stability_mean = np.mean(off_diagonal_similarities)
cf_stability_std = np.std(off_diagonal_similarities)

print("\nCounterfactual Stability - RF Clean Dataset:")
print("Mean Cosine Similarity:", cf_stability_mean)
print("Standard Deviation:", cf_stability_std)


# clean table
cf_stability_metric_df = pd.DataFrame({
    "Model": ["Random Forest"],
    "Dataset": ["Clean"],
    "Instance": ["Selected Instance"],
    "Mean Cosine Similarity": [cf_stability_mean],
    "Standard Deviation": [cf_stability_std],
    "Mean L2 Distance": [cf_stability_df["l2_distance"].mean()]
})

print("\nCounterfactual Stability Metric Table:")
print(cf_stability_metric_df)

# ==========================================================
# Counterfactual Sparsity - RF Clean Dataset
# ==========================================================

def compute_counterfactual_sparsity(delta_vector, feature_names, threshold=1e-6):
    delta_vector = np.asarray(delta_vector, dtype=float)

    changed_features = np.sum(np.abs(delta_vector) > threshold)
    total_features = len(feature_names)

    sparsity_score = 1 - (changed_features / total_features)

    return {
        "delta_vector": delta_vector,
        "changed_features": changed_features,
        "total_features": total_features,
        "sparsity_score": sparsity_score
    }


cf_sparsity = compute_counterfactual_sparsity(
    delta,
    feature_names
)

cf_sparsity_df = pd.DataFrame({
    "Model": ["Random Forest"],
    "Dataset": ["Clean"],
    "Instance": ["Selected Instance"],
    "Delta Vector": [cf_sparsity["delta_vector"]],
    "Changed Features": [cf_sparsity["changed_features"]],
    "Total Features": [cf_sparsity["total_features"]],
    "Sparsity Score": [cf_sparsity["sparsity_score"]]
})

print("\nCounterfactual Sparsity Metric Table:")
print(cf_sparsity_df)

# ==========================================================
# Counterfactual Sparsity Bar Plot
# ==========================================================

plt.figure(figsize=(6, 5))

bars = plt.bar(
    cf_sparsity_df["Instance"],
    cf_sparsity_df["Sparsity Score"]
)

plt.ylabel("Sparsity Score")
plt.xlabel("Instance")
plt.title("Counterfactual Sparsity - RF Clean Dataset")
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