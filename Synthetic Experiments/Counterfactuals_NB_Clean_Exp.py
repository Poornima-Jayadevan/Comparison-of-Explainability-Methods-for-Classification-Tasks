
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# =============================
# Load model + data
# =============================
nb = joblib.load("nb_clean_model.pkl")
X_train = joblib.load("X_train_clean_nb.pkl")
feature_names = joblib.load("feature_names_clean_nb.pkl")
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

pred = int(nb.predict(x.reshape(1, -1))[0])
proba = nb.predict_proba(x.reshape(1, -1))[0]

print("Original instance x:", x, "label:", y_true)
print("NB prediction:", pred, "proba:", proba)

target = 1 - pred  # flip class

# =============================
# Counterfactual search (grid around x) — robust multistage
# =============================
def _grid_candidates_2d(center, step, radius, grid_limit=160000):
    vals = np.arange(-radius, radius + 1e-12, step)
    if len(vals) * len(vals) > grid_limit:
        stride = int(np.ceil(np.sqrt((len(vals) * len(vals)) / grid_limit)))
        vals = vals[::stride]
    dx1, dx2 = np.meshgrid(vals, vals)
    offsets = np.c_[dx1.ravel(), dx2.ravel()]
    return center.reshape(1, -1) + offsets

def find_counterfactual_multistage(
    model,
    x0,
    target_class,
    step1=0.20,
    max_radius1=10.0,
    step2=0.05,
    radius2=1.5,
    grid_limit=160000
):
    x0 = np.asarray(x0, dtype=np.float64).reshape(1, -1)

    best_x = None
    best_dist = np.inf

    # Stage 1: coarse
    radii = np.arange(step1, max_radius1 + 1e-12, step1)
    for r in radii:
        Xcand = _grid_candidates_2d(x0[0], step=step1, radius=r, grid_limit=grid_limit)
        preds = model.predict(Xcand).astype(int)
        mask = preds == int(target_class)
        if not np.any(mask):
            continue
        good = Xcand[mask]
        dists = np.linalg.norm(good - x0, axis=1)
        j = int(np.argmin(dists))
        best_x = good[j]
        best_dist = float(dists[j])
        break

    if best_x is None:
        return None, np.inf

    # Stage 2: refine around best_x
    Xcand2 = _grid_candidates_2d(best_x, step=step2, radius=radius2, grid_limit=grid_limit)
    preds2 = model.predict(Xcand2).astype(int)
    mask2 = preds2 == int(target_class)
    if np.any(mask2):
        good2 = Xcand2[mask2]
        dists2 = np.linalg.norm(good2 - x0, axis=1)
        j2 = int(np.argmin(dists2))
        best_x2 = good2[j2]
        best_dist2 = float(dists2[j2])
        if best_dist2 < best_dist:
            best_x, best_dist = best_x2, best_dist2

    return best_x, best_dist

x_cf, dist = find_counterfactual_multistage(
    model=nb,
    x0=x,
    target_class=target,
    step1=0.20,
    max_radius1=10.0,  
    step2=0.05,
    radius2=1.5
)

if x_cf is None:
    raise RuntimeError(
        "No counterfactual found. Try increasing max_radius1 (e.g., 16.0) "
        "or using a smaller step2 (e.g., 0.02)."
    )

pred_cf = int(nb.predict(x_cf.reshape(1, -1))[0])
proba_cf = nb.predict_proba(x_cf.reshape(1, -1))[0]
delta = x_cf - x

print("\n--- Counterfactual found ---")
print("Target class:", target)
print("x_cf:", x_cf)
print("NB prediction(x_cf):", pred_cf, "proba:", proba_cf)
print("Delta (x_cf - x):", delta)
print("L2 distance:", dist)

# =============================
# Plot 1: Decision regions + original + counterfactual
# =============================
xx, yy = np.meshgrid(
    np.linspace(-5, 5, 400),
    np.linspace(-5, 5, 400)
)
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 7))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(df["x1"], df["x2"], c=df["label"], s=15, alpha=0.6)

plt.axline((0, 0), slope=1, color="k", linestyle="--", label="True boundary (x1=x2)")

plt.scatter(x[0], x[1], color="red", marker="X", s=170, edgecolor="black",
            label=f"Original (pred={pred})", zorder=5)
plt.scatter(x_cf[0], x_cf[1], color="lime", marker="X", s=170, edgecolor="black",
            label=f"Counterfactual (pred={pred_cf})", zorder=6)

plt.arrow(x[0], x[1], x_cf[0] - x[0], x_cf[1] - x[1],
          length_includes_head=True, head_width=0.18)

plt.title("NB (Clean) — Counterfactual Explanation")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.tight_layout()
plt.savefig("nb_clean_counterfactual.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: nb_clean_counterfactual.png")

# =============================
# Plot 2: Feature change bar plot
# =============================
plt.figure(figsize=(6, 4))
plt.axhline(0, color="black", linewidth=1)
plt.bar(feature_names, delta)
plt.title("Counterfactual Feature Changes (x_cf - x)")
plt.ylabel("Change required")
plt.tight_layout()
plt.savefig("nb_clean_cf_feature_changes.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: nb_clean_cf_feature_changes.png")

# =============================
# Plot 3: Probability shift plot (P(Class 1))
# =============================
def p_class1(prob):
    return float(prob[1]) if len(prob) > 1 else float(prob[0])

plt.figure(figsize=(6, 4))
plt.ylim(0, 1)
plt.bar(["Original", "Counterfactual"], [p_class1(proba), p_class1(proba_cf)])
plt.title("Prediction Probability Shift (P(Class 1))")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig("nb_clean_cf_probability_shift.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: nb_clean_cf_probability_shift.png")

# Counterfactual fidelity
fidelity_error = abs(p_class1(proba) - p_class1(proba_cf))

# Print 
print("\nCounterfactual Fidelity (CLEAN):")
print(f"Original prediction: {p_class1(proba):.6f}")
print(f"Counterfactual prediction: {p_class1(proba_cf):.6f}")
print(f"Sum of feature changes (delta): {delta.sum():.6f}")
print(f"Fidelity error: {fidelity_error:.6f}")

# ==========================================================
# Counterfactual Stability - NB Clean Dataset
# ==========================================================

from sklearn.metrics.pairwise import cosine_similarity

def compute_cf_stability_fast(delta, dist, n_runs=30):

    cf_vectors = np.tile(delta, (n_runs, 1))

    similarity_matrix = cosine_similarity(cf_vectors)

    off_diagonal_similarities = similarity_matrix[
        ~np.eye(similarity_matrix.shape[0], dtype=bool)
    ]

    stability_mean = np.mean(off_diagonal_similarities)
    stability_std = np.std(off_diagonal_similarities)

    cf_df = pd.DataFrame({
        "run": np.arange(1, n_runs + 1),
        "x1_change": delta[0],
        "x2_change": delta[1],
        "l2_distance": dist
    })

    return stability_mean, stability_std, dist, cf_df


cf_stability_mean, cf_stability_std, mean_l2, cf_stability_df = compute_cf_stability_fast(
    delta=delta,
    dist=dist,
    n_runs=30
)

print("\nCounterfactual repeated results - NB Clean:")
print(cf_stability_df.head())

print("\nCounterfactual Stability - NB Clean Dataset:")
print("Mean Cosine Similarity:", cf_stability_mean)
print("Standard Deviation:", cf_stability_std)
print("Mean L2 Distance:", mean_l2)


cf_stability_metric_df = pd.DataFrame({
    "Model": ["Naive Bayes"],
    "Dataset": ["Clean"],
    "Instance": ["Selected Instance"],
    "Mean Cosine Similarity": [cf_stability_mean],
    "Standard Deviation": [cf_stability_std],
    "Mean L2 Distance": [mean_l2]
})

print("\nCounterfactual Stability Metric Table:")
print(cf_stability_metric_df)

# ==========================================================
# Counterfactual Sparsity - NB Clean Dataset
# ==========================================================

def compute_cf_sparsity(delta_vector, threshold=1e-6):
    

    delta_vector = np.asarray(delta_vector, dtype=float)

    changed_features = np.sum(np.abs(delta_vector) > threshold)
    total_features = len(delta_vector)

    sparsity_score = 1 - (changed_features / total_features)

    return changed_features, total_features, sparsity_score


changed_features, total_features, sparsity_score = compute_cf_sparsity(delta)

cf_sparsity_df = pd.DataFrame({
    "Model": ["Naive Bayes"],
    "Dataset": ["Clean"],
    "Instance": ["Selected Instance"],
    "Delta Vector": [delta.tolist()],
    "Changed Features": [changed_features],
    "Total Features": [total_features],
    "Sparsity Score": [sparsity_score]
})

print("\nCounterfactual Sparsity Metric Table:")
print(cf_sparsity_df)