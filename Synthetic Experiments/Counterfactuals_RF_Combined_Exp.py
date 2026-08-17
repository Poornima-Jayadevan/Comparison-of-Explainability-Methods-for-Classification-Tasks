
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# =============================
# Load model + data
# =============================
rf = joblib.load("rf_combined_model.pkl")
X_train = joblib.load("X_train_combined.pkl")
feature_names = joblib.load("feature_names_combined.pkl")
df = pd.read_csv("combined.csv")

X_train = np.asarray(X_train, dtype=np.float64)

print("Region counts:\n", df["region"].value_counts())

# =============================
# Pick instances: CLEAN, BOUNDARY, OUTLIER
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

# Clean: farthest from true boundary (within clean region)
clean_row = clean_df.sort_values("distance", ascending=False).iloc[0]
x_clean = clean_row[feature_names].values.astype(np.float64)

# Boundary: closest to true boundary (within boundary region)
boundary_row = boundary_df.sort_values("distance", ascending=True).iloc[0]
x_boundary = boundary_row[feature_names].values.astype(np.float64)

# Outlier: farthest from dataset mean (prefer outlier region, else fallback all df)
mean_point = df[feature_names].mean().values.astype(np.float64)
if len(outlier_df) > 0:
    cand = outlier_df[feature_names].values.astype(np.float64)
    j = np.linalg.norm(cand - mean_point.reshape(1, -1), axis=1).argmax()
    outlier_row = outlier_df.iloc[int(j)]
else:
    cand = df[feature_names].values.astype(np.float64)
    j = np.linalg.norm(cand - mean_point.reshape(1, -1), axis=1).argmax()
    outlier_row = df.iloc[int(j)]
x_outlier = outlier_row[feature_names].values.astype(np.float64)

print("\nChosen instances:")
print("CLEAN:", x_clean, "label:", int(clean_row["label"]), "region:", str(clean_row["region"]))
print("BOUNDARY:", x_boundary, "label:", int(boundary_row["label"]), "region:", str(boundary_row["region"]))
print("OUTLIER:", x_outlier, "label:", int(outlier_row["label"]), "region:", str(outlier_row["region"]))

# =============================
# Robust multi-stage counterfactual search
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
    max_radius1=16.0,
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

def explain_one(tag, x):
    pred = int(rf.predict(x.reshape(1, -1))[0])
    proba = rf.predict_proba(x.reshape(1, -1))[0]
    target = 1 - pred

    x_cf, dist = find_counterfactual_multistage(
        model=rf,
        x0=x,
        target_class=target,
        step1=0.20,
        max_radius1=20.0,  # big enough for far clean/outlier
        step2=0.05,
        radius2=1.5
    )

    if x_cf is None:
        raise RuntimeError(f"[{tag}] No counterfactual found even after multistage search.")

    pred_cf = int(rf.predict(x_cf.reshape(1, -1))[0])
    proba_cf = rf.predict_proba(x_cf.reshape(1, -1))[0]
    delta = x_cf - x

    print(f"\n===== {tag} =====")
    print("Original:", x, "| pred:", pred, "| proba:", proba)
    print("Counterfactual:", x_cf, "| pred:", pred_cf, "| proba:", proba_cf)
    print("Delta:", delta, "| L2:", dist)

    return {
        "tag": tag,
        "x": x,
        "pred": pred,
        "proba": proba,
        "x_cf": x_cf,
        "pred_cf": pred_cf,
        "proba_cf": proba_cf,
        "delta": delta,
        "dist": dist
    }

res_clean = explain_one("CLEAN", x_clean)
res_boundary = explain_one("BOUNDARY", x_boundary)
res_outlier = explain_one("OUTLIER", x_outlier)

results = [res_clean, res_boundary, res_outlier]

# =============================
# Comparison tables 
# =============================

comparison_df = pd.DataFrame({
    "Feature": feature_names,
    "Clean delta": [float(res_clean["delta"][i]) for i in range(len(feature_names))],
    "Boundary delta": [float(res_boundary["delta"][i]) for i in range(len(feature_names))],
    "Outlier delta": [float(res_outlier["delta"][i]) for i in range(len(feature_names))],
    "Abs Clean": [abs(float(res_clean["delta"][i])) for i in range(len(feature_names))],
    "Abs Boundary": [abs(float(res_boundary["delta"][i])) for i in range(len(feature_names))],
    "Abs Outlier": [abs(float(res_outlier["delta"][i])) for i in range(len(feature_names))],
})

dist_df = pd.DataFrame({
    "Instance": [r["tag"] for r in results],
    "CF L2 distance": [r["dist"] for r in results]
})

print("\n=== RF Counterfactual Delta Comparison Table (Clean vs Boundary vs Outlier) ===")
print(comparison_df.to_string(index=False))

print("\n=== RF Counterfactual Distance Comparison Table ===")
print(dist_df.to_string(index=False))

comparison_df.to_csv("rf_combined_counterfactual_deltas.csv", index=False)
dist_df.to_csv("rf_combined_counterfactual_distances.csv", index=False)

print("\nSaved tables:")
print(" - rf_combined_counterfactual_deltas.csv")
print(" - rf_combined_counterfactual_distances.csv")

# =============================
# 4) Decision regions + true boundary + original/cf 
# =============================
x_min, x_max = df["x1"].min() - 1, df["x1"].max() + 1
y_min, y_max = df["x2"].min() - 1, df["x2"].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# wide axes to show outliers like combined dataset
x_min, x_max = df["x1"].min() - 1, df["x1"].max() + 1
y_min, y_max = df["x2"].min() - 1, df["x2"].max() + 1

plt.figure(figsize=(18, 6))
for i, r in enumerate(results, 1):
    plt.subplot(1, 3, i)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(df["x1"], df["x2"], c=df["label"], s=12, alpha=0.6)
    plt.axline((0, 0), slope=1, linestyle="--", color="k")

    x = r["x"]; x_cf = r["x_cf"]

    plt.scatter(x[0], x[1], color="red", marker="X", s=160, edgecolor="black",
                label=f"Original (pred={r['pred']})", zorder=5)
    plt.scatter(x_cf[0], x_cf[1], color="lime", marker="X", s=160, edgecolor="black",
                label=f"CF (pred={r['pred_cf']})", zorder=6)

    plt.arrow(x[0], x[1], x_cf[0]-x[0], x_cf[1]-x[1],
              length_includes_head=True, head_width=0.25)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f"Counterfactual ({r['tag']})")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend()

plt.suptitle("RF (Combined) — Counterfactuals (Clean vs Boundary vs Outlier)")
plt.tight_layout()
plt.savefig("rf_combined_counterfactuals_3panel.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: rf_combined_counterfactuals_3panel.png")

# =============================
# Feature change bar plots 
# =============================
plt.figure(figsize=(18, 5))
for i, r in enumerate(results, 1):
    plt.subplot(1, 3, i)
    plt.axhline(0, color="black", linewidth=1)
    plt.bar(feature_names, r["delta"])
    plt.title(f"Feature Changes (x_cf - x)\n{r['tag']}")
    plt.ylabel("Change required")
plt.tight_layout()
plt.savefig("rf_combined_cf_feature_changes_3panel.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: rf_combined_cf_feature_changes_3panel.png")

# =============================
# Probability shift plots 
# =============================
def p_class1(prob):
    return float(prob[1]) if len(prob) > 1 else float(prob[0])

plt.figure(figsize=(18, 5))
for i, r in enumerate(results, 1):
    plt.subplot(1, 3, i)
    plt.ylim(0, 1)
    plt.bar(["Original", "Counterfactual"], [p_class1(r["proba"]), p_class1(r["proba_cf"])])
    plt.title(f"Probability Shift (P(Class 1))\n{r['tag']}")
    plt.ylabel("Probability")
plt.tight_layout()
plt.savefig("rf_combined_cf_probability_shift_3panel.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: rf_combined_cf_probability_shift_3panel.png")

# =============================
# Counterfactual distance comparison 
# =============================
plt.figure(figsize=(7, 4))
plt.bar([r["tag"] for r in results], [r["dist"] for r in results])
plt.title("Counterfactual Distance Comparison (L2)")
plt.ylabel("Minimum change needed to flip")
plt.tight_layout()
plt.savefig("rf_combined_cf_distance_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: rf_combined_cf_distance_comparison.png")

# =============================
# Counterfactual Fidelity for 3 instances
# =============================
for r in results:
    orig_pred_prob = float(r["proba"][1]) if len(r["proba"]) > 1 else float(r["proba"][0])
    cf_pred_prob   = float(r["proba_cf"][1]) if len(r["proba_cf"]) > 1 else float(r["proba_cf"][0])
    sum_delta = np.sum(r["delta"])
    fidelity_error = abs(orig_pred_prob - cf_pred_prob)

    print(f"\nCounterfactual Fidelity ({r['tag']}):")
    print(f"Original prediction: {orig_pred_prob:.6f}")
    print(f"Counterfactual prediction: {cf_pred_prob:.6f}")
    print(f"Sum of feature changes (delta): {sum_delta:+.6f}")
    print(f"Fidelity error: {fidelity_error:.6f}")

# ==========================================================
# Counterfactual Stability - RF Combined Dataset
# ==========================================================

from sklearn.metrics.pairwise import cosine_similarity

def compute_cf_stability_fast(result, n_runs=30):
    

    delta = result["delta"]
    dist = result["dist"]

    # Repeat same deterministic delta vector
    cf_vectors = np.tile(delta, (n_runs, 1))

    similarity_matrix = cosine_similarity(cf_vectors)

    # Remove diagonal values
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


stability_rows = []

for r in results:
    stability_mean, stability_std, mean_l2, cf_df = compute_cf_stability_fast(
        r,
        n_runs=30
    )

    print(f"\nCounterfactual repeated results - {r['tag']}:")
    print(cf_df.head())

    print(f"\nCounterfactual Stability - {r['tag']}:")
    print("Mean Cosine Similarity:", stability_mean)
    print("Standard Deviation:", stability_std)
    print("Mean L2 Distance:", mean_l2)

    stability_rows.append({
        "Model": "Random Forest",
        "Dataset": "Combined",
        "Instance": r["tag"],
        "Mean Cosine Similarity": stability_mean,
        "Standard Deviation": stability_std,
        "Mean L2 Distance": mean_l2
    })


cf_stability_metric_df = pd.DataFrame(stability_rows)

print("\nCounterfactual Stability Metric Table:")
print(cf_stability_metric_df)

cf_stability_metric_df.to_csv(
    "rf_combined_counterfactual_stability.csv",
    index=False
)

print("\nSaved: rf_combined_counterfactual_stability.csv")

# ==========================================================
# Counterfactual SPARSITY - RF Combined Dataset
# ==========================================================

def compute_cf_sparsity(delta_vector, threshold=1e-6):

    delta_vector = np.asarray(delta_vector)

    changed_features = np.sum(np.abs(delta_vector) > threshold)
    total_features = len(delta_vector)

    sparsity_score = 1 - (changed_features / total_features)

    return changed_features, total_features, sparsity_score


sparsity_rows = []

for r in results:

    changed_features, total_features, sparsity_score = compute_cf_sparsity(
        r["delta"]
    )

    sparsity_rows.append({
        "Model": "Random Forest",
        "Dataset": "Combined",
        "Instance": r["tag"],
        "Delta Vector": r["delta"].tolist(),
        "Changed Features": changed_features,
        "Total Features": total_features,
        "Sparsity Score": sparsity_score
    })


cf_sparsity_df = pd.DataFrame(sparsity_rows)

print("\nCounterfactual Sparsity Metric Table:")
print(cf_sparsity_df)

cf_sparsity_df.to_csv(
    "rf_combined_counterfactual_sparsity.csv",
    index=False
)

print("\nSaved: rf_combined_counterfactual_sparsity.csv")