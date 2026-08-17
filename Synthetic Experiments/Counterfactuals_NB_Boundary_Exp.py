
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# =============================
# Load model + data
# =============================
nb = joblib.load("nb_boundary_model.pkl")
X_train = joblib.load("X_train_boundary_nb.pkl")
feature_names = joblib.load("feature_names_boundary_nb.pkl")
df = pd.read_csv("clean_plus_tight_boundary.csv")

X_train = np.asarray(X_train, dtype=np.float64)

# =============================
# Pick CLEAN vs BOUNDARY instances from df (using df["region"])
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

x_clean = clean_row[feature_names].values.astype(np.float64)
x_bound = bound_row[feature_names].values.astype(np.float64)

print("CLEAN instance:", x_clean, "label:", int(clean_row["label"]))
print("BOUNDARY instance:", x_bound, "label:", int(bound_row["label"]))

# =============================
# Counterfactual search 
# =============================
def _grid_candidates_2d(center, step, radius, grid_limit=200000):
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
    grid_limit=200000
):
    x0 = np.asarray(x0, dtype=np.float64).reshape(1, -1)

    best_x = None
    best_dist = np.inf

    
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

def explain_one_instance(tag, x0):
    pred0 = int(nb.predict(x0.reshape(1, -1))[0])
    proba0 = nb.predict_proba(x0.reshape(1, -1))[0]
    target = 1 - pred0

    x_cf, dist = find_counterfactual_multistage(
        model=nb,
        x0=x0,
        target_class=target,
        step1=0.20,
        max_radius1=12.0,  
        step2=0.05,
        radius2=1.5
    )

    if x_cf is None:
        raise RuntimeError(
            f"[{tag}] No counterfactual found. "
            f"Try increasing max_radius1 (e.g., 16.0) or using smaller step2 (e.g., 0.02)."
        )

    pred_cf = int(nb.predict(x_cf.reshape(1, -1))[0])
    proba_cf = nb.predict_proba(x_cf.reshape(1, -1))[0]
    delta = x_cf - x0

    print(f"\n--- {tag} ---")
    print("x:", x0, "pred:", pred0, "proba:", proba0)
    print("x_cf:", x_cf, "pred_cf:", pred_cf, "proba_cf:", proba_cf)
    print("delta:", delta, "L2:", dist)

    return {
        "tag": tag,
        "x": x0,
        "pred": pred0,
        "proba": proba0,
        "x_cf": x_cf,
        "pred_cf": pred_cf,
        "proba_cf": proba_cf,
        "delta": delta,
        "dist": float(dist),
    }

res_clean = explain_one_instance("CLEAN", x_clean)
res_bound = explain_one_instance("BOUNDARY", x_bound)

# =============================
# Decision region grid 
# =============================
xx, yy = np.meshgrid(
    np.linspace(-6, 6.5, 400),
    np.linspace(-6, 6.5, 400)
)
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# =============================
# decision regions + counterfactuals 
# =============================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, res in zip(axes, [res_clean, res_bound]):
    x0 = res["x"]
    xcf = res["x_cf"]
    pred0 = res["pred"]
    pred_cf = res["pred_cf"]
    tag = res["tag"]

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(df["x1"], df["x2"], c=df["label"], s=15, alpha=0.6)

    # True boundary
    ax.axline((0, 0), slope=1, color="k", linestyle="--", label="True boundary (x1=x2)")

    # Original + Counterfactual
    ax.scatter(x0[0], x0[1], color="red", marker="X", s=170, edgecolor="black",
               label=f"Original (pred={pred0})", zorder=5)
    ax.scatter(xcf[0], xcf[1], color="lime", marker="X", s=170, edgecolor="black",
               label=f"Counterfactual (pred={pred_cf})", zorder=6)

    # Arrow from original to counterfactual
    ax.arrow(
        x0[0], x0[1],
        xcf[0] - x0[0], xcf[1] - x0[1],
        length_includes_head=True,
        head_width=0.18
    )

    ax.set_title(f"NB (Clean+Boundary) — Counterfactual ({tag})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

plt.tight_layout()
plt.savefig("nb_boundary_counterfactuals_clean_vs_boundary.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: nb_boundary_counterfactuals_clean_vs_boundary.png")

# =============================
# Feature change bar plots (side-by-side)
# =============================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, res in zip(axes, [res_clean, res_bound]):
    delta = res["delta"]
    tag = res["tag"]
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(feature_names, delta)
    ax.set_title(f"Feature Changes (x_cf - x) — {tag}")
    ax.set_ylabel("Change required")

plt.tight_layout()
plt.savefig("nb_boundary_cf_feature_changes_clean_vs_boundary.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: nb_boundary_cf_feature_changes_clean_vs_boundary.png")

# =============================
# Probability shift plots (side-by-side) (P(Class 1))
# =============================
def p_class1(prob):
    return float(prob[1]) if len(prob) > 1 else float(prob[0])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, res in zip(axes, [res_clean, res_bound]):
    tag = res["tag"]
    p0 = p_class1(res["proba"])
    p1 = p_class1(res["proba_cf"])
    ax.set_ylim(0, 1)
    ax.bar(["Original", "Counterfactual"], [p0, p1])
    ax.set_title(f"Probability Shift P(Class 1) — {tag}")
    ax.set_ylabel("Probability")

plt.tight_layout()
plt.savefig("nb_boundary_cf_probability_shift_clean_vs_boundary.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: nb_boundary_cf_probability_shift_clean_vs_boundary.png")

# =============================
# Counterfactual distance comparison 
# =============================
plt.figure(figsize=(6, 4))
plt.bar(["CLEAN", "BOUNDARY"], [res_clean["dist"], res_bound["dist"]])
plt.title("Counterfactual Distance Comparison (L2)")
plt.ylabel("Minimum change needed to flip")
plt.tight_layout()
plt.savefig("nb_boundary_cf_distance_comparison_clean_vs_boundary.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: nb_boundary_cf_distance_comparison_clean_vs_boundary.png")

# =============================
# Print + save comparison table
# =============================
table = pd.DataFrame({
    "Case": ["CLEAN", "BOUNDARY"],
    "x1": [res_clean["x"][0], res_bound["x"][0]],
    "x2": [res_clean["x"][1], res_bound["x"][1]],
    "pred": [res_clean["pred"], res_bound["pred"]],
    "p(class1)_orig": [p_class1(res_clean["proba"]), p_class1(res_bound["proba"])],
    "x1_cf": [res_clean["x_cf"][0], res_bound["x_cf"][0]],
    "x2_cf": [res_clean["x_cf"][1], res_bound["x_cf"][1]],
    "pred_cf": [res_clean["pred_cf"], res_bound["pred_cf"]],
    "p(class1)_cf": [p_class1(res_clean["proba_cf"]), p_class1(res_bound["proba_cf"])],
    "dx1": [res_clean["delta"][0], res_bound["delta"][0]],
    "dx2": [res_clean["delta"][1], res_bound["delta"][1]],
    "L2_dist": [res_clean["dist"], res_bound["dist"]],
})

print("\n=== Counterfactual Comparison Table (NB: Clean vs Boundary) ===")
print(table)

table.to_csv("nb_boundary_counterfactuals_clean_vs_boundary.csv", index=False)
print("Saved table: nb_boundary_counterfactuals_clean_vs_boundary.csv")

# =============================
# Counterfactual fidelity
# =============================
f_clean = abs(p_class1(res_clean["proba"]) - p_class1(res_clean["proba_cf"]))
f_bound = abs(p_class1(res_bound["proba"]) - p_class1(res_bound["proba_cf"]))

print("\nCounterfactual Fidelity (CLEAN):")
print(f"Original prediction: {p_class1(res_clean['proba']):.6f}")
print(f"Counterfactual prediction: {p_class1(res_clean['proba_cf']):.6f}")
print(f"Sum of feature changes (delta): {res_clean['delta'].sum():+.6f}")
print(f"Fidelity error: {f_clean:.6f}")

print("\nCounterfactual Fidelity (BOUNDARY):")
print(f"Original prediction: {p_class1(res_bound['proba']):.6f}")
print(f"Counterfactual prediction: {p_class1(res_bound['proba_cf']):.6f}")
print(f"Sum of feature changes (delta): {res_bound['delta'].sum():+.6f}")
print(f"Fidelity error: {f_bound:.6f}")

# ==========================================================
# 11) Counterfactual Stability - NB Boundary Dataset
# ==========================================================

from sklearn.metrics.pairwise import cosine_similarity

def compute_cf_stability_fast(result, n_runs=30):
    

    delta = result["delta"]
    dist = result["dist"]

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


clean_stability_mean, clean_stability_std, clean_mean_l2, cf_clean_df = compute_cf_stability_fast(
    res_clean,
    n_runs=30
)

boundary_stability_mean, boundary_stability_std, boundary_mean_l2, cf_boundary_df = compute_cf_stability_fast(
    res_bound,
    n_runs=30
)

print("\nCounterfactual repeated results - CLEAN:")
print(cf_clean_df.head())

print("\nCounterfactual repeated results - BOUNDARY:")
print(cf_boundary_df.head())

print("\nCounterfactual Stability - NB Boundary Dataset:")
print("Clean-region Mean Cosine Similarity:", clean_stability_mean)
print("Clean-region Standard Deviation:", clean_stability_std)
print("Clean-region Mean L2 Distance:", clean_mean_l2)

print("Tight-boundary Mean Cosine Similarity:", boundary_stability_mean)
print("Tight-boundary Standard Deviation:", boundary_stability_std)
print("Tight-boundary Mean L2 Distance:", boundary_mean_l2)


cf_stability_metric_df = pd.DataFrame({
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
    ],
    "Mean L2 Distance": [
        clean_mean_l2,
        boundary_mean_l2
    ]
})

print("\nCounterfactual Stability Metric Table:")
print(cf_stability_metric_df)

# ==========================================================
# Counterfactual Sparsity
# ==========================================================

def compute_counterfactual_sparsity(delta_vector, tolerance=1e-8):

    delta_vector = np.asarray(delta_vector, dtype=np.float64)

    changed_features = np.sum(np.abs(delta_vector) > tolerance)
    total_features = len(delta_vector)

    sparsity_score = 1.0 - (changed_features / total_features)

    return changed_features, total_features, sparsity_score


# CLEAN instance sparsity
clean_changed, clean_total, clean_sparsity = compute_counterfactual_sparsity(
    res_clean["delta"]
)

# BOUNDARY instance sparsity
boundary_changed, boundary_total, boundary_sparsity = compute_counterfactual_sparsity(
    res_bound["delta"]
)

# -----------------------------
# Create Sparsity Table
# -----------------------------
cf_sparsity_df = pd.DataFrame({
    "Model": ["Naive Bayes", "Naive Bayes"],
    "Dataset": ["Boundary", "Boundary"],
    "Instance": ["Clean-region", "Tight-boundary"],
    "Delta Vector": [
        list(res_clean["delta"]),
        list(res_bound["delta"])
    ],
    "Changed Features": [
        clean_changed,
        boundary_changed
    ],
    "Total Features": [
        clean_total,
        boundary_total
    ],
    "Sparsity Score": [
        clean_sparsity,
        boundary_sparsity
    ]
})

print("\nCounterfactual Sparsity Metric Table:")
print(cf_sparsity_df)

cf_sparsity_df.to_csv(
    "nb_boundary_counterfactual_sparsity.csv",
    index=False
)

print("\nSaved: nb_boundary_counterfactual_sparsity.csv")