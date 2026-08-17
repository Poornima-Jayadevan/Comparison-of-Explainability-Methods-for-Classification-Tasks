
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.ioff()
# =============================
# Load model + data
# =============================
rf = joblib.load("rf_boundary_model.pkl")
X_train = joblib.load("X_train_boundary.pkl")
feature_names = joblib.load("feature_names_boundary.pkl")
df = pd.read_csv("clean_plus_tight_boundary.csv")

X_train = np.asarray(X_train, dtype=np.float64)

# =============================
# Pick CLEAN + BOUNDARY instances from df
# =============================
df["distance"] = np.abs(df["x1"] - df["x2"]) / np.sqrt(2)
region_lower = df["region"].astype(str).str.lower()

clean_df = df[region_lower.str.contains("clean")]
boundary_df = df[region_lower.str.contains("bound")]

if len(clean_df) == 0:
    raise ValueError("No clean region rows found in df['region'].")
if len(boundary_df) == 0:
    raise ValueError("No boundary region rows found in df['region'].")

# Clean: far from boundary (within clean region)
clean_row = clean_df.sort_values("distance", ascending=False).iloc[0]
# Boundary: close to boundary (within boundary region)
boundary_row = boundary_df.sort_values("distance", ascending=True).iloc[0]

x_clean = clean_row[feature_names].values.astype(np.float64)
x_boundary = boundary_row[feature_names].values.astype(np.float64)

# =============================
# Counterfactual search 
# =============================
def find_counterfactual_grid(
    model,
    x0,
    target_class,
    step=0.05,
    max_radius=3.0,
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

        # stop at first radius where a solution exists
        break

    return best_x, best_dist

# =============================
# Explain one instance 
# =============================
def explain_one_instance(tag, x0, step=0.05, max_radius=3.0):
    pred = int(rf.predict(x0.reshape(1, -1))[0])
    proba = rf.predict_proba(x0.reshape(1, -1))[0]

    target = 1 - pred
    x_cf, dist = find_counterfactual_grid(
        model=rf,
        x0=x0,
        target_class=target,
        step=step,
        max_radius=max_radius
    )

    if x_cf is None:
        raise RuntimeError(
            f"[{tag}] No counterfactual found. "
            f"Try increasing max_radius (e.g., 6.0) or using smaller step (e.g., 0.02)."
        )

    pred_cf = int(rf.predict(x_cf.reshape(1, -1))[0])
    proba_cf = rf.predict_proba(x_cf.reshape(1, -1))[0]
    delta = x_cf - x0

    print(f"\n================ {tag} ================")
    print("Original x:", x0, "pred:", pred, "proba:", proba)
    print("Counterfactual x_cf:", x_cf, "pred:", pred_cf, "proba:", proba_cf)
    print("Delta (x_cf - x):", delta)
    print("L2 distance:", dist)

    return {
        "tag": tag,
        "x": x0,
        "pred": pred,
        "proba": proba,
        "target": target,
        "x_cf": x_cf,
        "pred_cf": pred_cf,
        "proba_cf": proba_cf,
        "delta": delta,
        "dist": dist
    }

# =============================
# Run CLEAN + BOUNDARY explanations
# =============================

res_clean = explain_one_instance("CLEAN", x_clean, step=0.05, max_radius=6.0)
res_boundary = explain_one_instance("BOUNDARY", x_boundary, step=0.05, max_radius=3.0)

# =============================
# Plot decision regions + original + counterfactual 
# =============================

print("\nDistance Clean:", res_clean["dist"])
print("Distance Boundary:", res_boundary["dist"])

def plot_decision_cf_side_by_side(res_left, res_right, out_png="rf_boundary_counterfactual_side_by_side.png"):

    xx, yy = np.meshgrid(
        np.linspace(-5, 5, 400),
        np.linspace(-5, 5, 400)
    )
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(14, 6))

    # left
    plt.subplot(1, 2, 1)
    x = res_left["x"]; x_cf = res_left["x_cf"]
    pred = res_left["pred"]; pred_cf = res_left["pred_cf"]

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(df["x1"], df["x2"], c=df["label"], s=15, alpha=0.6)
    plt.axline((0, 0), slope=1, color="k", linestyle="--", label="True boundary (x1=x2)")

    plt.scatter(x[0], x[1], color="red", marker="X", s=170, edgecolor="black",
                label=f"Original (pred={pred})", zorder=5)
    plt.scatter(x_cf[0], x_cf[1], color="lime", marker="X", s=170, edgecolor="black",
                label=f"Counterfactual (pred={pred_cf})", zorder=6)

    plt.arrow(x[0], x[1], x_cf[0]-x[0], x_cf[1]-x[1], length_includes_head=True, head_width=0.18)
    plt.title("RF (Clean+Boundary) — Counterfactual (CLEAN)")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend()

    # right
    plt.subplot(1, 2, 2)
    x = res_right["x"]; x_cf = res_right["x_cf"]
    pred = res_right["pred"]; pred_cf = res_right["pred_cf"]

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(df["x1"], df["x2"], c=df["label"], s=15, alpha=0.6)
    plt.axline((0, 0), slope=1, color="k", linestyle="--", label="True boundary (x1=x2)")

    plt.scatter(x[0], x[1], color="red", marker="X", s=170, edgecolor="black",
                label=f"Original (pred={pred})", zorder=5)
    plt.scatter(x_cf[0], x_cf[1], color="lime", marker="X", s=170, edgecolor="black",
                label=f"Counterfactual (pred={pred_cf})", zorder=6)

    plt.arrow(x[0], x[1], x_cf[0]-x[0], x_cf[1]-x[1], length_includes_head=True, head_width=0.18)
    plt.title("RF (Clean+Boundary) — Counterfactual (BOUNDARY)")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"\nSaved: {out_png}")

plot_decision_cf_side_by_side(res_clean, res_boundary)

# =============================
# Feature change bar plot 
# =============================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(feature_names, res_clean["delta"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Counterfactual Feature Changes (CLEAN)")
plt.ylabel("Change required")

plt.subplot(1, 2, 2)
plt.bar(feature_names, res_boundary["delta"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Counterfactual Feature Changes (BOUNDARY)")
plt.ylabel("Change required")

plt.tight_layout()
plt.savefig("rf_boundary_cf_feature_changes_clean_vs_boundary.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: rf_boundary_cf_feature_changes_clean_vs_boundary.png")

# =============================
# Probability shift plot 
# =============================
def p_class1(proba_vec):
    return float(proba_vec[1]) if len(proba_vec) > 1 else float(proba_vec[0])

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(["Original", "Counterfactual"], [p_class1(res_clean["proba"]), p_class1(res_clean["proba_cf"])])
plt.ylim(0, 1)
plt.title("Prediction Probability Shift (CLEAN)")

plt.subplot(1, 2, 2)
plt.bar(["Original", "Counterfactual"], [p_class1(res_boundary["proba"]), p_class1(res_boundary["proba_cf"])])
plt.ylim(0, 1)
plt.title("Prediction Probability Shift (BOUNDARY)")

plt.tight_layout()
plt.savefig("rf_boundary_cf_probability_shift_clean_vs_boundary.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: rf_boundary_cf_probability_shift_clean_vs_boundary.png")

# =============================
# Counterfactual distance comparison 
# =============================
plt.figure(figsize=(6, 5))
plt.bar(["CLEAN", "BOUNDARY"], [res_clean["dist"], res_boundary["dist"]])
plt.title("Counterfactual Distance Comparison (L2)")
plt.ylabel("Minimum change needed to flip")
plt.tight_layout()
plt.savefig("rf_boundary_cf_distance_comparison_clean_vs_boundary.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: rf_boundary_cf_distance_comparison_clean_vs_boundary.png")

# =============================
# Extract info for CLEAN
# =============================
orig_pred_clean = float(res_clean["proba"][res_clean["pred"]])
cf_pred_clean   = float(res_clean["proba_cf"][res_clean["pred"]])
delta_clean     = np.sum(res_clean["delta"])
fidelity_clean  = abs(orig_pred_clean - cf_pred_clean)

print("\n🔍 Counterfactual Fidelity (CLEAN):")
print(f"Original prediction: {orig_pred_clean:.6f}")
print(f"Counterfactual prediction: {cf_pred_clean:.6f}")
print(f"Sum of feature changes (delta): {delta_clean:.6f}")
print(f"Fidelity error: {fidelity_clean:.6f}")

# =============================
# Extract info for BOUNDARY
# =============================
orig_pred_bound = float(res_boundary["proba"][res_boundary["pred"]])
cf_pred_bound   = float(res_boundary["proba_cf"][res_boundary["pred"]])
delta_bound     = np.sum(res_boundary["delta"])
fidelity_bound  = abs(orig_pred_bound - cf_pred_bound)

print("\n🔍 Counterfactual Fidelity (BOUNDARY):")
print(f"Original prediction: {orig_pred_bound:.6f}")
print(f"Counterfactual prediction: {cf_pred_bound:.6f}")
print(f"Sum of feature changes (delta): {delta_bound:.6f}")
print(f"Fidelity error: {fidelity_bound:.6f}")

# ==========================================================
# Counterfactual Stability - RF Boundary Dataset
# ==========================================================

from sklearn.metrics.pairwise import cosine_similarity

def compute_cf_stability_fast(instance_result, n_runs=30):

    delta = instance_result["delta"]
    dist = instance_result["dist"]

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
    res_boundary,
    n_runs=30
)

print("\nCounterfactual repeated results - CLEAN:")
print(cf_clean_df.head())

print("\nCounterfactual repeated results - BOUNDARY:")
print(cf_boundary_df.head())

print("\nCounterfactual Stability - RF Boundary Dataset:")
print("Clean-region Mean Cosine Similarity:", clean_stability_mean)
print("Clean-region Standard Deviation:", clean_stability_std)
print("Clean-region Mean L2 Distance:", clean_mean_l2)

print("Tight-boundary Mean Cosine Similarity:", boundary_stability_mean)
print("Tight-boundary Standard Deviation:", boundary_stability_std)
print("Tight-boundary Mean L2 Distance:", boundary_mean_l2)

cf_stability_metric_df = pd.DataFrame({
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
    ],
    "Mean L2 Distance": [
        clean_mean_l2,
        boundary_mean_l2
    ]
})

print("\nCounterfactual Stability Metric Table:")
print(cf_stability_metric_df)

# ==========================================================
# Counterfactual SPARSITY - RF Boundary Dataset
# ==========================================================

def compute_cf_sparsity(delta_vector, threshold=1e-6):

    delta_vector = np.asarray(delta_vector)

    changed_features = np.sum(np.abs(delta_vector) > threshold)
    total_features = len(delta_vector)

    sparsity_score = 1 - (changed_features / total_features)

    return changed_features, total_features, sparsity_score


# -------------------------------
# CLEAN sparsity
# -------------------------------
clean_changed, clean_total, clean_sparsity = compute_cf_sparsity(
    res_clean["delta"]
)

# -------------------------------
# BOUNDARY sparsity
# -------------------------------
boundary_changed, boundary_total, boundary_sparsity = compute_cf_sparsity(
    res_boundary["delta"]
)

# -------------------------------
# Create sparsity table
# -------------------------------
cf_sparsity_df = pd.DataFrame({
    "Model": ["Random Forest", "Random Forest"],
    "Dataset": ["Boundary", "Boundary"],
    "Instance": ["Clean-region", "Tight-boundary"],
    "Delta Vector": [
        res_clean["delta"].tolist(),
        res_boundary["delta"].tolist()
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