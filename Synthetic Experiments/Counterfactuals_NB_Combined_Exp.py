
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# =============================
# Load model + data
# =============================
nb = joblib.load("nb_combined_model.pkl")
X_train = joblib.load("X_train_combined_nb.pkl")
feature_names = joblib.load("feature_names_combined_nb.pkl")
df = pd.read_csv("combined.csv")

X_train = np.asarray(X_train, dtype=np.float64)

# =============================
# Helpers
# =============================
def find_counterfactual_grid(
    model,
    x0,
    target_class,
    step=0.05,
    max_radius=6.0,
    grid_limit=120000
):
    
    x0 = np.asarray(x0, dtype=np.float64).reshape(1, -1)

    best_x = None
    best_dist = np.inf

    radii = np.arange(step, max_radius + 1e-12, step)

    for r in radii:
        vals = np.arange(-r, r + 1e-12, step)

        # safety cap
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

        best_dist = float(dists[j])
        best_x = good[j]
        break  # first radius with a solution is minimal in L2 sense

    return best_x, best_dist


def pick_instances_from_df(df, feature_names):

    df = df.copy()
    df["distance"] = np.abs(df["x1"] - df["x2"]) / np.sqrt(2)
    region_lower = df["region"].astype(str).str.lower()

    clean_df = df[region_lower.str.contains("clean")]
    boundary_df = df[region_lower.str.contains("bound")]
    outlier_df = df[region_lower.str.contains("out")]

    if len(clean_df) == 0:
        raise ValueError("No clean rows found in df['region'].")
    if len(boundary_df) == 0:
        raise ValueError("No boundary rows found in df['region'].")

    clean_row = clean_df.sort_values("distance", ascending=False).iloc[0]
    boundary_row = boundary_df.sort_values("distance", ascending=True).iloc[0]

    # outlier selection
    mean_point = df[feature_names].mean().values.astype(np.float64)
    if len(outlier_df) > 0:
        d = np.linalg.norm(outlier_df[feature_names].values.astype(np.float64) - mean_point.reshape(1, -1), axis=1)
        outlier_row = outlier_df.iloc[int(np.argmax(d))]
    else:
        all_d = np.linalg.norm(df[feature_names].values.astype(np.float64) - mean_point.reshape(1, -1), axis=1)
        outlier_row = df.iloc[int(np.argmax(all_d))]

    return clean_row, boundary_row, outlier_row


def decision_background_like_original(ax, model, df):
    
    x_min, x_max = df["x1"].min() - 1, df["x1"].max() + 1
    y_min, y_max = df["x2"].min() - 1, df["x2"].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(df["x1"], df["x2"], c=df["label"], s=12, alpha=0.6)
    ax.axline((0, 0), slope=1, linestyle="--", color="black", label="True boundary (x1=x2)")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


# =============================
# Pick instances
# =============================
clean_row, boundary_row, outlier_row = pick_instances_from_df(df, feature_names)

x_clean = clean_row[feature_names].values.astype(np.float64)
x_boundary = boundary_row[feature_names].values.astype(np.float64)
x_outlier = outlier_row[feature_names].values.astype(np.float64)

print("Chosen instances:")
print("  CLEAN   :", x_clean, "label:", int(clean_row["label"]), "region:", clean_row["region"])
print("  BOUNDARY:", x_boundary, "label:", int(boundary_row["label"]), "region:", boundary_row["region"])
print("  OUTLIER :", x_outlier, "label:", int(outlier_row["label"]), "region:", outlier_row["region"])


# =============================
# Explain one instance
# =============================
def explain_one(name, x, step=0.05, max_radius=6.0):
    x = np.asarray(x, dtype=np.float64)
    pred = int(nb.predict(x.reshape(1, -1))[0])
    proba = nb.predict_proba(x.reshape(1, -1))[0]
    target = 1 - pred

    x_cf, dist = find_counterfactual_grid(
        model=nb,
        x0=x,
        target_class=target,
        step=step,
        max_radius=max_radius
    )

    if x_cf is None:
        raise RuntimeError(f"[{name}] No counterfactual found. Increase max_radius or reduce step.")

    pred_cf = int(nb.predict(x_cf.reshape(1, -1))[0])
    proba_cf = nb.predict_proba(x_cf.reshape(1, -1))[0]

    delta = x_cf - x

    print(f"\n--- {name} ---")
    print("x:", x, "| pred:", pred, "| proba:", proba)
    print("x_cf:", x_cf, "| pred_cf:", pred_cf, "| proba_cf:", proba_cf)
    print("delta:", delta, "| L2 distance:", dist)

    return {
        "name": name,
        "x": x,
        "pred": pred,
        "proba": proba,
        "x_cf": x_cf,
        "pred_cf": pred_cf,
        "proba_cf": proba_cf,
        "delta": delta,
        "dist": float(dist),
    }


res_clean = explain_one("CLEAN", x_clean, step=0.05, max_radius=8.0)
res_boundary = explain_one("BOUNDARY", x_boundary, step=0.05, max_radius=8.0)
res_outlier = explain_one("OUTLIER", x_outlier, step=0.05, max_radius=10.0)

results = [res_clean, res_boundary, res_outlier]


# =============================
# Decision regions + original + counterfactual (1x3)
# =============================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, res in zip(axes, results):
    decision_background_like_original(ax, nb, df)

    x = res["x"]
    x_cf = res["x_cf"]
    pred = res["pred"]
    pred_cf = res["pred_cf"]

    ax.scatter(x[0], x[1], color="red", marker="X", s=170, edgecolor="black",
               label=f"Original (pred={pred})", zorder=5)
    ax.scatter(x_cf[0], x_cf[1], color="lime", marker="X", s=170, edgecolor="black",
               label=f"Counterfactual (pred={pred_cf})", zorder=6)

    ax.arrow(
        x[0], x[1],
        x_cf[0] - x[0], x_cf[1] - x[1],
        length_includes_head=True,
        head_width=0.35
    )

    ax.set_title(f"NB (Combined) — Counterfactual ({res['name']})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("nb_combined_counterfactual_1x3.png", dpi=300, bbox_inches="tight")
plt.show()
print("\nSaved: nb_combined_counterfactual_1x3.png")


# =============================
# Feature change bar plots 
# =============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, res in zip(axes, results):
    delta = res["delta"]
    ax.bar(feature_names, delta)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Feature Changes (x_cf - x) — {res['name']}")
    ax.set_ylabel("Change required")

plt.tight_layout()
plt.savefig("nb_combined_cf_feature_changes_1x3.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: nb_combined_cf_feature_changes_1x3.png")


# =============================
# Probability shift plots (1x3) [P(Class 1)]
# =============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, res in zip(axes, results):
    p_orig = float(res["proba"][1]) if len(res["proba"]) > 1 else float(res["proba"][0])
    p_cf = float(res["proba_cf"][1]) if len(res["proba_cf"]) > 1 else float(res["proba_cf"][0])

    ax.bar(["Original", "Counterfactual"], [p_orig, p_cf])
    ax.set_ylim(0, 1)
    ax.set_title(f"Probability Shift P(Class 1) — {res['name']}")
    ax.set_ylabel("Probability")

plt.tight_layout()
plt.savefig("nb_combined_cf_probability_shift_1x3.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: nb_combined_cf_probability_shift_1x3.png")


# =============================
# Distance comparison 
# =============================
labels = [r["name"] for r in results]
dists = [r["dist"] for r in results]

plt.figure(figsize=(7, 5))
plt.bar(labels, dists)
plt.title("Counterfactual Distance Comparison (L2)")
plt.ylabel("Minimum change needed to flip")
plt.tight_layout()
plt.savefig("nb_combined_cf_distance_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: nb_combined_cf_distance_comparison.png")


# =============================
# Comparison table
# =============================
rows = []
for r in results:
    rows.append({
        "Instance": r["name"],
        "x1": float(r["x"][0]),
        "x2": float(r["x"][1]),
        "pred": int(r["pred"]),
        "P1_orig": float(r["proba"][1]),
        "x1_cf": float(r["x_cf"][0]),
        "x2_cf": float(r["x_cf"][1]),
        "pred_cf": int(r["pred_cf"]),
        "P1_cf": float(r["proba_cf"][1]),
        "delta_x1": float(r["delta"][0]),
        "delta_x2": float(r["delta"][1]),
        "L2_dist": float(r["dist"]),
    })

table = pd.DataFrame(rows)
print("\n=== Counterfactual Comparison Table (NB Combined) ===")
print(table)

table.to_csv("nb_combined_counterfactual_table.csv", index=False)
print("\nSaved table: nb_combined_counterfactual_table.csv")

print("\nCounterfactual Fidelity Summary (NB Combined):")

for r in results:
    delta_sum = np.sum(r["delta"])
    fidelity_error = abs(float(r["proba"][1]) - float(r["proba_cf"][1]))
    
    print(f"\nCounterfactual Fidelity ({r['name']}):")
    print(f"Original prediction: {float(r['proba'][1]):.6f}")
    print(f"Counterfactual prediction: {float(r['proba_cf'][1]):.6f}")
    print(f"Sum of feature changes (delta): {delta_sum:.6f}")
    print(f"Fidelity error: {fidelity_error:.6f}")

# ==========================================================
# Counterfactual Stability 
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


stability_rows = []

for r in results:
    stability_mean, stability_std, mean_l2, cf_df = compute_cf_stability_fast(
        r,
        n_runs=30
    )

    print(f"\nCounterfactual repeated results - {r['name']}:")
    print(cf_df.head())

    print(f"\nCounterfactual Stability - {r['name']}:")
    print("Mean Cosine Similarity:", stability_mean)
    print("Standard Deviation:", stability_std)
    print("Mean L2 Distance:", mean_l2)

    stability_rows.append({
        "Model": "Naive Bayes",
        "Dataset": "Combined",
        "Instance": r["name"],
        "Mean Cosine Similarity": stability_mean,
        "Standard Deviation": stability_std,
        "Mean L2 Distance": mean_l2
    })


cf_stability_metric_df = pd.DataFrame(stability_rows)

print("\nCounterfactual Stability Metric Table:")
print(cf_stability_metric_df)

cf_stability_metric_df.to_csv(
    "nb_combined_counterfactual_stability.csv",
    index=False
)

print("\nSaved: nb_combined_counterfactual_stability.csv")

# ==========================================================
# Counterfactual Sparsity - NB Combined Dataset
# ==========================================================

def compute_counterfactual_sparsity(delta_vector, tolerance=1e-8):

    delta_vector = np.asarray(delta_vector, dtype=np.float64)

    changed_features = np.sum(np.abs(delta_vector) > tolerance)
    total_features = len(delta_vector)

    sparsity_score = 1.0 - (changed_features / total_features)

    return changed_features, total_features, sparsity_score


sparsity_rows = []

for r in results:
    changed_features, total_features, sparsity_score = compute_counterfactual_sparsity(
        r["delta"]
    )

    sparsity_rows.append({
        "Model": "Naive Bayes",
        "Dataset": "Combined",
        "Instance": r["name"],
        "Delta Vector": r["delta"].tolist(),
        "Changed Features": changed_features,
        "Total Features": total_features,
        "Sparsity Score": sparsity_score
    })


cf_sparsity_df = pd.DataFrame(sparsity_rows)

print("\nCounterfactual Sparsity Metric Table:")
print(cf_sparsity_df)

cf_sparsity_df.to_csv(
    "nb_combined_counterfactual_sparsity.csv",
    index=False
)

print("\nSaved: nb_combined_counterfactual_sparsity.csv")