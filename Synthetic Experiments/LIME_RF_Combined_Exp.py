# ===============================================
# LIME Explanation with Arrows: Clean vs Boundary vs Outlier
# ===============================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import joblib
from lime.lime_tabular import LimeTabularExplainer

matplotlib.rcParams["figure.max_open_warning"] = 50

# =============================
# 1. Load saved model & training data
# =============================
rf = joblib.load("rf_combined_model.pkl")
X_train = joblib.load("X_train_combined.pkl")
feature_names = joblib.load("feature_names_combined.pkl")

# Load dataset for instance selection and visualization
df = pd.read_csv("combined.csv")

# =============================
# 2. Compute distance from boundary (x1 = x2)
# =============================
df["distance"] = np.abs(df["x1"] - df["x2"]) / np.sqrt(2)

# Clean-region instance (largest distance)
clean_point = df.sort_values("distance", ascending=False).iloc[0]
clean_instance = clean_point[feature_names].values.astype(float)

# Tight-boundary instance (smallest distance)
boundary_point = df.sort_values("distance", ascending=True).iloc[0]
boundary_instance = boundary_point[feature_names].values.astype(float)

# Outlier instance (farthest from dataset mean)
mean_point = df[feature_names].mean().values
outlier_idx = np.linalg.norm(df[feature_names].values - mean_point, axis=1).argmax()
outlier_point = df.iloc[outlier_idx]
outlier_instance = outlier_point[feature_names].values.astype(float)

print("Clean-region instance:", clean_instance, "Label:", clean_point["label"])
print("Tight-boundary instance:", boundary_instance, "Label:", boundary_point["label"])
print("Outlier instance:", outlier_instance, "Label:", outlier_point["label"])

print("\nModel predictions:")
print(
    "Clean prediction:", rf.predict(clean_instance.reshape(1, -1))[0],
    "Proba:", rf.predict_proba(clean_instance.reshape(1, -1))[0]
)
print(
    "Boundary prediction:", rf.predict(boundary_instance.reshape(1, -1))[0],
    "Proba:", rf.predict_proba(boundary_instance.reshape(1, -1))[0]
)
print(
    "Outlier prediction:", rf.predict(outlier_instance.reshape(1, -1))[0],
    "Proba:", rf.predict_proba(outlier_instance.reshape(1, -1))[0]
)

# =============================
# 3. Initialize LIME Explainer
# =============================
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=["Class 0", "Class 1"],
    mode="classification",
    discretize_continuous=False
)

# =============================
# 4. Generate LIME explanations
# =============================
exp_clean = explainer.explain_instance(
    clean_instance,
    rf.predict_proba,
    num_features=2
)
exp_boundary = explainer.explain_instance(
    boundary_instance,
    rf.predict_proba,
    num_features=2
)
exp_outlier = explainer.explain_instance(
    outlier_instance,
    rf.predict_proba,
    num_features=2
)

weights_clean = dict(exp_clean.as_list())
weights_boundary = dict(exp_boundary.as_list())
weights_outlier = dict(exp_outlier.as_list())

print("\nLIME Weights - Clean:", weights_clean)
print("LIME Weights - Boundary:", weights_boundary)
print("LIME Weights - Outlier:", weights_outlier)

# =============================
# 5. SIDE-BY-SIDE LIME FEATURE IMPORTANCE
# =============================
features = feature_names

clean_values = [weights_clean.get(f, 0) for f in features]
boundary_values = [weights_boundary.get(f, 0) for f in features]
outlier_values = [weights_outlier.get(f, 0) for f in features]

all_bar_values = clean_values + boundary_values + outlier_values
x_bar_min = min(all_bar_values) - 0.05
x_bar_max = max(all_bar_values) + 0.05

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Clean
colors_clean = ["green" if v > 0 else "red" for v in clean_values]
axes[0].barh(features, clean_values, color=colors_clean)
axes[0].axvline(0, color="black")
axes[0].set_xlim(x_bar_min, x_bar_max)
axes[0].set_title("LIME Feature Importance\nClean-region Instance")
axes[0].set_xlabel("LIME Weight")

# Boundary
colors_boundary = ["green" if v > 0 else "red" for v in boundary_values]
axes[1].barh(features, boundary_values, color=colors_boundary)
axes[1].axvline(0, color="black")
axes[1].set_xlim(x_bar_min, x_bar_max)
axes[1].set_title("LIME Feature Importance\nTight-boundary Instance")
axes[1].set_xlabel("LIME Weight")

# Outlier
colors_outlier = ["green" if v > 0 else "red" for v in outlier_values]
axes[2].barh(features, outlier_values, color=colors_outlier)
axes[2].axvline(0, color="black")
axes[2].set_xlim(x_bar_min, x_bar_max)
axes[2].set_title("LIME Feature Importance\nOutlier Instance")
axes[2].set_xlabel("LIME Weight")

plt.tight_layout()
plt.show()
plt.close()

# =============================
# 6. Normalize arrows for visualization
# =============================
all_weights = np.array(
    list(weights_clean.values()) +
    list(weights_boundary.values()) +
    list(weights_outlier.values())
)

max_weight = np.max(np.abs(all_weights)) if len(all_weights) > 0 else 1.0

def normalized_arrow(weight, scale=2.0, min_len=0.2):
    if max_weight == 0:
        return 0.0
    scaled = scale * weight / max_weight
    if weight != 0 and abs(scaled) < min_len:
        scaled = np.sign(weight) * min_len
    return scaled

# =============================
# 7. Decision boundary grid
# =============================
padding = 1.0
x1_min, x1_max = df["x1"].min() - padding, df["x1"].max() + padding
x2_min, x2_max = df["x2"].min() - padding, df["x2"].max() + padding

xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 500),
    np.linspace(x2_min, x2_max, 500)
)

Z = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# =============================
# 8. Side-by-side LIME plots
# =============================
plt.figure(figsize=(18, 5))

instances = [
    ("Clean-region", clean_instance, weights_clean, "blue"),
    ("Tight-boundary", boundary_instance, weights_boundary, "green"),
    ("Outlier", outlier_instance, weights_outlier, "purple")
]

for i, (title, instance, weights, color) in enumerate(instances, 1):
    plt.subplot(1, 3, i)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(df["x1"], df["x2"], c=df["label"], s=12, alpha=0.6)
    plt.axline((0, 0), slope=1, linestyle="--", color="black")

    plt.scatter(
        instance[0], instance[1],
        color="red",
        marker="X",
        s=150,
        edgecolor="black",
        label="Explained Instance",
        zorder=5
    )

    x_arrow = normalized_arrow(weights.get("x1", 0))
    y_arrow = normalized_arrow(weights.get("x2", 0))

    plt.arrow(
        instance[0], instance[1], x_arrow, 0,
        head_width=0.35,
        head_length=0.35,
        color=color,
        length_includes_head=True,
        zorder=6
    )

    plt.arrow(
        instance[0], instance[1], 0, y_arrow,
        head_width=0.35,
        head_length=0.35,
        color=color,
        length_includes_head=True,
        zorder=6
    )

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(f"{title} Instance")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper left")

plt.suptitle("LIME Explanation Comparison: Clean vs Boundary vs Outlier")
plt.tight_layout()
plt.show()
plt.close()

# =============================
# 9. LIME weights comparison table
# =============================
comparison_df = pd.DataFrame({
    "Feature": feature_names,
    "Clean-region": [weights_clean.get(f, 0) for f in feature_names],
    "Tight-boundary": [weights_boundary.get(f, 0) for f in feature_names],
    "Outlier": [weights_outlier.get(f, 0) for f in feature_names]
})

print("\nLIME Weight Comparison Table:")
print(comparison_df)

# =============================
# 10. LIME weight comparison grouped bar chart
# =============================
plot_df = comparison_df.set_index("Feature")
plot_df.plot(kind="bar", figsize=(8, 5))
plt.title("LIME Weight Comparison: Clean vs Boundary vs Outlier")
plt.ylabel("LIME Weight")
plt.axhline(0, color="black", linewidth=1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
plt.close()

# =============================
# 11. LIME stability analysis
# =============================
n_runs = 30
stability_clean = []
stability_boundary = []
stability_outlier = []

for run in range(n_runs):
    exp_c = explainer.explain_instance(
        clean_instance,
        rf.predict_proba,
        num_features=2
    )
    wc = dict(exp_c.as_list())
    stability_clean.append({
        "run": run + 1,
        "x1_weight": wc.get("x1", 0),
        "x2_weight": wc.get("x2", 0)
    })

    exp_b = explainer.explain_instance(
        boundary_instance,
        rf.predict_proba,
        num_features=2
    )
    wb = dict(exp_b.as_list())
    stability_boundary.append({
        "run": run + 1,
        "x1_weight": wb.get("x1", 0),
        "x2_weight": wb.get("x2", 0)
    })

    exp_o = explainer.explain_instance(
        outlier_instance,
        rf.predict_proba,
        num_features=2
    )
    wo = dict(exp_o.as_list())
    stability_outlier.append({
        "run": run + 1,
        "x1_weight": wo.get("x1", 0),
        "x2_weight": wo.get("x2", 0)
    })

stability_clean_df = pd.DataFrame(stability_clean)
stability_boundary_df = pd.DataFrame(stability_boundary)
stability_outlier_df = pd.DataFrame(stability_outlier)

print("\nStability results (clean instance):")
print(stability_clean_df.head())

print("\nStability results (boundary instance):")
print(stability_boundary_df.head())

print("\nStability results (outlier instance):")
print(stability_outlier_df.head())

# =============================
# 12. Stability boxplot
# =============================
plt.figure(figsize=(10, 5))
plt.boxplot(
    [
        stability_clean_df["x1_weight"],
        stability_clean_df["x2_weight"],
        stability_boundary_df["x1_weight"],
        stability_boundary_df["x2_weight"],
        stability_outlier_df["x1_weight"],
        stability_outlier_df["x2_weight"]
    ],
    tick_labels=[
        "Clean x1", "Clean x2",
        "Boundary x1", "Boundary x2",
        "Outlier x1", "Outlier x2"
    ]
)
plt.axhline(0, color="black", linewidth=1)
plt.ylabel("LIME Weight")
plt.title("LIME Stability Across Repeated Runs")
plt.tight_layout()
plt.show()
plt.close()

# =============================
# 13. Stability line plots
# =============================
plt.figure(figsize=(11, 5))

plt.plot(
    stability_clean_df["run"],
    stability_clean_df["x1_weight"],
    marker="o",
    label="Clean x1"
)
plt.plot(
    stability_clean_df["run"],
    stability_clean_df["x2_weight"],
    marker="s",
    label="Clean x2"
)

plt.plot(
    stability_boundary_df["run"],
    stability_boundary_df["x1_weight"],
    marker="^",
    label="Boundary x1"
)
plt.plot(
    stability_boundary_df["run"],
    stability_boundary_df["x2_weight"],
    marker="d",
    label="Boundary x2"
)

plt.plot(
    stability_outlier_df["run"],
    stability_outlier_df["x1_weight"],
    marker="x",
    label="Outlier x1"
)
plt.plot(
    stability_outlier_df["run"],
    stability_outlier_df["x2_weight"],
    marker="P",
    label="Outlier x2"
)

plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Run")
plt.ylabel("LIME Weight")
plt.title("LIME Weight Variation Across Repeated Runs")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

print("\nAll figures displayed successfully.")

# =============================
# 14. LIME FIDELITY - Clean vs Boundary vs Outlier
# =============================

def compute_lime_fidelity(instance, exp_obj, model):
    
    # Random Forest prediction
    rf_proba = model.predict_proba(instance.reshape(1, -1))[0]
    predicted_class = model.predict(instance.reshape(1, -1))[0]
    rf_pred_prob = rf_proba[predicted_class]

    # LIME local surrogate prediction
    lime_local_pred = exp_obj.local_pred[0]  # as array
    fidelity_error = abs(rf_pred_prob - lime_local_pred)

    return {
        "exp_score": exp_obj.score,
        "predicted_class": predicted_class,
        "rf_pred_prob": rf_pred_prob,
        "lime_pred_prob": lime_local_pred,
        "fidelity_error": fidelity_error
    }

# Compute fidelity for all three instances
fidelity_clean = compute_lime_fidelity(clean_instance, exp_clean, rf)
fidelity_boundary = compute_lime_fidelity(boundary_instance, exp_boundary, rf)
fidelity_outlier = compute_lime_fidelity(outlier_instance, exp_outlier, rf)

# Combine results into a DataFrame for display
fidelity_df = pd.DataFrame({
    "Instance": ["Clean-region", "Tight-boundary", "Outlier"],
    "LIME exp.score": [fidelity_clean["exp_score"], fidelity_boundary["exp_score"], fidelity_outlier["exp_score"]],
    "Predicted Class (RF)": [fidelity_clean["predicted_class"], fidelity_boundary["predicted_class"], fidelity_outlier["predicted_class"]],
    "RF Pred Prob": [fidelity_clean["rf_pred_prob"], fidelity_boundary["rf_pred_prob"], fidelity_outlier["rf_pred_prob"]],
    "LIME Surrogate Pred": [fidelity_clean["lime_pred_prob"], fidelity_boundary["lime_pred_prob"], fidelity_outlier["lime_pred_prob"]],
    "Fidelity Error": [fidelity_clean["fidelity_error"], fidelity_boundary["fidelity_error"], fidelity_outlier["fidelity_error"]]
})

print("\nLIME Fidelity Comparison:")
print(fidelity_df)

# =============================
# Fidelity metrics for plotting
# =============================
instances = ["Clean-region", "Tight-boundary", "Outlier"]
fidelity_errors = [0.062311, 0.186189, 0.063113]   # from your results
exp_scores = [0.057184, 0.444965, 0.117128]       # LIME exp.score from your results

x = np.arange(len(instances))
width = 0.35

# =============================
# Plot bar chart
# =============================
fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, fidelity_errors, width, label="Fidelity Error", color="salmon")
bars2 = ax.bar(x + width/2, exp_scores, width, label="LIME exp.score", color="skyblue")

# Labels, title, and legend
ax.set_ylabel("Score / Error")
ax.set_xlabel("Instance Type")
ax.set_title("LIME Fidelity Comparison: Clean vs Boundary vs Outlier")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.axhline(0, color="black", linewidth=1)
ax.legend()

# Add text labels on top of bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

# =============================
# 15. LIME STABILITY METRIC
# =============================

from sklearn.metrics.pairwise import cosine_similarity

def compute_lime_stability(stability_df):

    lime_vectors = stability_df[["x1_weight", "x2_weight"]].values

    similarity_matrix = cosine_similarity(lime_vectors)

    # Remove diagonal values because each run is compared with itself
    off_diagonal_similarities = similarity_matrix[
        ~np.eye(similarity_matrix.shape[0], dtype=bool)
    ]

    stability_mean = np.mean(off_diagonal_similarities)
    stability_std = np.std(off_diagonal_similarities)

    return stability_mean, stability_std


# Compute stability scores
clean_stability_mean, clean_stability_std = compute_lime_stability(
    stability_clean_df
)

boundary_stability_mean, boundary_stability_std = compute_lime_stability(
    stability_boundary_df
)

outlier_stability_mean, outlier_stability_std = compute_lime_stability(
    stability_outlier_df
)

# Print results
print("\nLIME Stability Comparison:")
print("Clean-region Mean Cosine Similarity:", clean_stability_mean)
print("Clean-region Standard Deviation:", clean_stability_std)

print("Tight-boundary Mean Cosine Similarity:", boundary_stability_mean)
print("Tight-boundary Standard Deviation:", boundary_stability_std)

print("Outlier Mean Cosine Similarity:", outlier_stability_mean)
print("Outlier Standard Deviation:", outlier_stability_std)


# clean table
stability_metric_df = pd.DataFrame({
    "Instance": ["Clean-region", "Tight-boundary", "Outlier"],
    "Mean Cosine Similarity": [
        clean_stability_mean,
        boundary_stability_mean,
        outlier_stability_mean
    ],
    "Standard Deviation": [
        clean_stability_std,
        boundary_stability_std,
        outlier_stability_std
    ]
})

print("\nLIME Stability Metric Table:")
print(stability_metric_df)

# =============================
# 16. LIME SPARSITY METRIC
# =============================

def compute_lime_sparsity(weights_dict, feature_names, threshold=1e-6):
    

    # Convert weights into fixed-order vector
    weight_vector = np.array([
        weights_dict.get(f, 0) for f in feature_names
    ])

    # Count non-zero features
    non_zero_features = np.sum(np.abs(weight_vector) > threshold)

    total_features = len(weight_vector)

    sparsity_score = 1 - (non_zero_features / total_features)

    return {
        "weight_vector": weight_vector,
        "non_zero_features": non_zero_features,
        "total_features": total_features,
        "sparsity_score": sparsity_score
    }


# =============================
# Compute sparsity for all instances
# =============================

sparsity_clean = compute_lime_sparsity(
    weights_clean,
    feature_names
)

sparsity_boundary = compute_lime_sparsity(
    weights_boundary,
    feature_names
)

sparsity_outlier = compute_lime_sparsity(
    weights_outlier,
    feature_names
)

# =============================
# Create comparison table
# =============================
sparsity_df = pd.DataFrame({
    "Instance": [
        "Clean-region",
        "Tight-boundary",
        "Outlier"
    ],
    "Non-zero Features": [
        sparsity_clean["non_zero_features"],
        sparsity_boundary["non_zero_features"],
        sparsity_outlier["non_zero_features"]
    ],
    "Total Features": [
        sparsity_clean["total_features"],
        sparsity_boundary["total_features"],
        sparsity_outlier["total_features"]
    ],
    "Sparsity Score": [
        sparsity_clean["sparsity_score"],
        sparsity_boundary["sparsity_score"],
        sparsity_outlier["sparsity_score"]
    ]
})

print("\nLIME Sparsity Comparison:")
print(sparsity_df)

# =============================
# Sparsity Bar Plot
# =============================
plt.figure(figsize=(7, 5))

bars = plt.bar(
    sparsity_df["Instance"],
    sparsity_df["Sparsity Score"]
)

plt.ylabel("Sparsity Score")
plt.xlabel("Instance Type")
plt.title("LIME Sparsity: Clean vs Boundary vs Outlier")
plt.axhline(0, color="black", linewidth=1)

# Add labels on bars
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