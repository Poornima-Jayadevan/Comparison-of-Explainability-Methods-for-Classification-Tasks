# ===============================================
# LIME Comparison: Tight-boundary
# ===============================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import joblib
from lime.lime_tabular import LimeTabularExplainer

# =============================
# 1. Load trained model & data
# =============================
rf = joblib.load("rf_boundary_model.pkl")
X_train = joblib.load("X_train_boundary.pkl")
feature_names = joblib.load("feature_names_boundary.pkl")

# Load full dataset for visualization
df = pd.read_csv("clean_plus_tight_boundary.csv")
X = df[feature_names].values
y = df["label"].values

# =============================
# 2. Compute distance metric
# =============================
# Distance from boundary line x1 = x2
df["distance"] = np.abs(df["x1"] - df["x2"]) / np.sqrt(2)

# Clean-region instance (largest distance)
clean_point = df.sort_values("distance", ascending=False).iloc[0]
clean_instance = clean_point[feature_names].values.astype(float)

# Tight-boundary instance (smallest distance)
boundary_point = df.sort_values("distance").iloc[0]
boundary_instance = boundary_point[feature_names].values.astype(float)

print("Clean-region instance:", clean_instance, "Label:", clean_point["label"])
print("Tight-boundary instance:", boundary_instance, "Label:", boundary_point["label"])

print("RF prediction for clean instance:", rf.predict(clean_instance.reshape(1, -1))[0])
print("RF probabilities for clean instance:", rf.predict_proba(clean_instance.reshape(1, -1))[0])

print("RF prediction for boundary instance:", rf.predict(boundary_instance.reshape(1, -1))[0])
print("RF probabilities for boundary instance:", rf.predict_proba(boundary_instance.reshape(1, -1))[0])

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

weights_clean = dict(exp_clean.as_list())
weights_boundary = dict(exp_boundary.as_list())

print("\nLIME weights (Clean-region):", weights_clean)
print("LIME weights (Boundary):", weights_boundary)

# =============================
# 5. SIDE-BY-SIDE LIME FEATURE IMPORTANCE
# =============================
features = feature_names

clean_values = [weights_clean.get(f, 0) for f in features]
boundary_values = [weights_boundary.get(f, 0) for f in features]

colors_clean = ["green" if v > 0 else "red" for v in clean_values]
colors_boundary = ["green" if v > 0 else "red" for v in boundary_values]

# Shared x-axis range for fair comparison
all_values = clean_values + boundary_values
x_bar_min = min(all_values) - 0.05
x_bar_max = max(all_values) + 0.05

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Clean-region bar plot
axes[0].barh(features, clean_values, color=colors_clean)
axes[0].axvline(0, color="black")
axes[0].set_xlim(x_bar_min, x_bar_max)
axes[0].set_title("LIME Feature Importance\nClean-region Instance")
axes[0].set_xlabel("LIME Weight")

# Tight-boundary bar plot
axes[1].barh(features, boundary_values, color=colors_boundary)
axes[1].axvline(0, color="black")
axes[1].set_xlim(x_bar_min, x_bar_max)
axes[1].set_title("LIME Feature Importance\nTight-boundary Instance")
axes[1].set_xlabel("LIME Weight")

plt.tight_layout()
plt.show()

# =============================
# 6. Create decision boundary grid
# =============================
padding = 1.0
x_min = df["x1"].min() - padding
x_max = df["x1"].max() + padding
y_min = df["x2"].min() - padding
y_max = df["x2"].max() + padding

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

Z = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# =============================
# 7. Normalize arrows for fair comparison
# =============================
all_weights = np.array(
    list(weights_clean.values()) + list(weights_boundary.values())
)

max_weight = np.max(np.abs(all_weights)) if len(all_weights) > 0 else 1.0
arrow_scale = 1.5

def get_scaled_arrow(weights, feature):
    raw = weights.get(feature, 0)
    if max_weight == 0:
        return 0.0
    scaled = arrow_scale * raw / max_weight

    min_len = 0.2
    if raw != 0 and abs(scaled) < min_len:
        scaled = np.sign(raw) * min_len
    return scaled

# =============================
# 8. Side-by-side LIME plots
# =============================
plt.figure(figsize=(14, 6))

# Plot 1: Clean-region
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.2, cmap="viridis")
plt.scatter(df["x1"], df["x2"], c=df["label"], s=15, alpha=0.4)
plt.axline((0, 0), slope=1, linestyle="--", color="black")
plt.scatter(
    clean_instance[0], clean_instance[1],
    color="red", marker="X", s=150,
    edgecolor="black", label="Explained Instance"
)

x_arrow_clean = get_scaled_arrow(weights_clean, "x1")
y_arrow_clean = get_scaled_arrow(weights_clean, "x2")

plt.arrow(
    clean_instance[0], clean_instance[1], x_arrow_clean, 0,
    head_width=0.15, head_length=0.18,
    color="blue", length_includes_head=True
)
plt.arrow(
    clean_instance[0], clean_instance[1], 0, y_arrow_clean,
    head_width=0.15, head_length=0.18,
    color="blue", length_includes_head=True
)

plt.title("Clean-region Instance")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()

# Plot 2: Tight-boundary
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.2, cmap="viridis")
plt.scatter(df["x1"], df["x2"], c=df["label"], s=15, alpha=0.4)
plt.axline((0, 0), slope=1, linestyle="--", color="black")
plt.scatter(
    boundary_instance[0], boundary_instance[1],
    color="red", marker="X", s=150,
    edgecolor="black", label="Explained Instance"
)

x_arrow_boundary = get_scaled_arrow(weights_boundary, "x1")
y_arrow_boundary = get_scaled_arrow(weights_boundary, "x2")

plt.arrow(
    boundary_instance[0], boundary_instance[1], x_arrow_boundary, 0,
    head_width=0.15, head_length=0.18,
    color="green", length_includes_head=True
)
plt.arrow(
    boundary_instance[0], boundary_instance[1], 0, y_arrow_boundary,
    head_width=0.15, head_length=0.18,
    color="green", length_includes_head=True
)

plt.title("Tight-boundary Instance")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()

plt.suptitle("Side-by-Side LIME Explanation Comparison")
plt.tight_layout()
plt.show()

# =============================
# 9. LIME weight comparison table
# =============================
comparison_df = pd.DataFrame({
    "Feature": feature_names,
    "Clean-region weight": [weights_clean.get(f, 0) for f in feature_names],
    "Boundary weight": [weights_boundary.get(f, 0) for f in feature_names]
})

print("\nLIME Weight Comparison:")
print(comparison_df)

# =============================
# 10. LIME weight comparison bar chart
# =============================
plot_df = comparison_df.set_index("Feature")
plot_df.plot(kind="bar", figsize=(7, 5))
plt.title("LIME Weight Comparison: Clean-region vs Tight-boundary")
plt.ylabel("LIME Weight")
plt.axhline(0, color="black", linewidth=1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =============================
# 11. LIME stability analysis
# =============================
n_runs = 30
stability_clean = []
stability_boundary = []

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

stability_clean_df = pd.DataFrame(stability_clean)
stability_boundary_df = pd.DataFrame(stability_boundary)

print("\nStability results (clean instance):")
print(stability_clean_df.head())

print("\nStability results (boundary instance):")
print(stability_boundary_df.head())

# =============================
# 12. Stability boxplot
# =============================
plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        stability_clean_df["x1_weight"],
        stability_clean_df["x2_weight"],
        stability_boundary_df["x1_weight"],
        stability_boundary_df["x2_weight"]
    ],
    tick_labels=["Clean x1", "Clean x2", "Boundary x1", "Boundary x2"]
)
plt.axhline(0, color="black", linewidth=1)
plt.ylabel("LIME Weight")
plt.title("LIME Stability Across Repeated Runs")
plt.tight_layout()
plt.show()

# =============================
# 13. Stability line plots
# =============================
plt.figure(figsize=(10, 5))

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

plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Run")
plt.ylabel("LIME Weight")
plt.title("LIME Weight Variation Across Repeated Runs")
plt.legend()
plt.tight_layout()
plt.show()

# =============================
# 14. LIME Fidelity - Clean vs Boundary
# =============================

def compute_lime_fidelity(instance, exp_obj, model):
    
    # Random Forest prediction
    rf_proba = model.predict_proba(instance.reshape(1, -1))[0]
    predicted_class = model.predict(instance.reshape(1, -1))[0]
    rf_pred_prob = rf_proba[predicted_class]

    # LIME local surrogate prediction (probability for the predicted class)
    lime_local_pred = exp_obj.local_pred[0]  # array
    fidelity_error = abs(rf_pred_prob - lime_local_pred)

    return {
        "exp_score": exp_obj.score,
        "predicted_class": predicted_class,
        "rf_pred_prob": rf_pred_prob,
        "lime_pred_prob": lime_local_pred,
        "fidelity_error": fidelity_error
    }

# Compute fidelity for clean and boundary instances
fidelity_clean = compute_lime_fidelity(clean_instance, exp_clean, rf)
fidelity_boundary = compute_lime_fidelity(boundary_instance, exp_boundary, rf)

# Combine into a DataFrame
fidelity_df = pd.DataFrame({
    "Instance": ["Clean-region", "Tight-boundary"],
    "LIME exp.score": [fidelity_clean["exp_score"], fidelity_boundary["exp_score"]],
    "Predicted Class (RF)": [fidelity_clean["predicted_class"], fidelity_boundary["predicted_class"]],
    "RF Pred Prob": [fidelity_clean["rf_pred_prob"], fidelity_boundary["rf_pred_prob"]],
    "LIME Surrogate Pred": [fidelity_clean["lime_pred_prob"], fidelity_boundary["lime_pred_prob"]],
    "Fidelity Error": [fidelity_clean["fidelity_error"], fidelity_boundary["fidelity_error"]]
})

print("\nLIME Fidelity Comparison:")
print(fidelity_df)

# =============================
# Fidelity metrics 
# =============================
instances = ["Clean-region", "Tight-boundary"]
fidelity_errors = [0.149184, 0.135145]  # Fidelity Error
exp_scores = [0.588613, 0.740942]       # LIME exp.score

x = np.arange(len(instances))
width = 0.35

# =============================
# Plot bar chart
# =============================
fig, ax = plt.subplots(figsize=(7, 5))

# Bars
bars1 = ax.bar(x - width/2, fidelity_errors, width, label="Fidelity Error", color="salmon")
bars2 = ax.bar(x + width/2, exp_scores, width, label="LIME exp.score", color="skyblue")

# Labels, title, legend
ax.set_ylabel("Score / Error")
ax.set_xlabel("Instance Type")
ax.set_title("LIME Fidelity Comparison: Clean vs Tight-boundary")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.axhline(0, color="black", linewidth=1)
ax.legend()

# Add numeric labels on top of bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),  # vertical offset
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

# =============================
# 15. LIME STABILITY METRIC
# Clean-region vs Tight-boundary
# =============================

from sklearn.metrics.pairwise import cosine_similarity

def compute_lime_stability(stability_df):
    

    lime_vectors = stability_df[["x1_weight", "x2_weight"]].values

    similarity_matrix = cosine_similarity(lime_vectors)

    # remove diagonal because each run is compared with itself
    off_diagonal = similarity_matrix[
        ~np.eye(similarity_matrix.shape[0], dtype=bool)
    ]

    stability_mean = np.mean(off_diagonal)
    stability_std = np.std(off_diagonal)

    return stability_mean, stability_std


clean_stability_mean, clean_stability_std = compute_lime_stability(
    stability_clean_df
)

boundary_stability_mean, boundary_stability_std = compute_lime_stability(
    stability_boundary_df
)

stability_metric_df = pd.DataFrame({
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

print("\nLIME Stability Metric Table:")
print(stability_metric_df)

# =============================
# 16. LIME SPARSITY METRIC
# Clean-region vs Tight-boundary
# =============================

def compute_lime_sparsity(weights_dict, feature_names, threshold=1e-6):
    

    # Convert weights into fixed-order vector
    weight_vector = np.array([
        weights_dict.get(f, 0) for f in feature_names
    ])

    # Count non-zero contributions
    non_zero_features = np.sum(np.abs(weight_vector) > threshold)

    total_features = len(weight_vector)

    sparsity_score = 1 - (non_zero_features / total_features)

    return {
        "weight_vector": weight_vector,
        "non_zero_features": non_zero_features,
        "total_features": total_features,
        "sparsity_score": sparsity_score
    }


# Compute sparsity for clean-region explanation
sparsity_clean = compute_lime_sparsity(
    weights_clean,
    feature_names
)

# Compute sparsity for boundary explanation
sparsity_boundary = compute_lime_sparsity(
    weights_boundary,
    feature_names
)

# =============================
# Create comparison table
# =============================
sparsity_df = pd.DataFrame({
    "Instance": ["Clean-region", "Tight-boundary"],
    "Non-zero Features": [
        sparsity_clean["non_zero_features"],
        sparsity_boundary["non_zero_features"]
    ],
    "Total Features": [
        sparsity_clean["total_features"],
        sparsity_boundary["total_features"]
    ],
    "Sparsity Score": [
        sparsity_clean["sparsity_score"],
        sparsity_boundary["sparsity_score"]
    ]
})

print("\nLIME Sparsity Comparison:")
print(sparsity_df)

# =============================
# Sparsity Bar Plot
# =============================
plt.figure(figsize=(6, 5))

plt.bar(
    sparsity_df["Instance"],
    sparsity_df["Sparsity Score"]
)

plt.ylabel("Sparsity Score")
plt.xlabel("Instance Type")
plt.title("LIME Sparsity: Clean-region vs Tight-boundary")
plt.axhline(0, color="black", linewidth=1)

# Add labels on bars
for i, value in enumerate(sparsity_df["Sparsity Score"]):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center")

plt.tight_layout()
plt.show()