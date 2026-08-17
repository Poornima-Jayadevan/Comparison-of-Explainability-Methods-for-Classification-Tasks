# ===============================================
# LIME Explanation for Pre-trained Random Forest (Clean Dataset Only)
# ===============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from lime.lime_tabular import LimeTabularExplainer

# =============================
# 1. Load saved model & training data
# =============================
rf = joblib.load("rf_clean_model.pkl")
X_train = joblib.load("X_train_clean.pkl")
feature_names = joblib.load("feature_names_clean.pkl")

# Load dataset for selecting instances
df_clean = pd.read_csv("df_clean.csv")

# =============================
# 2. Pick one representative instance
# =============================
instance_idx = 0   
instance = df_clean.iloc[instance_idx][feature_names].values.astype(float)
true_label = df_clean.iloc[instance_idx]["label"]

print("Selected Instance:", instance, "True Label:", true_label)
print("Model Prediction:", rf.predict(instance.reshape(1, -1))[0])
print("Predicted Probabilities:", rf.predict_proba(instance.reshape(1, -1))[0])

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
# 4. Generate LIME explanation
# =============================
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=rf.predict_proba,
    num_features=2
)

weights = dict(exp.as_list())

print("\nLIME Weights:")
print(weights)

# =============================
# 5. STANDARD LIME BAR PLOT
# =============================
fig = exp.as_pyplot_figure()
plt.title("LIME Feature Importance - Random Forest (Clean Dataset)")
plt.tight_layout()
plt.show()

# =============================
# 6. Compute arrow lengths
# =============================
max_weight = max(abs(v) for v in weights.values()) if weights else 1.0

x_arrow = 2 * weights.get("x1", 0) / max_weight
y_arrow = 2 * weights.get("x2", 0) / max_weight

# Arrow endpoints
x_end = instance[0] + x_arrow
y_end = instance[1] + y_arrow

# =============================
# 7. Define plot limits so arrows fit
# =============================
pad = 0.8

x_min = min(df_clean["x1"].min() - 1, instance[0], x_end) - pad
x_max = max(df_clean["x1"].max() + 1, instance[0], x_end) + pad
y_min = min(df_clean["x2"].min() - 1, instance[1], y_end) - pad
y_max = max(df_clean["x2"].max() + 1, instance[1], y_end) + pad

# =============================
# 8. Create decision boundary grid
# =============================
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = rf.predict(grid).reshape(xx.shape)

# =============================
# 9. DECISION BOUNDARY + LIME ARROWS
# =============================
plt.figure(figsize=(6, 6))

# Background decision regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

# Scatter plot of dataset
plt.scatter(
    df_clean["x1"],
    df_clean["x2"],
    c=df_clean["label"],
    s=12,
    alpha=0.6
)

# True boundary line x1 = x2
plt.axline((0, 0), slope=1, linestyle="--", color="k")

# Explained instance
plt.scatter(
    instance[0],
    instance[1],
    color="red",
    marker="X",
    s=150,
    edgecolor="black",
    label="Explained Instance",
    zorder=5
)

# LIME arrows
plt.arrow(
    instance[0], instance[1], x_arrow, 0,
    head_width=0.2,
    head_length=0.2,
    color="blue",
    length_includes_head=True,
    zorder=6
)

plt.arrow(
    instance[0], instance[1], 0, y_arrow,
    head_width=0.2,
    head_length=0.2,
    color="blue",
    length_includes_head=True,
    zorder=6
)

# Apply limits so arrows stay inside border
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("LIME Explanation on Clean Dataset")
plt.legend()
plt.tight_layout()
plt.show()

# =============================
# 10. MULTI-INSTANCE LIME COMPARISON
# =============================

instance_indices = [0, len(df_clean)//2, len(df_clean)-1]

comparison_rows = []

for idx in instance_indices:
    inst = df_clean.iloc[idx][feature_names].values.astype(float)
    exp_i = explainer.explain_instance(
        data_row=inst,
        predict_fn=rf.predict_proba,
        num_features=2
    )
    w = dict(exp_i.as_list())

    comparison_rows.append({
        "instance_idx": idx,
        "x1_weight": w.get("x1", 0),
        "x2_weight": w.get("x2", 0)
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\nLIME weight comparison across instances:")
print(comparison_df)

# Bar chart for comparison
plot_df = comparison_df.set_index("instance_idx")[["x1_weight", "x2_weight"]]
plot_df.plot(kind="bar", figsize=(7, 5))
plt.title("LIME Weight Comparison Across Multiple Clean Instances")
plt.xlabel("Instance Index")
plt.ylabel("LIME Weight")
plt.axhline(0, color="black", linewidth=1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =============================
# 11. LIME STABILITY PLOT
# =============================
# Run LIME several times on the same instance
n_runs = 30
stability_rows = []

for run in range(n_runs):
    exp_run = explainer.explain_instance(
        data_row=instance,
        predict_fn=rf.predict_proba,
        num_features=2
    )
    w_run = dict(exp_run.as_list())

    stability_rows.append({
        "run": run + 1,
        "x1_weight": w_run.get("x1", 0),
        "x2_weight": w_run.get("x2", 0)
    })

stability_df = pd.DataFrame(stability_rows)

print("\nLIME stability results:")
print(stability_df.head())

# Boxplot
plt.figure(figsize=(6, 5))
plt.boxplot(
    [stability_df["x1_weight"], stability_df["x2_weight"]],
    labels=["x1", "x2"]
)
plt.axhline(0, color="black", linewidth=1)
plt.ylabel("LIME Weight")
plt.title("LIME Stability Across Repeated Runs")
plt.tight_layout()
plt.show()

# =============================
# 12. LIME STABILITY LINE PLOT
# =============================
plt.figure(figsize=(7, 5))
plt.plot(stability_df["run"], stability_df["x1_weight"], marker="o", label="x1")
plt.plot(stability_df["run"], stability_df["x2_weight"], marker="s", label="x2")
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Run")
plt.ylabel("LIME Weight")
plt.title("LIME Weight Variation Across Repeated Runs")
plt.legend()
plt.tight_layout()
plt.show()

# =============================
# 13. LIME FIDELITY
# =============================


# the Random Forest model around the selected instance

print("\nLIME Fidelity:")

# 1. LIME local surrogate score

lime_fidelity_score = exp.score

print("LIME Fidelity Score (exp.score):", lime_fidelity_score)

# 2. Compare RF prediction probability and LIME local prediction
rf_proba = rf.predict_proba(instance.reshape(1, -1))[0]

# LIME local prediction
lime_local_pred = exp.local_pred

print("Random Forest predicted probabilities:", rf_proba)
print("LIME local surrogate prediction:", lime_local_pred)

# 3. Absolute difference between RF and LIME prediction
# For binary classification, compare predicted class probability
predicted_class = rf.predict(instance.reshape(1, -1))[0]

rf_pred_prob = rf_proba[predicted_class]
lime_pred_prob = lime_local_pred[0]

fidelity_error = abs(rf_pred_prob - lime_pred_prob)

print("Predicted class:", predicted_class)
print("RF probability for predicted class:", rf_pred_prob)
print("LIME surrogate probability:", lime_pred_prob)
print("Fidelity Error:", fidelity_error)

# =============================
# 14. LIME STABILITY METRIC
# =============================

from sklearn.metrics.pairwise import cosine_similarity

n_runs = 30
lime_vectors = []

for run in range(n_runs):
    exp_run = explainer.explain_instance(
        data_row=instance,
        predict_fn=rf.predict_proba,
        num_features=2
    )

    w_run = dict(exp_run.as_list())

    
    weight_vector = [
        w_run.get("x1", 0),
        w_run.get("x2", 0)
    ]

    lime_vectors.append(weight_vector)

lime_vectors = np.array(lime_vectors)

# Pairwise cosine similarity between all repeated explanations
similarity_matrix = cosine_similarity(lime_vectors)


off_diagonal_similarities = similarity_matrix[
    ~np.eye(similarity_matrix.shape[0], dtype=bool)
]

# Final stability score
lime_stability_score = np.mean(off_diagonal_similarities)
lime_stability_std = np.std(off_diagonal_similarities)

print("\nLIME Stability:")
print("Mean Cosine Similarity:", lime_stability_score)
print("Standard Deviation:", lime_stability_std)

# =============================
# 15. LIME SPARSITY METRIC
# =============================


print("\nLIME Sparsity:")

# Get LIME weights for selected instance
lime_weights = dict(exp.as_list())

# Convert to fixed feature order
weight_vector = np.array([
    lime_weights.get("x1", 0),
    lime_weights.get("x2", 0)
])

# Small threshold to avoid counting tiny numerical values
threshold = 1e-6

# Number of non-zero feature contributions
non_zero_features = np.sum(np.abs(weight_vector) > threshold)

# Total number of features
total_features = len(weight_vector)

# Sparsity score

sparsity_score = 1 - (non_zero_features / total_features)

print("LIME weight vector:", weight_vector)
print("Number of non-zero features:", non_zero_features)
print("Total features:", total_features)
print("Sparsity Score:", sparsity_score)