# ===============================================
# LIME Explanation for Pre-trained Naive Bayes 
# ===============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from lime.lime_tabular import LimeTabularExplainer

# =============================
# Load saved model & training data
# =============================
nb = joblib.load("nb_clean_model.pkl")
X_train = joblib.load("X_train_clean_nb.pkl")
feature_names = joblib.load("feature_names_clean_nb.pkl")

# Load dataset for selecting instances
df_clean = pd.read_csv("df_clean.csv")

# =============================
# Pick one representative instance
# =============================
instance_idx = 0   
instance = df_clean.iloc[instance_idx][feature_names].values.astype(float)
true_label = df_clean.iloc[instance_idx]["label"]

print("Selected Instance:", instance, "True Label:", true_label)
print("Model Prediction:", nb.predict(instance.reshape(1, -1))[0])
print("Predicted Probabilities:", nb.predict_proba(instance.reshape(1, -1))[0])

# =============================
# Initialize LIME Explainer
# =============================
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=["Class 0", "Class 1"],
    mode="classification",
    discretize_continuous=False
)

# =============================
# Generate LIME explanation
# =============================
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=nb.predict_proba,
    num_features=2
)

weights = dict(exp.as_list())

print("\nLIME Weights:")
print(weights)

# =============================
# STANDARD LIME BAR PLOT
# =============================
fig = exp.as_pyplot_figure()
plt.title("LIME Feature Importance - Naive Bayes (Clean Dataset)")
plt.tight_layout()
plt.show()
plt.close()

# =============================
# Compute arrow lengths
# =============================
max_weight = max(abs(v) for v in weights.values()) if weights else 1.0

x_arrow = 2 * weights.get("x1", 0) / max_weight
y_arrow = 2 * weights.get("x2", 0) / max_weight

# Arrow endpoints
x_end = instance[0] + x_arrow
y_end = instance[1] + y_arrow

# =============================
# Define plot limits so arrows fit
# =============================
pad = 0.8

x_min = min(df_clean["x1"].min() - 1, instance[0], x_end) - pad
x_max = max(df_clean["x1"].max() + 1, instance[0], x_end) + pad
y_min = min(df_clean["x2"].min() - 1, instance[1], y_end) - pad
y_max = max(df_clean["x2"].max() + 1, instance[1], y_end) + pad

# =============================
# Create decision boundary grid
# =============================
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = nb.predict(grid).reshape(xx.shape)

# =============================
# DECISION BOUNDARY + LIME ARROWS
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
plt.close()

# =============================
# MULTI-INSTANCE LIME COMPARISON
# =============================

instance_indices = [0, len(df_clean)//2, len(df_clean)-1]

comparison_rows = []

for idx in instance_indices:
    inst = df_clean.iloc[idx][feature_names].values.astype(float)
    exp_i = explainer.explain_instance(
        data_row=inst,
        predict_fn=nb.predict_proba,
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
plt.close()

# =============================
# LIME STABILITY PLOT
# =============================
# Run LIME several times on the same instance
n_runs = 30
stability_rows = []

for run in range(n_runs):
    exp_run = explainer.explain_instance(
        data_row=instance,
        predict_fn=nb.predict_proba,
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
    tick_labels=["x1", "x2"]
)
plt.axhline(0, color="black", linewidth=1)
plt.ylabel("LIME Weight")
plt.title("LIME Stability Across Repeated Runs")
plt.tight_layout()
plt.show()
plt.close()

# =============================
# LIME STABILITY LINE PLOT
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
plt.close()

# =============================
# LIME Fidelity Comparison for Naive Bayes
# =============================

# Compute fidelity for multiple instances 
instance_indices = [0, len(df_clean)//2, len(df_clean)-1]
fidelity_rows = []

for idx in instance_indices:
    inst = df_clean.iloc[idx][feature_names].values.astype(float)
    exp_inst = explainer.explain_instance(
        data_row=inst,
        predict_fn=nb.predict_proba,
        num_features=2
    )
    
    # Compute fidelity
    rf_proba = nb.predict_proba(inst.reshape(1, -1))[0]
    predicted_class = nb.predict(inst.reshape(1, -1))[0]
    rf_pred_prob = rf_proba[predicted_class]
    lime_pred_prob = exp_inst.local_pred[0]
    fidelity_error = abs(rf_pred_prob - lime_pred_prob)
    
    fidelity_rows.append({
        "Instance": f"Instance {idx}",
        "LIME exp.score": exp_inst.score,
        "Predicted Class (NB)": predicted_class,
        "NB Pred Prob": rf_pred_prob,
        "LIME Surrogate Pred": lime_pred_prob,
        "Fidelity Error": fidelity_error
    })

# Convert to DataFrame for display
fidelity_df = pd.DataFrame(fidelity_rows)
print("\nLIME Fidelity Comparison (Naive Bayes):")
print(fidelity_df)

# =============================
# Plot Fidelity Error vs LIME exp.score
# =============================

instances = fidelity_df["Instance"].values
fidelity_errors = fidelity_df["Fidelity Error"].values
exp_scores = fidelity_df["LIME exp.score"].values

x = np.arange(len(instances))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, fidelity_errors, width, label="Fidelity Error", color="salmon")
bars2 = ax.bar(x + width/2, exp_scores, width, label="LIME exp.score", color="skyblue")

# Labels, title, legend
ax.set_ylabel("Score / Error")
ax.set_xlabel("Instance")
ax.set_title("LIME Fidelity Comparison: Naive Bayes")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.axhline(0, color="black", linewidth=1)
ax.legend()

# Add numeric labels on top of bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # vertical offset
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

# =============================
# LIME STABILITY METRIC
# =============================

from sklearn.metrics.pairwise import cosine_similarity

# Each explanation vector = [x1_weight, x2_weight]
lime_vectors = stability_df[["x1_weight", "x2_weight"]].values

# Pairwise cosine similarity between all repeated explanations
similarity_matrix = cosine_similarity(lime_vectors)

# Remove diagonal values because each run is compared with itself
off_diagonal_similarities = similarity_matrix[
    ~np.eye(similarity_matrix.shape[0], dtype=bool)
]

# Final stability score
lime_stability_mean = np.mean(off_diagonal_similarities)
lime_stability_std = np.std(off_diagonal_similarities)

stability_metric_df = pd.DataFrame({
    "Model": ["Naive Bayes"],
    "Dataset": ["Clean"],
    "Mean Cosine Similarity": [lime_stability_mean],
    "Standard Deviation": [lime_stability_std]
})

print("\nLIME Stability Metric Table:")
print(stability_metric_df)

# =============================
# LIME SPARSITY METRIC
# =============================

def compute_lime_sparsity(weights_dict, feature_names, threshold=1e-6):
    

    weight_vector = np.array([
        weights_dict.get(f, 0) for f in feature_names
    ])

    non_zero_features = np.sum(np.abs(weight_vector) > threshold)
    total_features = len(weight_vector)

    sparsity_score = 1 - (non_zero_features / total_features)

    return weight_vector, non_zero_features, total_features, sparsity_score


# Compute sparsity for the selected instance
weight_vector, non_zero_features, total_features, sparsity_score = compute_lime_sparsity(
    weights,
    feature_names
)

sparsity_df = pd.DataFrame({
    "Model": ["Naive Bayes"],
    "Dataset": ["Clean"],
    "Weight Vector": [weight_vector],
    "Non-zero Features": [non_zero_features],
    "Total Features": [total_features],
    "Sparsity Score": [sparsity_score]
})

print("\nLIME Sparsity Metric Table:")
print(sparsity_df)

# =============================
# Sparsity for multiple instances
# =============================

multi_sparsity_rows = []

for idx in instance_indices:
    inst = df_clean.iloc[idx][feature_names].values.astype(float)

    exp_i = explainer.explain_instance(
        data_row=inst,
        predict_fn=nb.predict_proba,
        num_features=2
    )

    w_i = dict(exp_i.as_list())

    weight_vector_i, non_zero_i, total_i, sparsity_i = compute_lime_sparsity(
        w_i,
        feature_names
    )

    multi_sparsity_rows.append({
        "Instance": f"Instance {idx}",
        "Non-zero Features": non_zero_i,
        "Total Features": total_i,
        "Sparsity Score": sparsity_i
    })

multi_sparsity_df = pd.DataFrame(multi_sparsity_rows)

print("\nLIME Sparsity Comparison Across Instances:")
print(multi_sparsity_df)

# =============================
# Sparsity Bar Plot
# =============================

plt.figure(figsize=(7, 5))

bars = plt.bar(
    multi_sparsity_df["Instance"],
    multi_sparsity_df["Sparsity Score"]
)

plt.ylabel("Sparsity Score")
plt.xlabel("Instance")
plt.title("LIME Sparsity Comparison: Naive Bayes Clean Dataset")
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