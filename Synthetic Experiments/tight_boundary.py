import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Reproducibility
# -----------------------------
np.random.seed(42)

# -----------------------------
# 2. Parameters
# -----------------------------
n_samples = 100
x_min, x_max = -5, 5
clean_margin = 0.7   # distance from boundary (controls "cleanliness")

# -----------------------------
# 3. Decision function
# -----------------------------
def decision_function(x1, x2):
    return x1 - x2   # decision boundary: x1 - x2 = 0

# -----------------------------
# 4. Generate candidate points
# -----------------------------
x1 = np.random.uniform(x_min, x_max, n_samples * 3)
x2 = np.random.uniform(x_min, x_max, n_samples * 3)

f = decision_function(x1, x2)
distance = np.abs(f) / np.sqrt(2)

# -----------------------------
# 5. Select ONLY clean data
# -----------------------------
mask_clean = distance > clean_margin

x1_clean = x1[mask_clean][:n_samples]
x2_clean = x2[mask_clean][:n_samples]
f_clean = f[mask_clean][:n_samples]

labels = (f_clean > 0).astype(int)

# -----------------------------
# 6. Create DataFrame
# -----------------------------
df_clean = pd.DataFrame({
    "x1": x1_clean,
    "x2": x2_clean,
    "label": labels,
    "distance_to_boundary": np.abs(f_clean) / np.sqrt(2),
    "region": "clean"
})

print(df_clean.head())
print(f"\nTotal clean samples: {len(df_clean)}")

# -----------------------------
# 7. Parameters for tight boundary
# -----------------------------
n_boundary = 300
boundary_width = 0.08   # epsilon: controls tightness

# -----------------------------
# 8. Generate candidate points
# -----------------------------
x1_b = np.random.uniform(x_min, x_max, n_boundary * 5)
x2_b = np.random.uniform(x_min, x_max, n_boundary * 5)

f_b = decision_function(x1_b, x2_b)
distance_b = np.abs(f_b) / np.sqrt(2)

# -----------------------------
# 9. Select tight-boundary points
# -----------------------------
mask_boundary = distance_b < boundary_width

x1_boundary = x1_b[mask_boundary][:n_boundary]
x2_boundary = x2_b[mask_boundary][:n_boundary]
f_boundary = f_b[mask_boundary][:n_boundary]

labels_boundary = (f_boundary > 0).astype(int)

# -----------------------------
# 10. Create DataFrame
# -----------------------------
df_boundary = pd.DataFrame({
    "x1": x1_boundary,
    "x2": x2_boundary,
    "label": labels_boundary,
    "distance_to_boundary": np.abs(f_boundary) / np.sqrt(2),
    "region": "tight_boundary"
})

print(df_boundary.head())
print(f"\nTotal boundary samples: {len(df_boundary)}")



# -----------------------------
# 11. Combined visualization
# -----------------------------
plt.figure()

plt.scatter(
    df_clean["x1"], df_clean["x2"],
    c=df_clean["label"],
    s=15, alpha=0.6, label="Clean"
)

plt.scatter(
    df_boundary["x1"], df_boundary["x2"],
    c=df_boundary["label"],
    s=20, label="Tight Boundary"
)

plt.axline((0, 0), slope=1)
plt.xlabel("x1")
plt.ylabel("x2")
#plt.legend()
plt.title("Clean Data vs Tight Boundary Region")
plt.show()

# =============================
# 12. Combine datasets
# =============================
df_combined2 = pd.concat(
    [df_clean, df_boundary],
    ignore_index=True
)

print(df_combined2["region"].value_counts())

# =============================
# 13. Save to disk
# =============================
df_combined2.to_csv("clean_plus_tight_boundary.csv", index=False)

print("Saved dataset to clean_plus_tight_boundary.csv")