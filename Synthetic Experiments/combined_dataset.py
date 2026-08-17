import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# 1. Reproducibility
# =============================
np.random.seed(42)

# =============================
# 2. Parameters
# =============================
n_clean = 100
n_boundary = 300
n_outliers = 50

x_min, x_max = -5, 5
clean_margin = 0.7
boundary_width = 0.08
outlier_range = 12
flip_prob = 0.5

# =============================
# 3. Decision function
# =============================
def decision_function(x1, x2):
    return x1 - x2

# =============================
# 4. CLEAN DATA
# =============================
x1 = np.random.uniform(x_min, x_max, n_clean * 3)
x2 = np.random.uniform(x_min, x_max, n_clean * 3)

f = decision_function(x1, x2)
distance = np.abs(f) / np.sqrt(2)

mask_clean = distance > clean_margin

df_clean = pd.DataFrame({
    "x1": x1[mask_clean][:n_clean],
    "x2": x2[mask_clean][:n_clean],
    "label": (f[mask_clean][:n_clean] > 0).astype(int),
    "distance_to_boundary": distance[mask_clean][:n_clean],
    "region": "clean"
})

# =============================
# 5. TIGHT BOUNDARY DATA
# =============================
x1_b = np.random.uniform(x_min, x_max, n_boundary * 5)
x2_b = np.random.uniform(x_min, x_max, n_boundary * 5)

f_b = decision_function(x1_b, x2_b)
distance_b = np.abs(f_b) / np.sqrt(2)

mask_boundary = distance_b < boundary_width

df_boundary = pd.DataFrame({
    "x1": x1_b[mask_boundary][:n_boundary],
    "x2": x2_b[mask_boundary][:n_boundary],
    "label": (f_b[mask_boundary][:n_boundary] > 0).astype(int),
    "distance_to_boundary": distance_b[mask_boundary][:n_boundary],
    "region": "tight_boundary"
})

# =============================
# 6. OUTLIERS
# =============================
x1_o = np.random.uniform(-outlier_range, outlier_range, n_outliers)
x2_o = np.random.uniform(-outlier_range, outlier_range, n_outliers)

f_o = decision_function(x1_o, x2_o)
labels_o = (f_o > 0).astype(int)

flip_mask = np.random.rand(n_outliers) < flip_prob
labels_o[flip_mask] = 1 - labels_o[flip_mask]

df_outliers = pd.DataFrame({
    "x1": x1_o,
    "x2": x2_o,
    "label": labels_o,
    "distance_to_boundary": np.abs(f_o) / np.sqrt(2),
    "region": "outliers"
})

# =============================
# 7. COMBINE DATASETS
# =============================
df_all = pd.concat(
    [df_clean, df_boundary, df_outliers],
    ignore_index=True
)

print(df_all["region"].value_counts())
df_all.to_csv("combined.csv", index=False)

# =============================
# 8. FINAL VISUALIZATION
# =============================
plt.figure(figsize=(7, 7))

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

plt.scatter(
    df_outliers["x1"], df_outliers["x2"],
    c=df_outliers["label"],
    s=35, edgecolors="k", linewidths=0.8,
    label="Outliers"
)

plt.xlim(-13, 13)
plt.ylim(-13, 13)
plt.gca().set_aspect('equal', adjustable='box')


plt.axline((0, 0), slope=1)
plt.xlabel("x1")
plt.ylabel("x2")
#plt.legend()
plt.title("Clean vs Tight Boundary vs Outliers")
plt.show()

