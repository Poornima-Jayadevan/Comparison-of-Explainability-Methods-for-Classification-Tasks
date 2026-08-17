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
# 3. Decision function (ground truth)
# -----------------------------
def decision_function(x1, x2):
    return x1 - x2   # decision boundary: x1 - x2 = 0

# -----------------------------
# 4. Generate candidate points
# -----------------------------
x1 = np.random.uniform(x_min, x_max, n_samples * 3)
x2 = np.random.uniform(x_min, x_max, n_samples * 3)

f = decision_function(x1, x2)
distance = np.abs(f) / np.sqrt(2)       #distance from boundary (perpendicular distance from line x1 = x2)

# -----------------------------
# 5. Select ONLY clean data
# -----------------------------
mask_clean = distance > clean_margin

x1_clean = x1[mask_clean][:n_samples]
x2_clean = x2[mask_clean][:n_samples]
f_clean = f[mask_clean][:n_samples]

labels = (f_clean > 0).astype(int)
'''

# Create features
x1_subset = x1[:n_samples]
x2_subset = x2[:n_samples]

# Compute labels
f_subset = decision_function(x1_subset, x2_subset)
labels = (f_subset > 0).astype(int)
'''
# -----------------------------
# 6. Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "x1": x1_clean,
    "x2": x2_clean,
    "label": labels
})

assert mask_clean.sum() >= n_samples
print(df['label'].value_counts())
print(df.head())
print(f"\nTotal clean samples: {len(df)}")
df.to_csv("df_clean.csv", index=False)

# -----------------------------
# 7. Visualization 
# -----------------------------
plt.figure()
plt.scatter(
    df["x1"],
    df["x2"],
    c=df["label"],
    s=15
)
plt.axline((0, 0), slope=1)  # decision boundary x1 = x2
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Clean Synthetic 2D Dataset (No Boundary, No Outliers)")
plt.show()