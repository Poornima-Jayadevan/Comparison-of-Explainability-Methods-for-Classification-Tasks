import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# =============================
# 1. Load combined dataset
# =============================
df = pd.read_csv("combined.csv")

print(df["region"].value_counts())

# =============================
# 2. Features and labels
# =============================
feature_names = ["x1", "x2"]
X = df[["x1", "x2"]].values
y = df["label"].values

# =============================
# 3. Train / test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =============================
# 4. Train Random Forest
# =============================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =============================
# 5. Evaluation 
# =============================
y_pred = rf.predict(X_test)

print("Overall Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =============================
# 6. Per-region accuracy
# =============================
test_df = df.iloc[y_test.index] if hasattr(y_test, "index") else None

if test_df is not None:
    test_df = test_df.copy()
    test_df["pred"] = y_pred

    print("\nAccuracy by region:")
    for region in test_df["region"].unique():
        mask = test_df["region"] == region
        acc = accuracy_score(test_df.loc[mask, "label"],
                              test_df.loc[mask, "pred"])
        print(f"{region}: {acc:.3f}")

# =============================
# 7. Decision boundary visualization
# =============================

padding = 1.0
x1_min, x1_max = df["x1"].min() - padding, df["x1"].max() + padding
x2_min, x2_max = df["x2"].min() - padding, df["x2"].max() + padding

xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 500),
    np.linspace(x2_min, x2_max, 500)
)

Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# =============================
# 8. Save Everything Needed
# =============================

joblib.dump(rf, "rf_combined_model.pkl")
joblib.dump(X_train, "X_train_combined.pkl")
joblib.dump(feature_names, "feature_names_combined.pkl")

print("\nModel and training data saved successfully.")


plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(df["x1"], df["x2"], c=df["label"], s=12, alpha=0.6)
plt.axline((0, 0), slope=1, linestyle="--", color="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Random Forest trained on combined dataset")
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.show()