import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
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
# 4. Train Naive Bayes
# =============================
nb = GaussianNB()
nb.fit(X_train, y_train)

# =============================
# 5. Evaluation
# =============================
y_pred = nb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =============================
# 6. Decision boundary visualization
# =============================
padding = 1.0
x1_min, x1_max = df["x1"].min() - padding, df["x1"].max() + padding
x2_min, x2_max = df["x2"].min() - padding, df["x2"].max() + padding

xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 500),
    np.linspace(x2_min, x2_max, 500)
)

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# =============================
# 7. Save model and data
# =============================
joblib.dump(nb, "nb_combined_model.pkl")
joblib.dump(X_train, "X_train_combined_nb.pkl")
joblib.dump(X_test, "X_test_combined_nb.pkl")
joblib.dump(y_test, "y_test_combined_nb.pkl")
joblib.dump(feature_names, "feature_names_combined_nb.pkl")



# =============================
# 8. Plot
# =============================
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(
    df["x1"],
    df["x2"],
    c=df["label"],
    s=15
)
plt.axline((0, 0), slope=1, linestyle="--", color="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Naive Bayes Trained on Combined Dataset")
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.show()