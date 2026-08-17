import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =============================
# 1. Load dataset
# =============================
df = pd.read_csv("clean_plus_tight_boundary.csv")

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
xx, yy = np.meshgrid(
    np.linspace(-5, 5, 400),
    np.linspace(-5, 5, 400)
)

Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


joblib.dump(nb, "nb_boundary_model.pkl")
joblib.dump(X_train, "X_train_boundary_nb.pkl")
joblib.dump(X_test, "X_test_boundary_nb.pkl")
joblib.dump(y_test, "y_test_boundary_nb.pkl")
joblib.dump(feature_names, "feature_names_boundary_nb.pkl")

print("\nNB model and data saved successfully.")

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
plt.title("Naive Bayes Trained on Clean + Tight Boundary Data")
plt.show()