import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =============================
# 1. Load clean dataset
# =============================
df_clean = pd.read_csv("df_clean.csv")

# =============================
# 2. Features and labels
# =============================
feature_names = ["x1", "x2"]
X = df_clean[["x1", "x2"]].values
y = df_clean["label"].values

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

#errors = (y_pred != y_test)

'''print("Errors:", np.sum(errors))
print("Accuracy:", np.mean(y_pred == y_test))

print("Misclassified points:")
print(X_test[errors])'''

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


joblib.dump(nb, "nb_clean_model.pkl")
joblib.dump(X_train, "X_train_clean_nb.pkl")
joblib.dump(X_test, "X_test_clean_nb.pkl")
joblib.dump(y_test, "y_test_clean_nb.pkl")
joblib.dump(feature_names, "feature_names_clean_nb.pkl")

print("\nNB model and data saved successfully.")

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(
    df_clean["x1"],
    df_clean["x2"],
    c=df_clean["label"],
    s=15
)
plt.axline((0, 0), slope=1, linestyle="--", color="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Naive Bayes Trained on Clean Data")
plt.show()