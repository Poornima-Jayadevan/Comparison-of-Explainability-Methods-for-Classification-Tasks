from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib


# =============================
# 1. Load clean dataset
# =============================
df_clean = pd.read_csv("df_clean.csv")


# =============================
# 2. Select features and target
# =============================
feature_names = ["x1", "x2"]

X = df_clean[feature_names].values
y = df_clean["label"].values

# =============================
# 3. Train / Test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
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

joblib.dump(rf, "rf_clean_model.pkl")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =============================
# 6. Create Grid
# =============================
xx, yy = np.meshgrid(
    np.linspace(-5, 5, 400),
    np.linspace(-5, 5, 400)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = rf.predict(grid)
Z = Z.reshape(xx.shape)

# =============================
# 7. Save Everything Needed
# =============================

joblib.dump(rf, "rf_clean_model.pkl")
joblib.dump(X_train, "X_train_clean.pkl")
joblib.dump(feature_names, "feature_names_clean.pkl")

print("\nModel and training data saved successfully.")


# =============================
# 8. Plot
# =============================
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(
    df_clean["x1"],
    df_clean["x2"],
    c=df_clean["label"],
    s=15
)



plt.axline((0, 0), slope=1, color="k", linestyle="--")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Random Forest Trained on Clean Data")
plt.show()
