# ==========================================
# FLOWER CLASSIFICATION - IRIS DATASET
# ==========================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================================
# 1. LOAD IRIS DATASET
# ==========================================

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

print("\nDataset Shape:", X.shape)
print("\nFirst 5 Rows:")
print(X.head())

print("\nMissing Values:")
print(X.isnull().sum())


# ==========================================
# 2. EDA
# ==========================================

df = X.copy()
df["Species"] = iris.target_names[y]

print("\nSpecies Count:")
print(df["Species"].value_counts())

print("\nStatistical Summary:")
print(X.describe())


# ==========================================
# 3. SPECIES DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Species"
)

plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("species_distribution.png")
plt.show()


# ==========================================
# 4. FEATURE PAIR VISUALIZATION
# ==========================================

sns.pairplot(
    df,
    hue="Species"
)

plt.savefig("feature_pairplot.png")
plt.show()


# ==========================================
# 5. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    X.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))


# ==========================================
# 7. LOGISTIC REGRESSION
# ==========================================

logistic_model = LogisticRegression(
    max_iter=200
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_pred = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

print("\nLogistic Regression Accuracy:",
      round(logistic_accuracy * 100, 2), "%")


# ==========================================
# 8. DECISION TREE
# ==========================================

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_pred = tree_model.predict(X_test)

tree_accuracy = accuracy_score(
    y_test,
    tree_pred
)

print("Decision Tree Accuracy:",
      round(tree_accuracy * 100, 2), "%")


# ==========================================
# 9. MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        logistic_accuracy,
        tree_accuracy
    ]
})

print("\nModel Comparison:")
print(comparison)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy"
)

plt.ylim(0, 1.05)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    logistic_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# ==========================================
# 11. SELECT BEST MODEL
# ==========================================

if logistic_accuracy >= tree_accuracy:
    best_model = logistic_model
    best_name = "Logistic Regression"
else:
    best_model = tree_model
    best_name = "Decision Tree"

print("\nBest Model:", best_name)


# ==========================================
# 12. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/iris_model.pkl"
)

print("Model saved successfully!")


# ==========================================
# 13. SAMPLE PREDICTION
# ==========================================

sample = [[
    5.1,
    3.5,
    1.4,
    0.2
]]

prediction = best_model.predict(sample)

print(
    "\nSample Prediction:",
    iris.target_names[prediction[0]]
)

print("\nProject Completed Successfully!")