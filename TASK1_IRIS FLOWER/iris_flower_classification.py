# OIBSIP Data Science — Task 1: Iris Flower Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Load the built-in Iris dataset
iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("Classes:", list(iris.target_names))
print(df.head())

# 2. Initial inspection
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isnull().sum())
print("\\nDuplicate rows:", df.duplicated().sum())
print("\\nDescriptive statistics:\\n", df.describe())

# 3. EDA: class distribution
print("\\nClass distribution:\\n", df["species"].value_counts())
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="species", hue="species", legend=False)
plt.title("Number of Samples by Iris Species")
plt.xlabel("Species"); plt.ylabel("Number of samples")
plt.tight_layout(); plt.savefig("outputs/class_distribution.png", dpi=180); plt.show()

# 4. Pairplot and box plots
sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Iris Feature Relationships by Species", y=1.02)
plt.savefig("outputs/pairplot.png", dpi=180); plt.show()

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, feature in zip(axes.flatten(), features):
    sns.boxplot(data=df, x="species", y=feature, hue="species", legend=False, ax=ax)
    ax.set_title(feature); ax.set_xlabel("Species"); ax.set_ylabel("Measurement (cm)")
plt.suptitle("Feature Distributions by Species", y=1.02)
plt.tight_layout(); plt.savefig("outputs/boxplots.png", dpi=180); plt.show()

print("\\nFeature-selection observation: petal length and petal width show the clearest separation, while all four measurements are retained for training.")

# 5. Train/test split (80/20, stratified)
X = df[features]
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# 6. Train two classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5)
}
predictions, results = {}, []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": report["weighted avg"]["precision"],
        "Recall (weighted)": report["weighted avg"]["recall"],
        "F1-score (weighted)": report["weighted avg"]["f1-score"]
    })

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\\nModel comparison:\\n", results_df.round(3).to_string(index=False))
results_df.to_csv("outputs/model_comparison.csv", index=False)

# 7. Evaluation: classification reports and confusion matrices
for name, y_pred in predictions.items():
    print("\\n" + "=" * 60)
    print(name)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\\n", confusion_matrix(y_test, y_pred))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred, labels=iris.target_names)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
plt.tight_layout(); plt.savefig("outputs/confusion_matrices.png", dpi=180); plt.show()

# 8. Best model selection
best_model_name = results_df.loc[0, "Model"]
best_accuracy = results_df.loc[0, "Accuracy"]
print(f"\\nBest model: {best_model_name}")
print(f"Test accuracy: {best_accuracy:.2%}")
print("Justification: selected because it achieved the highest accuracy on the fixed stratified test set, supported by its classification report and confusion matrix.")
