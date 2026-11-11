
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# Load the dataset
df = pd.read_csv("weather_classification.csv")

numeric = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[numeric + ["season"]]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric])
X_test_scaled = scaler.transform(X_test[numeric])

print("Scaled training feature means:", X_train_scaled.mean(axis=0))
# Fit on X_train only to prevent data leakage from the test set.

# One-hot encoding the season feature
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

encoded_train = encoder.fit_transform(X_train[["season"]])
encoded_test = encoder.transform(X_test[["season"]])

print("Encoded training shape:", encoded_train.shape)
print("Season features:", encoder.get_feature_names_out(["season"]))
# The season column becomes separate binary columns for its categories.
# The encoder is fit on training data only.

# Preprocessing pipeline with numeric and categorical features
numeric_features = numeric
categorical_features = ["season"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

preprocessing_model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

preprocessing_model.fit(X_train, y_train)

print(
    "Preprocessing pipeline test accuracy:",
    accuracy_score(y_test, preprocessing_model.predict(X_test)),
)
# The pipeline fits preprocessing on training data and applies the same
# transformations to test data before making predictions.

# KNN Q1: Scaled KNN with n_neighbors=5
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

knn_scaled.fit(X_train[numeric], y_train)
knn_scaled_predictions = knn_scaled.predict(X_test[numeric])

print(
    "KNN Q1 test accuracy:",
    accuracy_score(y_test, knn_scaled_predictions),
)
print("KNN Q1 classification report:")
print(classification_report(y_test, knn_scaled_predictions))

# KNN Q2: Compare k values using cross-validation
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results = {}

for k in k_values:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score(knn, X_train[numeric], y_train, cv=5)
    cv_results[k] = scores.mean()

    print(f"k={k:2d}: mean={scores.mean():.3f}  std={scores.std():.3f}")

best_k = max(cv_results, key=cv_results.get)
print(f"Best k based on mean cross-validation accuracy: {best_k}")
# In this run, k=11 has the highest mean cross-validation accuracy (0.942).

# KNN Q3: Compare KNN with and without scaling
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train[numeric], y_train)
knn_raw_predictions = knn_raw.predict(X_test[numeric])

print(
    "KNN Q3 accuracy without scaling:",
    accuracy_score(y_test, knn_raw_predictions),
)
print(
    "KNN Q3 accuracy with scaling:",
    accuracy_score(y_test, knn_scaled_predictions),
)
# Scaling improves test accuracy in this run from 0.925 to 0.958.
# KNN uses distances, so features with larger numeric ranges can dominate
# distances when features are not scaled.

# Logistic Regression using scaled numeric features
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

model.fit(X_train[numeric], y_train)
logreg_predictions = model.predict(X_test[numeric])

print(
    "Logistic Regression test accuracy:",
    accuracy_score(y_test, logreg_predictions),
)
print(
    "Precision:",
    precision_score(y_test, logreg_predictions, pos_label=1),
)
print(
    "Recall:",
    recall_score(y_test, logreg_predictions, pos_label=1),
)
print(
    "F1:",
    f1_score(y_test, logreg_predictions, pos_label=1),
)

logreg = model.named_steps["logreg"]

print("Logistic Regression coefficients:")
for name, coef in zip(numeric, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")

# Positive coefficients push predictions toward class 1 ("good");
# negative coefficients push predictions toward class 0 ("skip").
# In this run, precipitation and wind speed have negative coefficients,
# so higher values push predictions toward "skip."

# Predicted probabilities
probabilities = model.predict_proba(X_test[numeric])[:, 1]
print("First five predicted probabilities:", probabilities[:5].round(2))
# A probability near 0.5 indicates uncertainty about whether the day
# is good for running.

# Logistic Regression confusion matrix
cm = confusion_matrix(y_test, logreg_predictions, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"],
)

disp.plot(colorbar=False)
plt.title("Logistic Regression Confusion Matrix")

# With labels ordered as [skip, good]:
# cm[0, 1] = false positives: predicted good, actually skip
# cm[1, 0] = false negatives: predicted skip, actually good
false_positives = cm[0, 1]
false_negatives = cm[1, 0]

print("False positives:", false_positives)
print("False negatives:", false_negatives)
# In this run, there are 9 false positives and 7 false negatives.
# False positives may matter more for a running app because they
# recommend running when conditions are unsuitable.

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.show()

# Save and reload the Logistic Regression model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/warmup_model.pkl")
print("Model saved.")

loaded = joblib.load("models/warmup_model.pkl")
assert (
    loaded.predict(X_test[numeric]) == model.predict(X_test[numeric])
).all()

print("Predictions match.")

# Without the scaler, raw features would be passed to LogisticRegression,
# potentially producing incorrect predictions because the model was
# trained on scaled data.
