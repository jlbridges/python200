import requests
import pandas as pd
import os
import joblib
import platform
import sklearn
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.23,
    "longitude": -80.84,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print(df.shape)
print(df.head())

# Chosen city: Charlotte, North Carolina.
df["good_for_running"] = (
    (df["temperature_2m_max"].between(7, 26))
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

print(
    df["good_for_running"]
    .value_counts()
    .rename(index={1: "Good days", 0: "Skip days"})
)

# The fraction of good days is the number of good days divided by all days.
# If the classes are imbalanced, prioritize precision, recall, and F1 alongside accuracy.

features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Logistic Regression test accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label=1))
print("Recall:", recall_score(y_test, y_pred, pos_label=1))
print("F1:", f1_score(y_test, y_pred, pos_label=1))
print(classification_report(y_test, y_pred))

logreg = model.named_steps["logreg"]
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")

# Negative coefficients push predictions toward "skip" (0).
# The largest negative coefficients indicate the conditions that most strongly push a day toward skip.

# KNN with scaling and cross-validation
k_values = [1, 3, 5, 7, 9]
cv_scores = {}

for k in k_values:
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score(knn_pipeline, X_train, y_train, cv=5)
    cv_scores[k] = scores.mean()
    print(f"k={k}: mean CV accuracy={scores.mean():.3f}")

best_k = max(cv_scores, key=cv_scores.get)

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])

knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

print("Best k:", best_k)
print("KNN test accuracy:", accuracy_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))

# Compare the models using their actual evaluation results.
# For a running app, I would prioritize precision to reduce false positives,
# which could recommend running on a bad-weather day. False negatives mean
# missing a good running day. Both matter, but sending someone out in unsafe
# weather is the greater concern.
# Logistic Regression is small, fast, and its coefficients help explain
# which weather conditions influence predictions.
# It can also provide probability confidence scores using predict_proba().

os.makedirs("models", exist_ok=True)

# Save the fitted Logistic Regression pipeline
joblib.dump(model, "models/weather_classifier.pkl")

# Calculate metrics for the saved model
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, pos_label=1)

metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "features": features,
    "label_rule": (
        "Good day (1) if maximum temperature is between 7 and 26°C, "
        "minimum temperature is at least 0°C, precipitation is below 3 mm, "
        "and maximum wind speed is below 30 km/h; otherwise skip (0)."
    ),
    "city": "Charlotte, North Carolina",
    "test_accuracy": test_accuracy,
    "test_f1": test_f1,
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Model and metadata saved successfully.")