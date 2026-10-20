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

# Fetch weather data for Charlotte, North Carolina.
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

# Define good running days.
df["good_for_running"] = (
    (df["temperature_2m_max"].between(7, 26))
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

class_counts = df["good_for_running"].value_counts().rename(
    index={1: "Good days", 0: "Skip days"}
)
print(class_counts)

good_fraction = df["good_for_running"].mean()
print(f"Fraction of good days: {good_fraction:.1%}")

# This fraction can be reasonable for Charlotte's climate because hot,
# humid summers and cooler winters affect outdoor running conditions.
# The actual fraction depends on the weather thresholds defined above.

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

# Task 3: KNN classifier with scaling and cross-validation.
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
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("Best k:", best_k)
print("KNN test accuracy:", knn_accuracy)
print(classification_report(y_test, knn_predictions))

# Task 4: Logistic Regression.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
logreg_accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression test accuracy:", logreg_accuracy)
print("Precision:", precision_score(y_test, y_pred, pos_label=1))
print("Recall:", recall_score(y_test, y_pred, pos_label=1))
print("F1:", f1_score(y_test, y_pred, pos_label=1))
print(classification_report(y_test, y_pred))

logreg = model.named_steps["logreg"]
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")

# Negative coefficients push predictions toward "skip" (0).
# The largest negative coefficients indicate the conditions that most
# strongly push a day toward skip.

# Task 5: Compare models and choose a model for deployment.
if logreg_accuracy > knn_accuracy:
    print("Logistic Regression had higher test accuracy than KNN.")
elif knn_accuracy > logreg_accuracy:
    print("KNN had higher test accuracy than Logistic Regression.")
else:
    print("Both models had the same test accuracy.")

# For a running app, I would prioritize precision to reduce false positives,
# which could recommend running on a bad-weather day. False negatives mean
# missing a good running day. Both matter, but recommending a run in unsafe
# weather is the greater concern.
# I would deploy Logistic Regression because it is small, fast, and its
# coefficients help explain which weather conditions influence predictions.
# It can also provide probability confidence scores using predict_proba().

# Save the fitted Logistic Regression pipeline.
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/weather_classifier.pkl")

# Calculate metrics for the saved model.
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