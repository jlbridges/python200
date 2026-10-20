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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)


df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric + ["season"]]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric])
X_test_scaled = scaler.transform(X_test[numeric])

print(X_train_scaled.mean(axis=0))
# Fit on X_train only to prevent data leakage from the test set.

encoder = OneHotEncoder(sparse_output=False)


encoded = encoder.fit_transform(df[["season"]])
print(encoded.shape)
print(encoder.get_feature_names_out())
# The single season column became 4 columns because there are 4 unique categories, each represented by its own binary column.

numeric_features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
categorical_features = ["season"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))
# The pipeline automatically preprocesses the numeric and categorical features before fitting the model and making predictions.

knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_scaled.fit(X_train[numeric], y_train)


for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train[numeric], y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")
# Choose k=11 because it has the highest mean cross-validation accuracy (0.942).


knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train[numeric], y_train)
print(classification_report(y_test, knn_scaled.predict(X_test[numeric])))
print("Without scaling:", accuracy_score(y_test, knn_raw.predict(X_test[numeric])))
print("With scaling:", accuracy_score(y_test, knn_scaled.predict(X_test[numeric])))
# Scaling helped, improving accuracy from 0.925 to 0.958; scaling matters because KNN uses distances, and features with larger numeric ranges can otherwise dominate those distances.

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
model.fit(X_train[numeric], y_train)
print("Test accuracy:", accuracy_score(y_test, model.predict(X_test[numeric])))
print("Precision:", precision_score(y_test, model.predict(X_test[numeric]), pos_label=1))
print("Recall:", recall_score(y_test, model.predict(X_test[numeric]), pos_label=1))
print("F1:", f1_score(y_test, model.predict(X_test[numeric]), pos_label=1))

logreg = model.named_steps["logreg"]
for name, coef in zip(numeric, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")
    # Higher wind speed and precipitation push most strongly toward "skip," which matches the intuition that windy and rainy weather is less suitable for running.

probabilities = model.predict_proba(X_test[numeric])[:, 1]   # probability of "good"
print(probabilities[:5].round(2))
# A probability near 0.5 indicates the model is uncertain whether the day is good for running.

cm = confusion_matrix(y_test, model.predict(X_test[numeric]))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["skip", "good"])
disp.plot(colorbar=False)
plt.title("KNN Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.show()
# The model made 9 false positives and 7 false negatives. False positives may matter more for a running app because they recommend running when conditions are unsuitable.
joblib.dump(model, "models/warmup_model.pkl")
print("Model saved.")
loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test[numeric]) == model.predict(X_test[numeric])).all()
print("Predictions match.")
# Without the scaler, raw features would be passed to LogisticRegression, potentially producing incorrect predictions because the model was trained on scaled data.