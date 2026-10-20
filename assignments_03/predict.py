import json
import joblib
import pandas as pd

# Load the saved model and metadata
model = joblib.load("models/weather_classifier.pkl")

with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

# Print key metadata
print("City:", metadata["city"])
print("Features:", metadata["features"])
print("Test accuracy:", metadata["test_accuracy"])

# Build hypothetical weather days using the metadata feature order
days = pd.DataFrame([
    # Clearly good day
    [22, 14, 0, 10],
    # Clearly bad day: heavy precipitation and high wind
    [18, 12, 15, 40],
    # Borderline day: maximum temperature at the upper limit
    [26, 8, 1, 15],
    # Borderline day: near-freezing minimum temperature
    [12, 1, 2, 20],
], columns=metadata["features"])

# Predict labels and probabilities
predictions = model.predict(days)
probabilities = model.predict_proba(days)

for i, row in days.iterrows():
    label = "good" if predictions[i] == 1 else "skip"
    good_probability = probabilities[i][list(model.classes_).index(1)]

    print(f"\nDay {i + 1}:")
    print(row.to_dict())
    print("Prediction:", label)
    print(f"Probability of good: {good_probability:.1%}")
"""
1. The borderline day received a probability of [insert probability] for
   being good for running. A probability near 0.50 indicates uncertainty.
   If the model predicts 0.52, I would treat the prediction cautiously
   because the model is only slightly more likely to classify the day as
   good than skip. I would consider the actual weather conditions before
   recommending a run.

2. If predict.py runs before train_classifier.py, the saved model file
   will not exist, and joblib.load() will raise a FileNotFoundError.
   I could catch this exception and print a message explaining that
   train_classifier.py must be run first to create the model file.

3. For a cloud pipeline, I would change predict.py to accept weather data
   automatically from the pipeline instead of using hardcoded hypothetical
   days.
"""