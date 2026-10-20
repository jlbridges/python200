
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

good_index = list(model.classes_).index(1)

for i, row in days.iterrows():
    label = "good" if predictions[i] == 1 else "skip"
    good_probability = probabilities[i][good_index]

    print(f"\nDay {i + 1}:")
    print(row.to_dict())
    print("Prediction:", label)
    print(f"Probability of good: {good_probability:.1%}")


"""
1. The borderline day received a probability of 27.3% for being good
   for running. The model predicted skip. A probability below 0.50
   indicates that the model considers skip more likely than good.
   I would consider the actual weather conditions before recommending
   a run.

2. If predict.py runs before train_classifier.py, the saved model file
   will not exist, and joblib.load() will raise a FileNotFoundError.
   I could catch this exception and print a message explaining that
   train_classifier.py must be run first to create the model file.

3. For a cloud pipeline, I would change predict.py to accept weather data
   automatically from the pipeline instead of using hardcoded hypothetical
   days.
"""
