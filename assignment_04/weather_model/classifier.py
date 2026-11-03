import os
from dataclasses import dataclass

import joblib
import pandas as pd

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


@dataclass
class Prediction:
    """A classifier's verdict on whether a day is good for running."""

    label: str  # "good" or "skip"
    probability: float  # probability, from 0 to 1, that the day is good for running


class WeatherClassifier:
    """Predicts whether each day's weather is good for running.

    Wraps a saved scikit-learn pipeline loaded from a .pkl file.
    """

    def __init__(self, model_path: str) -> None:
        """Load the saved pipeline from model_path.

        Args:
            model_path: Path to a saved .pkl file.

        Raises:
            FileNotFoundError: If no file exists at model_path.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"No model file found at '{model_path}'. "
                "Check the path and make sure the .pkl file has been saved there."
            )
        self.model = joblib.load(model_path)

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Predict whether each day is good for running.

        Args:
            days: One dictionary per day, each with the keys
                temperature_2m_max, temperature_2m_min, precipitation_sum,
                and wind_speed_10m_max.

        Returns:
            One Prediction per day, in the same order as days. An empty list
            if days is empty.

        Raises:
            ValueError: If any day is missing one or more required features.
        """
        if not days:
            return []

        missing = [f for f in FEATURES if any(f not in day for day in days)]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")

        X = pd.DataFrame([[day[f] for f in FEATURES] for day in days], columns=FEATURES)
        classes = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return [
            Prediction(
                label="good" if cls == 1 else "skip",
                probability=float(prob),
            )
            for cls, prob in zip(classes, probabilities)
        ]