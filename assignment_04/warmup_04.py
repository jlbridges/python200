import json
import os
import pandas as pd
import joblib
from dataclasses import dataclass
import pytest
class ModelInfo:
    def __init__(self, path):
        with open(path) as f:
            self.data = json.load(f)
    def feature_count(self) -> int:
        feature_count = len(self.data['features'])
        return feature_count

path = os.path.join(os.path.dirname(__file__), 'models/weather_classifier_metadata.json')
clf = joblib.load("models/weather_classifier.pkl")
my_model = ModelInfo(path)
total_features = my_model.feature_count()

with open("model.json", "w") as f:
    json.dump({"features": ["a", "b", "c"]}, f)

info = ModelInfo("model.json")
print(info.feature_count())
# Loading in __init__ means the file is read once. Loading in feature_count()
# would re-read and re-parse it on every call.

from dataclasses import dataclass


@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        rain = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c}C, {rain}"


day1 = DayForecast("2023-06-15", 24.1, True)
day2 = DayForecast("2023-06-16", 27.5, False)

print(day1.summary())
print(day2.summary())

# @dataclass wrote __init__ for me, so I didn't have to assign each field by hand.

def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"invalid temperature: {text!r}")


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Without match, pytest.raises(ValueError) passes for any ValueError, so it
# couldn't tell our error from float()'s own "could not convert string" one.