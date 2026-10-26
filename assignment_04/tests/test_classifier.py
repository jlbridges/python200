import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import Prediction, WeatherClassifier

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

GOOD_DAY = {
    "temperature_2m_max": 22.0,
    "temperature_2m_min": 12.0,
    "precipitation_sum": 0.0,
    "wind_speed_10m_max": 8.0,
}
BAD_DAY = {
    "temperature_2m_max": 2.0,
    "temperature_2m_min": -3.0,
    "precipitation_sum": 18.0,
    "wind_speed_10m_max": 45.0,
}


@pytest.fixture
def classifier(tmp_path):
    """A WeatherClassifier loaded from a tiny model saved in tmp_path."""
    rows = [
        # temperature_2m_max, temperature_2m_min, precipitation_sum, wind_speed_10m_max
        [20, 10, 0, 8],
        [23, 13, 0, 10],
        [18, 9, 1, 12],
        [25, 14, 0, 6],
        [21, 11, 0, 9],
        [3, -2, 15, 40],
        [1, -4, 20, 50],
        [5, 0, 12, 35],
        [2, -3, 18, 45],
        [4, -1, 10, 42],
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    X = pd.DataFrame(rows, columns=FEATURES)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, labels)

    path = tmp_path / "tiny_model.pkl"
    joblib.dump(model, path)
    return WeatherClassifier(path)


@pytest.mark.parametrize(
    "day, expected_label",
    [(GOOD_DAY, "good"), (BAD_DAY, "skip")],
    ids=["good_day", "bad_day"],
)
def test_label_for_clear_days(classifier, day, expected_label):
    assert classifier.predict([day])[0].label == expected_label


def test_probability_between_0_and_1(classifier):
    for prediction in classifier.predict([GOOD_DAY, BAD_DAY]):
        assert 0 <= prediction.probability <= 1


def test_three_days_return_three_predictions_in_order(classifier):
    predictions = classifier.predict([GOOD_DAY, BAD_DAY, GOOD_DAY])
    assert len(predictions) == 3
    assert all(isinstance(p, Prediction) for p in predictions)
    assert [p.label for p in predictions] == ["good", "skip", "good"]


def test_empty_list_returns_empty_list(classifier):
    assert classifier.predict([]) == []


def test_missing_feature_raises_value_error(classifier):
    incomplete = {k: v for k, v in GOOD_DAY.items() if k != "wind_speed_10m_max"}
    with pytest.raises(ValueError, match="wind_speed_10m_max"):
        classifier.predict([incomplete])


def test_missing_model_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file found"):
        WeatherClassifier(tmp_path / "does_not_exist.pkl")


# Mutation check: I changed classifier.py to always return label="skip".
# test_label_for_clear_days[good_day] and
# test_three_days_return_three_predictions_in_order failed, so they caught it.
# I then restored the original line and all 7 tests passed again.