import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse


def make_response(**overrides):
    """Build a small valid response dict, with optional top-level overrides."""
    data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [16.8, 15.1, 13.4],
            "precipitation": [0.0, 0.0, 0.0],
        },
    }
    data.update(overrides)
    return data


def test_valid_response_has_168_hours():
    # A plain relative path like open("weather_raw.json") is resolved against
    # the current working directory, which depends on where pytest is run from
    # (or which run configuration the IDE uses). Building the path from
    # __file__ anchors it to this test file's location instead.
    path = Path(__file__).parent.parent / "weather_raw.json"
    with open(path) as f:
        data = json.load(f)
    response = WeatherResponse.model_validate(data)
    assert len(response.hourly.time) == 168


def test_latitude_out_of_range_raises():
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(make_response(latitude=200.0))


def test_mismatched_list_lengths_raise():
    data = make_response()
    data["hourly"]["temperature_2m"] = [16.8, 15.1]  # one element short
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature_raises():
    data = make_response()
    data["hourly"]["temperature_2m"] = [16.8, None, 13.4]
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)