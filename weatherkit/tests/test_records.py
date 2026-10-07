import pytest

from weatherkit import HourlyReading, WeatherResponse, to_readings


@pytest.fixture
def response():
    """A small valid response with distinct values at every index."""
    return WeatherResponse.model_validate({
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [16.8, 15.1, 13.4],
            "precipitation": [0.0, 0.5, 1.2],
        },
    })


def test_to_readings_one_per_hour_in_order(response):
    readings = to_readings(response)
    assert len(readings) == 3
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"


def test_reading_values_match_input_index(response):
    readings = to_readings(response)
    h = response.hourly
    for i, reading in enumerate(readings):
        assert reading.timestamp == h.time[i]
        assert reading.temperature_c == h.temperature_2m[i]
        assert reading.precipitation_mm == h.precipitation[i]


def test_identical_readings_compare_equal():
    a = HourlyReading("2026-04-08T00:00", 16.8, 0.0)
    b = HourlyReading("2026-04-08T00:00", 16.8, 0.0)
    assert a == b