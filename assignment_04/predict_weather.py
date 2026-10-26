import os

from weather_model import WeatherClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "weather_classifier.pkl")

DAYS = [
    # clearly good
    {"temperature_2m_max": 22.0, "temperature_2m_min": 12.0,
     "precipitation_sum": 0.0, "wind_speed_10m_max": 8.0},
    # clearly bad
    {"temperature_2m_max": 2.0, "temperature_2m_min": -3.0,
     "precipitation_sum": 18.0, "wind_speed_10m_max": 45.0},
    # borderline: mild but drizzly
    {"temperature_2m_max": 14.0, "temperature_2m_min": 7.0,
     "precipitation_sum": 3.0, "wind_speed_10m_max": 20.0},
    # borderline: dry but cold and breezy
    {"temperature_2m_max": 8.0, "temperature_2m_min": 1.0,
     "precipitation_sum": 0.5, "wind_speed_10m_max": 25.0},
]


def format_day(day, prediction):
    return (
        f"max {day['temperature_2m_max']}C, min {day['temperature_2m_min']}C, "
        f"rain {day['precipitation_sum']}mm, wind {day['wind_speed_10m_max']}km/h "
        f"-> {prediction.label} ({prediction.probability:.2f})"
    )


def main():
    classifier = WeatherClassifier(MODEL_PATH)
    for day, prediction in zip(DAYS, classifier.predict(DAYS)):
        print(format_day(day, prediction))


# Without this guard, importing predict_weather.py from another script to reuse
# format_day would also run main(), loading the model and printing predictions
# as a side effect of the import.
if __name__ == "__main__":
    main()

# 1. The model goes by position, not name. If predict() used the caller's order,
#    wind speed could end up in the temperature slot and nothing would complain.
# 2. The real .pkl might be missing or change when retrained, and then tests fail
#    for no reason related to the code. The fixture model is always the same.
# 3. Ready: predict() takes a whole batch at once. To add: handle None/NaN values.