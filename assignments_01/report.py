import json
from pathlib import Path

from weatherkit import WeatherResponse, to_readings, DailyAggregator


def main() -> None:
    """Load, validate, convert, aggregate, and print the daily report."""
    path = Path(__file__).parent / "weather_raw.json"
    with open(path) as f:
        data_raw = json.load(f)

    response = WeatherResponse.model_validate(data_raw)
    readings = to_readings(response)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    dropped = aggregator.incomplete_days(readings)

    print(f"{'Date':<12}{'High':>7}{'Low':>7}{'Precip':>9}{'Range':>8}")
    for s in summaries:
        print(
            f"{s.date:<12}{s.temp_max:>7.1f}{s.temp_min:>7.1f}"
            f"{s.precipitation_sum:>9.1f}{s.temp_range():>8.1f}"
        )

    if dropped:
        print(f"WARNING: dropped incomplete days: {', '.join(dropped)}")


# Without this guard, main() would run at import time. If someone imported
# report.py just to reuse a helper function, they would also trigger the file
# load, validation, and printing as a side effect. The guard makes main() run
# only when the file is executed directly.
if __name__ == "__main__":
    main()