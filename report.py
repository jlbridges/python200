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

# ==========================================
# Reflection
# ==========================================

# 1. Null tolerance: strict vs. tolerant schema
# HourlyBlock currently requires every temperature to be a float, so a single
# null makes the whole response fail validation (see test_null_temperature_raises).
# Strict: bad data is caught at the boundary, so everything downstream
# (to_readings, DailyAggregator) can trust the values and use max/min/sum
# without None checks. The cost is that one missing hour throws away the
# other 167 good ones.
# Tolerant: declaring temperature_2m as list[float | None] keeps the good
# data, but every downstream step must now handle None. Otherwise max(), min()
# and sum() raise TypeError, or worse, a null silently gets treated as 0 and
# skews the results. The decision about what a missing value means is pushed
# later into the code instead of being made once.
# I chose strict because the aggregation maths needs real numbers, and a loud
# failure is easier to debug than quietly wrong summaries. If real API data
# often contains nulls, the better approach is to allow None in the schema and
# skip or flag those hours in the aggregator on purpose.

# 2. min_hours and the half-day problem
# A day with only a few readings (e.g. the data starts at noon, or the last day
# is cut off) gives a misleading temp_max, temp_min and precipitation_sum,
# because they describe part of a day but look like a full day. min_hours drops
# days with fewer readings than the threshold, and incomplete_days() reports
# which ones were dropped so they aren't lost silently.
# The tradeoff is where to set it. A high value (24) only keeps full days, but
# it also drops a day that is missing a single hour. A low value keeps almost
# everything, including the misleading half days. A count alone also can't tell
# whether the missing hours were the hottest part of the day. Making it a
# parameter lets the caller choose the right balance for their use.

# 3. Packaging benefit (Week 10)
# Putting the code in the weatherkit package with a clear public API in
# __init__.py means other code can use it with `from weatherkit import
# WeatherResponse, DailyAggregator` instead of copying files or editing paths.
# Once it is installable (pip install -e .), the imports work from any folder,
# the tests run against the same package users would install, and the project
# can be versioned and reused in other work. The schema, records and
# aggregation code also stay separated, so each part can change or be tested
# on its own.