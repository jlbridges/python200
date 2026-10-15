import pytest

from weatherkit import DailyAggregator, HourlyReading


@pytest.fixture
def readings():
    """Four readings: three on 2026-04-08 and one on 2026-04-09."""
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 14.0, 0.2),
        HourlyReading("2026-04-08T02:00", 8.0, 0.3),
        HourlyReading("2026-04-09T00:00", 12.0, 0.0),
    ]


def test_grouping_produces_one_summary_per_date(readings):
    summaries = DailyAggregator(min_hours=1).summarize(readings)
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]


def test_temp_max_and_min(readings):
    summary = DailyAggregator(min_hours=1).summarize(readings)[0]
    assert summary.temp_max == 14.0
    assert summary.temp_min == 8.0


def test_precipitation_sum(readings):
    summary = DailyAggregator(min_hours=1).summarize(readings)[0]
    assert summary.precipitation_sum == pytest.approx(0.6)


def test_short_day_is_dropped_and_reported(readings):
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(readings)
    assert [s.date for s in summaries] == ["2026-04-08"]
    assert aggregator.incomplete_days(readings) == ["2026-04-09"]


def test_lowering_min_hours_keeps_the_short_day(readings):
    # With min_hours=3 the 04-09 day (1 reading) is dropped (previous test).
    # With min_hours=1 the same day must be kept.
    aggregator = DailyAggregator(min_hours=1)
    summaries = aggregator.summarize(readings)
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]
    assert aggregator.incomplete_days(readings) == []


@pytest.mark.parametrize(
    "min_hours, expected_days",
    [
        (1, 2),
        (3, 1),
        (4, 0),
    ],
)
def test_min_hours_threshold(readings, min_hours, expected_days):
    summaries = DailyAggregator(min_hours=min_hours).summarize(readings)
    assert len(summaries) == expected_days