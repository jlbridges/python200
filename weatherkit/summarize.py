from dataclasses import dataclass

from .records import HourlyReading


@dataclass
class DailySummary:
    """A single hourly weather observation.

    Attributes:
        timestamp: Observation time as an ISO 8601 string (e.g. "2025-01-01T00:00").
        temperature_c: Air temperature at 2 m above ground, in degrees Celsius.
        precipitation_mm: Precipitation total for the hour, in millimetres.
    """

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the day's high and low, in degrees Celsius."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Group hourly readings into daily summaries.

    Attributes:
        min_hours: Minimum number of hourly observations a day needs
            before it is reported.
    """

    def __init__(self, min_hours: int = 24) -> None:
        self.min_hours = min_hours

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Summarize readings per calendar date, dropping incomplete days.

        Args:
            readings: Hourly readings, in any order.

        Returns:
            One DailySummary per date with at least min_hours readings,
            sorted by date.
        """
        groups: dict[str, list[HourlyReading]] = {}
        for r in readings:
            date = r.timestamp[:10]
            if date not in groups:
                groups[date] = []
            groups[date].append(r)

        summaries = []
        for date in sorted(groups):
            day = groups[date]
            if len(day) < self.min_hours:
                continue
            temps = [r.temperature_c for r in day]
            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(temps),
                    temp_min=min(temps),
                    precipitation_sum=sum(r.precipitation_mm for r in day),
                    hours_observed=len(day),
                )
            )
        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return the dates dropped by summarize().

        Args:
            readings: Hourly readings, in any order.

        Returns:
            Dates with fewer than min_hours readings, sorted.
        """
        counts: dict[str, int] = {}
        for r in readings:
            date = r.timestamp[:10]
            counts[date] = counts.get(date, 0) + 1
        return [d for d in sorted(counts) if counts[d] < self.min_hours]
