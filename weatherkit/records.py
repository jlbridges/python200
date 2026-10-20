from dataclasses import dataclass

from .schemas import WeatherResponse

# Design Choice Rationale:
# - WeatherResponse uses Pydantic because it handles external API ingestion,
#   data type coercion, and strict boundary validation.
# - HourlyReading uses a dataclass because it represents internal, trusted
#   domain objects created after successful validation, so it only needs to
#   hold data and doesn't need Pydantic's validation overhead.


@dataclass
class HourlyReading:
    """A single hourly weather observation.

    Attributes:
        timestamp: Observation time as an ISO 8601 string.
        temperature_c: Air temperature at 2 m above ground, in degrees Celsius.
        precipitation_mm: Precipitation total for the hour, in millimetres.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a columnar Open-Meteo response into one record per hour.

    The API returns parallel lists (time, temperature_2m, precipitation).
    This function pairs up the values at each index to build one
    HourlyReading per hour.

    Args:
        response: A validated WeatherResponse whose hourly block holds
            equal-length time, temperature_2m and precipitation lists.

    Returns:
        A list of HourlyReading objects, one per hour, in the same order as
        the input. The list is empty if the response has no hourly entries.
    """
    h = response.hourly
    return [
        HourlyReading(
            timestamp=h.time[i],
            temperature_c=h.temperature_2m[i],
            precipitation_mm=h.precipitation[i],
        )
        for i in range(len(h.time))
    ]