from dataclasses import dataclass
from .schemas import WeatherResponse

# Design Choice Rationale:
# - WeatherResponse uses Pydantic because it handles external API ingestion, data type coercion,
#   and strict boundary validation[cite: 2].
# - HourlyReading uses a dataclass because it represents internal, trusted domain objects
#   transformed after successful validation, prioritizing a lightweight memory footprint[cite: 1, 2].


@dataclass
class HourlyReading:
    """A single hourly weather observation[cite: 1].

    Attributes:
        timestamp: Observation time as an ISO 8601 string[cite: 1].
        temperature_c: Air temperature at 2 m above ground, in degrees Celsius[cite: 1].
        precipitation_mm: Precipitation total for the hour, in millimetres[cite: 1].
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a columnar Open-Meteo response into one record per hour[cite: 1, 2]."""
    h = response.hourly
    return [
        HourlyReading(
            timestamp=h.time[i],
            temperature_c=h.temperature_2m[i],
            precipitation_mm=h.precipitation[i],
        )
        for i in range(len(h.time))
    ]