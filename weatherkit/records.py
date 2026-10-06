from dataclasses import dataclass
from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    """A single hourly weather observation.

       Attributes:
           timestamp: Observation time as an ISO 8601 string (e.g. "2025-01-01T00:00").
           temperature_c: Air temperature at 2 m above ground, in degrees Celsius.
           precipitation_mm: Precipitation total for the hour, in millimetres.
       """
    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a columnar Open-Meteo response into one record per hour.

    Args:
        response: A validated WeatherResponse.

    Returns:
        One HourlyReading per hour, in the order returned by the API.
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