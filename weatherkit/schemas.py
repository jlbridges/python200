from pydantic import BaseModel, Field, model_validator, ValidationError

class HourlyBlock(BaseModel):

    time: list[str]
    temperature_2m : list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def validate_len(self):
        if not (len(self.time) == len(self.temperature_2m) == len(self.precipitation)):
            raise ValueError(
                "time, temperature_2m and precipitation must have the same length. "
                f"got time: {len(self.time)}, temperature_2m: {len(self.temperature_2m)}, "
                f"precipitation: {len(self.precipitation)}"
            )
        return self
class WeatherResponse(BaseModel):
    """A daily historical weather response from the Open-Meteo archive API."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock

