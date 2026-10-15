from dataclasses import dataclass, field, FrozenInstanceError
import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator


# ==========================================
# --- Classes ---
# ==========================================

# Q1
class Thermometer:
    """A thermometer that tracks temperature readings for a specific location."""
    def __init__(self, location: str, readings: list[float] = None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading: float) -> None:
        """Append a new temperature reading."""
        self.readings.append(reading)

    def average(self) -> float | None:
        """Return the mean of the readings, or None if there are none."""
        # Comment: Why does average() need to handle the empty case? What would happen without that check?
        # Without checking for an empty list, sum() / len() would attempt to divide by zero
        # (ZeroDivisionError), crashing the program. Handling it explicitly allows the method
        # to safely return None instead.
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self) -> float | None:
        """Return the highest reading, or None if there are none."""
        if not self.readings:
            return None
        return max(self.readings)

    # Q2
    def __repr__(self) -> str:
        return f"Thermometer(location='{self.location}', n_readings={len(self.readings)}, average={self.average()})"


# Demonstration of Classes Q1 & Q2
my_thermometer = Thermometer("Charlotte", [15.0, 18.5, 22.0, 19.5])
print("--- Classes Q1 & Q2 Output ---")
print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())
print("Direct print:", my_thermometer)
print("List print:", [my_thermometer, Thermometer("Raleigh", [10.0, 12.0])])

# Comment: What Python displays when a class has no __repr__, and why that is unhelpful when debugging:
# Without a custom __repr__, Python displays the default object representation (e.g.,
# <__main__.Thermometer object at 0x7f8b9c...>), which only shows the memory address
# and provides zero insight into the object's actual state, attribute values, or data contents.


# Q3
class TemperatureAlert:
    """Encapsulates a threshold rule to detect temperature breaches."""
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer) -> list[float]:
        """Return a list of every reading in the Thermometer above the threshold."""
        return [r for r in thermometer.readings if r > self.threshold]


# Demonstration of Classes Q3
alert_standard = TemperatureAlert(20.0)
alert_strict = TemperatureAlert(18.0)
print("\n--- Classes Q3 Output ---")
print("Standard threshold breaches:", alert_standard.breaches(my_thermometer))
print("Strict threshold breaches:", alert_strict.breaches(my_thermometer))

# Comment: Why is the threshold stored on TemperatureAlert rather than passed as an argument to breaches()?
# What advantage does that give you if you have twenty thermometers to check?
# Storing the threshold on the instance encapsulates the configuration rule directly on the object.
# If you have twenty thermometers to check against the same threshold, you can reuse a single
# TemperatureAlert instance across all of them without needing to repeatedly pass the threshold value
# into every method call.


# ==========================================
# --- Dataclasses, Type Hints, and Docstrings ---
# ==========================================

# Dataclass Question 1 & 2
@dataclass(frozen=True)
class Station:
    """Represents a physical weather monitoring station.

    Attributes:
        station_id: Unique identifier for the station.
        name: Common name of the station.
        latitude: Geographic latitude coordinate.
        longitude: Geographic longitude coordinate.
        elevation: Elevation above sea level in meters.
    """