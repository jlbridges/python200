from dataclasses import dataclass, field, FrozenInstanceError

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator


# ==========================================
# --- Classes ---
# ==========================================

# Q1
class Thermometer:
    """A thermometer that stores temperature readings (Celsius) for a location."""

    def __init__(self, location: str, readings: list[float] | None = None):
        self.location = location
        # Use None as the default and create a fresh list per instance, so
        # thermometers never share one list.
        self.readings = readings if readings is not None else []

    def add(self, reading: float) -> None:
        """Append one reading."""
        self.readings.append(reading)

    def average(self) -> float | None:
        """Return the mean of the readings, or None if there are none."""
        # Why handle the empty case? Without this check, sum([]) / len([]) is
        # 0 / 0, which raises ZeroDivisionError and crashes the program.
        # Returning None lets the caller handle "no data" safely.
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
        return (
            f"Thermometer(location='{self.location}', "
            f"n_readings={len(self.readings)}, average={self.average()})"
        )


print("--- Classes Q1 ---")
durham = Thermometer("Durham", [10.0, 20.0, 30.0, 40.0])
durham.add(45.0)
print("Average:", durham.average())
print("Hottest:", durham.hottest())

print("\n--- Classes Q2 ---")
raleigh = Thermometer("Raleigh", [10.0, 20.0])
print(durham)
print([durham, raleigh])
# With no __repr__, Python shows the default, e.g.
# <__main__.Thermometer object at 0x7f...>, which is just the class name and a
# memory address. It says nothing about the object's state (location,
# readings), so when debugging or printing a list of objects you can't tell
# them apart.


# Q3
class TemperatureAlert:
    """Flags readings above a configurable threshold."""

    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer) -> list[float]:
        """Return every reading in the thermometer above the threshold."""
        return [r for r in thermometer.readings if r > self.threshold]


print("\n--- Classes Q3 ---")
alert_low = TemperatureAlert(15)
alert_default = TemperatureAlert()  # default threshold is 30.0
print("Threshold 15:", alert_low.breaches(durham))
print("Threshold 30:", alert_default.breaches(durham))
# The threshold is stored on TemperatureAlert because it is configuration of
# the rule, not data about one check. With twenty thermometers you create one
# alert and reuse it (alert.breaches(t) for each), instead of passing the same
# threshold into every call, which is repetitive and easy to get inconsistent.


# ==========================================
# --- Dataclasses, Type Hints, and Docstrings ---
# ==========================================

# Dataclass Q1 and Q2 (Station is frozen, per Q2)
@dataclass(frozen=True)
class Station:
    """A weather monitoring station at a fixed location.

    Attributes:
        station_id: Unique id of the station.
        name: Name of the station.
        latitude: Latitude of the station in degrees.
        longitude: Longitude of the station in degrees.
        elevation: Elevation of the station above sea level.
    """

    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: float


print("\n--- Dataclass Q1 ---")
station_a = Station(123, "test", 40.2, 30.5, 20.4)
station_b = Station(123, "test", 40.2, 30.5, 20.4)
print(station_a == station_b)
# Prints True: @dataclass generates an __eq__ that compares all fields, so two
# Stations with identical values are equal. With the original hand-written
# class there is no __eq__, so == falls back to identity (are they the same
# object?) and would print False.

print("\n--- Dataclass Q2 ---")
try:
    station_a.name = "new_name"
except FrozenInstanceError as e:
    print("Cannot modify:", e)

station_c = Station(456, "test_c", 30.2, 40.5, 60.4)
station_set = {station_a, station_b, station_c}
print(len(station_set))  # 2, because a and b are identical
# Besides immutability, frozen=True (with the generated __eq__) makes the
# class hashable: a __hash__ is generated from the fields. That is why
# Stations can go in a set or be dict keys, and why duplicates collapse. A
# plain mutable dataclass sets __hash__ to None and can't be put in a set.
# It is useful here because a station's identity shouldn't change once created.


# Dataclass Q3
# First attempt:
#
#     @dataclass
#     class StationBatch:
#         region: str
#         stations: list[Station] = []
#
# Error when run:
#     ValueError: mutable default <class 'list'> for field stations is not
#     allowed: use default_factory
#
# Python refuses because a default value is created once and shared by every
# instance. All batches would append to the same list, so adding a station to
# one batch would show up in all the others. default_factory=list calls list()
# for each new instance, giving each its own list.
@dataclass
class StationBatch:
    """A group of stations belonging to one region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda s: s.elevation)


print("\n--- Dataclass Q3 ---")
batch = StationBatch("Piedmont")
print(batch.highest())  # None
batch.add(station_a)
batch.add(station_c)
print(batch.highest())  # station_c has the greater elevation


# ==========================================
# --- Pydantic ---
# ==========================================

# Pydantic Q1 (the model_validator is added in Q4)
class Reading(BaseModel):
    """A single weather reading from a station."""

    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    # Q4
    @model_validator(mode="after")
    def reject_failed_sensor(self):
        """Reject humidity of exactly 0.0 combined with temperature below -40."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError(
                "failed sensor: humidity is 0.0 and temperature_c is below -40"
            )
        return self


print("\n--- Pydantic Q1 ---")
valid = Reading(
    station_id="ABC", timestamp="2026-10-05T12:00", temperature_c=14.8, humidity=50.0
)
print(valid)

# Pydantic Q2
print("\n--- Pydantic Q2 ---")
print("Missing required field:")
try:
    Reading(station_id="ABC", temperature_c=14.8, humidity=50.0)
except ValidationError as e:
    print(e)

print("\ntemperature_c = 150.0:")
try:
    Reading(
        station_id="ABC", timestamp="2026-10-05T12:00", temperature_c=150.0, humidity=50.0
    )
except ValidationError as e:
    print(e)

print('\nhumidity = "very humid":')
try:
    Reading(
        station_id="ABC",
        timestamp="2026-10-05T12:00",
        temperature_c=14.8,
        humidity="very humid",
    )
except ValidationError as e:
    print(e)

print("\nCoercion:")
coerced = Reading(
    station_id="ABC", timestamp="2026-10-05T12:00", temperature_c="21.5", humidity=40
)
print(coerced)
print(type(coerced.temperature_c))
print(type(coerced.humidity))
# Pydantic accepts "21.5" because in its default (lax) mode it tries to
# convert the input to the declared type, and "21.5" is a string that
# represents a valid float. It also turns the int 40 into 40.0. It rejects
# "very humid" because that text can't be interpreted as a number at all.
# The rule: convert if the value sensibly represents the target type,
# otherwise fail.

# Pydantic Q3
print("\n--- Pydantic Q3 ---")
try:
    Reading(station_id="12", temperature_c="two", humidity=40)
except ValidationError as e:
    for error in e.errors():
        print("loc:", error["loc"], "| msg:", error["msg"])
# Three errors were reported (station_id too short, timestamp missing,
# temperature_c not a number). Reporting all at once is more useful because
# you can fix every problem in one pass. Stopping at the first error would
# force a fix-run-fix-run cycle, one problem at a time.

# Pydantic Q4
print("\n--- Pydantic Q4 ---")
ok = Reading(
    station_id="ABC", timestamp="2026-10-05T12:00", temperature_c=-41.0, humidity=5.0
)
print("Valid:", ok)
try:
    Reading(
        station_id="ABC", timestamp="2026-10-05T12:00", temperature_c=-41.0, humidity=0.0
    )
except ValidationError as e:
    print("Rejected:", e)
# Field constraints validate one field at a time, in isolation (e.g. ge/le on
# temperature_c only looks at temperature_c). This rule depends on two fields
# together: humidity == 0.0 AND temperature_c < -40. Each value is fine on its
# own, so only a model-level validator that sees the whole object can check
# the combination.


# ==========================================
# --- pytest ---
# (run with: pytest warmup_01.py -v)
# ==========================================

# pytest Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return celsius * (9 / 5) + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    # pytest.approx is needed because floating-point math is not exact:
    # 37 * (9 / 5) + 32 evaluates to 98.60000000000001, not exactly 98.6, so a
    # plain == fails. approx allows a tiny tolerance.
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)


# pytest Q2
def mean(values: list[float]) -> float:
    """Return the arithmetic mean of values. Raises ValueError if empty."""
    if not values:
        raise ValueError("cannot compute mean of an empty list")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])
    # pytest.raises(ValueError) alone passes for ANY ValueError, including one
    # raised for an unrelated reason (e.g. a bug elsewhere in the function).
    # match= also checks the message, confirming it failed for the expected
    # reason.


# pytest Q3
@pytest.mark.parametrize(
    "values, expected",
    [
        ([5.0], 5.0),
        ([1.0, 2.0, 3.0], 2.0),
        ([1.0, 2.0], 1.5),
        ([-1.0, -2.0, -3.0], -2.0),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)
# Summary line from `pytest warmup_01.py -v`:
# ============================== 6 passed in 0.13s ===============================
#
# One parametrized test is better than four near-identical functions because
# the logic is written once, adding a case is one line of data, and each case
# is still reported (and fails) separately in the output.


# pytest Q4
# I changed 9 / 5 to 9 / 4 in celsius_to_fahrenheit and re-ran. Output:
#
#     >       assert celsius_to_fahrenheit(100) == 212
#     E       assert 257.0 == 212
#     E        +  where 257.0 = celsius_to_fahrenheit(100)
#     FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
#
# (The 0 C check still passed, since 0 * 9 / 4 is 0; the 100 C check failed.)
# Then I fixed the function back to 9 / 5.
#
# pytest showed the exact line that failed, the actual value (257.0), the
# expected value (212), and the function call that produced it. That is more
# useful than a bare "assertion failed" because you can see what went wrong
# immediately, and the numbers hint at the cause, without adding print
# statements or a debugger.