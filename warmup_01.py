from dataclasses import dataclass, FrozenInstanceError

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

class Thermometer:
    def __init__(self, location, readings = []):
        self.location = location
        self.readings = readings

    def add(self, temps):
        self.readings.append(temps)

    def average(self):
        #return None if 0 to handle divide by 0 errors
        if len(self.readings) == 0:
            return None
        average = sum(self.readings) / len(self.readings)
        return average
    def hottest(self):
        return max(self.readings)
    #add __repr__ to return object attributes so it does not print the memory location of the thermometer object
    def __repr__(self):
        return f"Thermometer(location={self.location}, n_readings={self.readings}, average={self.average()}"

class TemperatureAlert:
    def __init__(self, threshold = 30.0):
        self.threshold = threshold
    def breaches(self, readings):
        temp_breaches = []
        for i in readings:
            if i > self.threshold:
                temp_breaches.append(i)
        return temp_breaches
        #returns a list of temps above the threshold
@dataclass(frozen=True)
class Station:
    """ I'm not a weather expert so I'm not sure what a station is supposed to represent.
        My best guess is it's a weather station, but not sure why the lat/long would be needed

        Attributes:
            station_id: id of the station as int
            name: the name of the station as str
            latitude: the latitude of where the station is? float
            longitude: the longitude of where the station is? float
            elevation: the elevation of the station I guess? float
    """
    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: float


my_test = Thermometer('durham', [10,20,30,40])
my_test.add(45)
breach_test_one = TemperatureAlert(15)
breach_test_two = TemperatureAlert() #default 30
print(breach_test_one.breaches(my_test.readings))
print(breach_test_two.breaches(my_test.readings))
print(my_test)

station_a = Station(123, 'test', 40.2, 30.5, 20.4)
station_b = Station(123, 'test', 40.2, 30.5, 20.4)
#returns True because dataclass generates __eq__
#the original way would require the developer to write an __eq__ method
print(station_a == station_b)
station_c = Station(456, 'test_c', 30.2, 40.5, 60.4)
print(station_c)
try:
    station_a.name = 'new_test'
except FrozenInstanceError as e:
    print("Cannot modify:", e)
station_set = {station_a, station_b, station_c}
print(len(station_set))
#so that objects are not modified after they are created
@dataclass
class StationBatch:
    region: str
    stations: list[Station]
    def add(self, station: Station) -> None:
        self.stations.add(station)
    def highest(self) -> Station | None:
        return max(self.stations, key=lambda s: s.elevation)
test_batch = StationBatch('rest', station_set)
#ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
print(test_batch)
test_batch.add(Station(456, 'test_h', 30.2, 40.5, 60.4))
print(test_batch.highest())


class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_temp_order(self):

        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError(
                f"failed sensor: humidity: ({self.humidity }) temperature_c: ({self.temperature_c})"
            )
        return self



try:
    #valid
   new_reading_1 = Reading (station_id="123", timestamp="10/5/2026", temperature_c=14.8,
                 humidity=50.0)
except ValidationError as e:
    print("\n", e)
try:
    #missing required field
   new_reading_2 = Reading(station_id="123", temperature_c=14.8,
                           humidity=50.0)
except ValidationError as e:
    print("\n", e)
try:
    #a temperature of 150
   new_reading_3 = Reading(station_id="123", timestamp="10/5/2026", temperature_c=150.0,humidity=50.0)
except ValidationError as e:
    print("\n", e)
try:
   new_reading_4 = Reading(station_id="123", timestamp="10/5/2026", temperature_c=14.8,
                           humidity="very humid")
except ValidationError as e:
   print("\n", e)
try:
   new_reading_5 = Reading(station_id="123", timestamp="10/5/2026", temperature_c="21.5",
                           humidity=40)
   print(type(new_reading_5.temperature_c))
   print(type(new_reading_5.humidity))
   #pydantic validates by converting input types to the correct type if possible
except ValidationError as e:
    print("\n", e)
try:
    new_reading_6 = Reading(station_id="13", temperature_c="two",
                           humidity=40)
except ValidationError as e:
    for error in e.errors():
        print("location:", error["loc"])
        print("message: ", error["msg"])
# 3 errors were reported. So the program can run fully and the developer can see which errors occurred and where
try:
    new_reading_7 = Reading(station_id="123",timestamp="10/5/2026", temperature_c=-41.0,
                           humidity=0.0)
except ValidationError as e:
    for error in e.errors():
        print("location:", error["loc"])
        print("message: ", error["msg"])
# this rule needs to check if both are true to indicate a failed sensor. if one is true and one if false then it could be a valid reading

def celsius_to_fahrenheit(celsius: float) -> float:
    """doc string """
    return celsius * 9 / 5 + 32
def test_celsius_to_fatest_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100)  == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

def mean(values: list[float]) -> float:
    if len(values) <=0:
        raise ValueError('values are empty')
    return sum(values) / len(values)
def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])
@pytest.mark.parametrize(
    "values, expected",
    [
        ([5.0], 5.0),
        ([1.0, 2.0, 3.0], 2.0),
        ([1.0, 2.0], 1.5),
        ([-1.0, -2.0, -3.0], -2.0),
        ([-5.0, 5.0], 0.0),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)
#assert 257.0 == 212 +  where 257.0 = celsius_to_fahrenheit(100)