import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.23,
    "longitude": -80.84,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print(df.shape)
print(df.head())
print(df.describe())
## City chosen: Charlotte, North Carolina (latitude 35.23, longitude -80.84)

# Missing-value check
print(df.isnull().sum())
# Each column shows 0 missing values, so no rows need to be dropped or filled.

# Exploratory plot
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(6, 5))
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"])
plt.title("Daily High vs Low (Charlotte, 2023)")
plt.xlabel("Daily low (°C)")
plt.ylabel("Daily high (°C)")
plt.savefig("outputs/low_vs_high_charlotte.png", dpi=150, bbox_inches="tight")
plt.show()
# The plot shows a strong, roughly linear upward relationship: warmer nights
# go with warmer days, so temperature_2m_min should be the strongest predictor.
corr = df.select_dtypes(include="number").corr()["temperature_2m_max"].sort_values()
print(corr)
# The strongest relationship with the daily high is temperature_2m_min.
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)
print(df["is_summer"].value_counts())
# will help predict high temps


X = df[["temperature_2m_min"]]
y = df["temperature_2m_max"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Slope: {model.coef_[0]:.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² (test): {model.score(X_test, y_test):.4f}")
# it's off by 7 degrees

feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Train R²: {model.score(X_train, y_train):.4f}")
print(f"Test R²:  {model.score(X_test, y_test):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# it helps a lot

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")

#temps high feature raises predictied high. low feature preditcted low. how consistent it is. they are close. it dones't tell me why there's a gap
lims = [min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max())]

plt.figure(figsize=(6, 6))
plt.scatter(y_pred, y_test)
plt.plot(lims, lims, "k--")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted daily high (°C)")
plt.ylabel("Actual daily high (°C)")
plt.savefig("outputs/predicted_vs_actual_high.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------- Task 6: Summary ----------------
# Dataset size: the data has 365 days of Charlotte, North Carolina weather
# from 2023. The 80/20 split puts 292 days in the training set and 73 days
# in the test set.
#
# Model fit: the full model uses four features (temperature_2m_min,
# precipitation_sum, wind_speed_10m_max, and is_summer). Its test R² is
# 0.8805, so it explains about 88% of the variation in daily highs. Its test
# RMSE is 2.5955, which means its predictions are typically off by about
# 2.6°C.
#
# Where the model struggles: [high end / low end / roughly even across the
# range]. [One sentence on what you see in the plot, e.g. "the points stay
# about the same distance from the diagonal from cold days to hot days."]
#
# Largest effect: is_summer has the largest coefficient (about 1.34).
# Holding the other features fixed, a summer day (June, July, or August) is
# predicted to have a high about 1.3°C warmer than a non-summer day.
# temperature_2m_min is smaller per degree (about 0.89), but it varies over
# a much wider range, so over the whole year it moves the predicted high
# more than the summer flag does. precipitation_sum (about -0.20) lowers the
# predicted high by about 0.2°C per mm of rain, and wind_speed_10m_max
# (about -0.04) has almost no effect.
#
# Surprise: I expected is_summer to have a much bigger effect, since summer
# days are so much hotter than winter days. Its effect is only about 1.3°C
# because temperature_2m_min (about 0.89 per °C) already captures most of
# the seasonal warmth.
