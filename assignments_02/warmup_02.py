import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)

pred_6 = model.predict([[6]])
pred_18 = model.predict([[18]])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for a low of 6:", pred_6[0])
print("Predicted high for a low of 18:", pred_18[0])

x = np.array([10, 20, 30, 40, 50])
print(x.shape)

x_2d = x.reshape(-1, 1)
print(x_2d.shape)

#it needs a 2-d shape because it needs rows and columns

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(8, 6))
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Low temperature (temp_min)")
plt.ylabel("High temperature (temp_max)")
plt.savefig("outputs/high_vs_low.png", dpi=150, bbox_inches="tight")
plt.show()

# The points form a strong upward linear trend, and they split into two
# visible groups by color: rainy days (red) sit consistently below dry days
# (blue) at the same low temperature. This suggests is_rainy shifts the daily
# high down, so it should help the model.

X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train:", y_train.shape)
print("y_test: ", y_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

print(f"Slope:     {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print(f"RMSE: {rmse:.4f}")
print(f"R² (test): {r2:.4f}")

# Slope (~0.94): for every 1°C increase in the overnight low, the model
# predicts the daily high will be about 0.94°C warmer.

X_full = np.column_stack([temp_min, is_rainy])

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(Xf_train, yf_train)
r2_full = model_full.score(Xf_test, yf_test)

print(f"R² temp_min only:       {r2:.4f}")
print(f"R² temp_min + is_rainy: {r2_full:.4f}")
print(f"Coefficient temp_min:   {model_full.coef_[0]:.4f}")
print(f"Coefficient is_rainy:   {model_full.coef_[1]:.4f}")


print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])
# is_rainy coefficient (~-2.92): holding the low temperature fixed, a rainy
# day is predicted to have a daily high about 2.9°C cooler than a dry day.
yf_pred = model_full.predict(Xf_test)

lims = [min(yf_pred.min(), yf_test.min()), max(yf_pred.max(), yf_test.max())]

plt.figure(figsize=(6, 6))
plt.scatter(yf_pred, yf_test)
plt.plot(lims, lims, "k--", label="Perfect prediction (predicted = actual)")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted high temperature")
plt.ylabel("Actual high temperature")
plt.legend()
plt.savefig("outputs/predicted_vs_actual_warmup.png", dpi=150, bbox_inches="tight")
plt.show()
# A point above the diagonal means the actual high was greater than predicted
# (the model under-predicted). A point below the diagonal means the actual
# high was lower than predicted (the model over-predicted).

## City chosen: Charlotte, North Carolina (latitude 35.23, longitude -80.84)