import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ticker = input("Enter the stock ticker symbol: ")
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")

if data is None:
    print("No data found for the given ticker symbol.")
    exit()

print(data.head(), "\n")

# sanitize data

data.dropna(inplace=True)
data.sort_index(inplace=True)
print("Number of duplicates:", data.duplicated().sum(), "\n")

# data features

print(data.info(), "\n")
print(data.describe(), "\n")
print(data.corr(), "\n")

# data visualization

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.4f', cmap='coolwarm')
plt.title(f'{ticker} Feature Correlation Matrix')
plt.tight_layout()
plt.show()

sns.pairplot(data)
plt.show()

# model prep

data['Close_Lag1'] = data['Close'].shift(1)
data['Close_Lag2'] = data['Close'].shift(2)
data['Volume_Lag1'] = data['Volume'].shift(1)
data.dropna(inplace=True)

independent_vars = data[['Close_Lag1', 'Close_Lag2', 'Volume_Lag1']].values
dependent_var = data['Close'].values
dates = data.index

split_idx = int(len(data) * 0.75)
x_train = independent_vars[:split_idx]
x_test = independent_vars[split_idx:]
y_train = dependent_var[:split_idx]
y_test = dependent_var[split_idx:]
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train).ravel()
y_test = np.array(y_test).ravel()

print(f"Training period: {train_dates[0]} to {train_dates[-1]}")
print(f"Testing period: {test_dates[0]} to {test_dates[-1]}\n")

model = LinearRegression()
model.fit(x_train, y_train)

prediction = model.predict(x_test)

# predicted vs actual

comparison = pd.DataFrame({"Predicted": prediction, "Actual": y_test})
print(comparison, "\n")
print(comparison.describe(), "\n")

# model eval

r2 = r2_score(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))
mae = mean_absolute_error(y_test, prediction)
mape = np.mean(np.abs((y_test - prediction) / y_test)) * 100

print("MODEL EVALUATION METRICS")
print("=" * 50)
print(f"R² Score (Model Confidence): {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# line chart
test_df = pd.DataFrame({"Actual": y_test, "Predicted": prediction}, index=test_dates).sort_index()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_df.index, test_df['Actual'], label="Actual", color="tab:blue", linewidth=2)
ax.plot(test_df.index, test_df['Predicted'], label="Predicted", color="tab:orange", linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.set_title(f"{ticker} Actual vs Predicted (Time Series Split)\nR²: {r2:.4f} | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
