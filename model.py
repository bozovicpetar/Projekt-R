import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

# preuzimanje podataka

data = yf.download("AAPL", start='2010-01-01', end='2025-01-01')

if data is None or data.empty:
    raise SystemExit("Data failed to download or is empty.")

print(data.head())

# pripremanje podataka (bez curenja informacija)

data.dropna(inplace=True)
data["Close_next"] = data["Close"].shift(-1)  # predict next-day close
data.dropna(inplace=True)

X = data[["Open", "High", "Low", "Close", "Volume"]]
y = data["Close_next"]

split_idx = int(len(data) * 0.75)
x_train, x_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# izgradnja modela

model = LinearRegression()
model.fit(x_train, y_train)

# evaluacija modela

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# vizualizacija rezultata

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())
plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2)
plt.xlabel('Stvarne vrijednosti (sljedeći dan)')
plt.ylabel('Predviđene vrijednosti (sljedeći dan)')
plt.title('Stvarne vs Predviđene vrijednosti zatvaranja dionica (t+1)')
plt.show()