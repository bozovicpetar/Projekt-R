import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

def train_and_save_model():
    # 1. Postavke
    ticker = "AAPL" 
    print(f"Dohvaćanje podataka za trening ({ticker})...")
    
    data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
    
    if data is None or data.empty:
        print("Greška: Nema podataka. Provjeri internet vezu.")
        return

    # 2. Priprema podataka
    data.dropna(inplace=True)
    data.sort_index(inplace=True)

    # Feature Engineering
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    data['Volume_Lag1'] = data['Volume'].shift(1)
    data.dropna(inplace=True)

    # Definiranje X i y
    independent_vars = data[['Close_Lag1', 'Close_Lag2', 'Volume_Lag1']].values
    dependent_var = data['Close'].values

    # 3. Train/Test Split (75/25)
    split_idx = int(len(data) * 0.75)
    x_train = independent_vars[:split_idx]
    x_test = independent_vars[split_idx:]
    y_train = dependent_var[:split_idx]
    y_test = dependent_var[split_idx:]
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    print(f"Training period: {data.index[0]} to {data.index[split_idx-1]}")
    print(f"Testing period: {data.index[split_idx]} to {data.index[-1]}\n")

    # 4. Treniranje modela na train setu
    print(f"Treniranje modela na {len(x_train)} dana...")
    model = LinearRegression()
    model.fit(x_train, y_train)

    # 5. Evaluacija na TEST setu 
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    print("Model treniran.")
    print(f"Intercept: {model.intercept_}")
    print(f"Koeficijenti: {model.coef_}")
    print(f"\nMODEL EVALUATION METRICS (na test setu):")
    print("=" * 50)
    print(f"R² Score (Model Confidence): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"\nNAPOMENA: Ažuriraj MODEL_R2 i MODEL_MAPE u main.py s ovim vrijednostima!")

    # 6. Treniranje na CIJELOM datasetu za produkciju (najbolji model)
    print(f"\nTreniranje finalnog modela na CIJELOM datasetu ({len(data)} dana) za produkciju...")
    final_model = LinearRegression()
    final_model.fit(independent_vars, dependent_var)
    
    # 7. Spremanje finalnog modela
    filename = 'stock_model.pkl'
    joblib.dump(final_model, filename)
    print(f"USPJEH: Finalni model je spremljen u datoteku '{filename}'")

if __name__ == "__main__":
    train_and_save_model()
