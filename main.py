from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

# CORS middleware za frontend
# VAŽNO: Ako frontend radi na drugom portu (npr. localhost:3000), 
# browser će blokirati zahtjeve bez CORS-a. Ako frontend koristi isti port
# ili proxy, možeš zakomentirati ovaj blok.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dozvoljava sve domene (OK za development)
    allow_credentials=True,
    allow_methods=["*"],  # Dozvoljava sve HTTP metode (GET, POST, itd.)
    allow_headers=["*"],  # Dozvoljava sve header-e
)

# Učitaj model
model_path = 'stock_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model datoteka '{model_path}' ne postoji! Prvo pokreni trainModel.py da kreiraš model.")
model = joblib.load(model_path)

# Fiksne metrike modela (izračunate na treniranju)
# Ove vrijednosti možemo koristiti kao "pouzdanost modela" za sve tickere
# Ažuriraj ove vrijednosti nakon što pokreneš trainModel.py!
MODEL_R2 = 0.988  # Ažuriraj nakon treniranja
MODEL_MAPE = 1.02  # Ažuriraj nakon treniranja

class StockRequest(BaseModel):
    ticker: str

@app.post("/predict")
def predict_stock(request: StockRequest):
    ticker = request.ticker.upper()  # Normaliziraj ticker
    
    # 1. Dohvati podatke za graf (zadnjih 30 dana) i za predikciju
    historical_data = yf.download(ticker, period="30d")
    
    if historical_data.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")
    
    # Flatten multi-index columns ako postoje
    if isinstance(historical_data.columns, pd.MultiIndex):
        historical_data.columns = historical_data.columns.get_level_values(0)
    
    # Minimalno 3 reda za lagove
    if len(historical_data) < 3:
        raise HTTPException(status_code=400, detail="Nedovoljno podataka. Trebamo minimalno 3 dana podataka.")
    
    # 2. Pripremi značajke za predikciju
    last_row = historical_data.iloc[-1]
    prev_row = historical_data.iloc[-2]
    
    features = np.array([[
        last_row['Close'],      # Close_Lag1 (današnja cijena za sutra)
        prev_row['Close'],      # Close_Lag2 (jučerašnja cijena)
        last_row['Volume']      # Volume_Lag1 (današnji volumen)
    ]])
    
    # 3. Predikcija
    prediction = model.predict(features)[0]
    change_percent = float((prediction - last_row['Close']) / last_row['Close'] * 100)
    
    # 4. Pripremi podatke za graf (zadnjih 30 dana) - sortirano po datumu
    chart_data = []
    for date, row in historical_data.iterrows():
        # Konvertiraj datum u string format
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)
        
        chart_data.append({
            "date": date_str,
            "close": round(float(row['Close']), 2),  # Zaokruži na 2 decimale
            "is_prediction": False  # Stvarni podaci
        })
    
    # Sortiraj po datumu (ako nije već sortirano)
    chart_data.sort(key=lambda x: x['date'])
    
    # Dodaj predikciju za sutra (narančasta točka)
    last_date = pd.to_datetime(chart_data[-1]['date'])
    tomorrow_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    chart_data.append({
        "date": tomorrow_date,
        "close": round(float(prediction), 2),
        "is_prediction": True  # Predikcija (narančasta točka)
    })
    
    # 5. Generiraj tekst objašnjenja
    direction = "rast" if change_percent > 0 else "pad"
    explanation = (
        f"Model predviđa {direction} od {abs(change_percent):.2f}% sutra "
        f"na temelju volumena i trenda zadnja 2 dana."
    )
    
    return {
        "ticker": ticker,
        "current_price": round(float(last_row['Close']), 2),
        "predicted_price_tomorrow": round(float(prediction), 2),
        "change_percent": round(change_percent, 2),
        "is_positive": change_percent > 0,
        "model_metrics": {
            "r2_score": round(MODEL_R2, 4),
            "mape": round(MODEL_MAPE, 2)
        },
        "chart_data": chart_data,
        "explanation": explanation
    }

@app.get("/")
def root():
    return {
        "message": "Stock Prediction API",
        "endpoints": {
            "POST /predict": "Predviđa cijenu dionice za sutra",
            "GET /health": "Provjera statusa API-ja"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists(model_path)}

# Za pokretanje: u terminalu upiši 'uvicorn main:app --reload'
