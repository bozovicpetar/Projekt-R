from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "stock_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model '{MODEL_PATH}' ne postoji. Prvo pokreni trainModel.py"
    )

model = joblib.load(MODEL_PATH)

MODEL_R2 = 0.988
MODEL_MAPE = 1.02


class StockRequest(BaseModel):
    ticker: str

@app.post("/predict")
def predict_stock(request: StockRequest):
    ticker = request.ticker.upper()

    df = yf.download(ticker, period="30d", progress=False)

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' nije pronađen ili nema podataka"
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if len(df) < 3:
        raise HTTPException(
            status_code=400,
            detail="Nedovoljno podataka (minimalno 3 dana)"
        )

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    X = np.array([[
        float(last_row["Close"]), 
        float(prev_row["Close"]),   
        float(last_row["Volume"])   
    ]])

    prediction = model.predict(X)

    if len(prediction) == 0:
        raise HTTPException(status_code=500, detail="Predikcija nije uspjela")

    prediction = float(prediction.item())


    current_price = float(last_row["Close"])
    change_percent = ((prediction - current_price) / current_price) * 100
    change_percent = float(change_percent)

    chart_data = []
    for date, row in df.iterrows():
        chart_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
            "is_prediction": False
        })

    chart_data.sort(key=lambda x: x["date"])

    last_date = pd.to_datetime(chart_data[-1]["date"])
    tomorrow = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    chart_data.append({
        "date": tomorrow,
        "close": round(prediction, 2),
        "is_prediction": True
    })

    direction = "rast" if change_percent > 0 else "pad"
    explanation = (
        f"Model predviđa {direction} od {abs(change_percent):.2f}% sutra "
        f"na temelju volumena i trenda zadnjih mjesec dana."
    )

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "predicted_price_tomorrow": round(prediction, 2),
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
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH)
    }

# Mount static files (frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        # Serve index.html for all routes (SPA)
        if full_path and not full_path.startswith("api"):
            file_path = os.path.join(STATIC_DIR, full_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
