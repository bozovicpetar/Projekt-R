# Projekt-R
Grupa 272

# Stock Prediction API

API za predviđanje cijena dionica koristeći Linear Regression model.

## Instalacija

### Backend
```bash
cd backend
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## Kako pokrenuti

### 1. Treniranje modela (prvo pokreni ovo!)

```bash
cd backend
python trainModel.py
```

Ovo će:
- Preuzeti podatke za AAPL (Apple) dionice
- Trenirati Linear Regression model
- Spremiti model u `stock_model.pkl`

### 2. Pokretanje API servera

```bash
cd backend
python main.py
```

ili

```bash
cd backend
uvicorn main:app --reload
python -m uvicorn main:app --reload   // ako uvicorn nije globalno instaliran
```

API će biti dostupan na: `http://localhost:8000`

- Dokumentacija: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 3. Pokretanje Frontend aplikacije

```bash
cd frontend
npm run dev
```

Frontend će biti dostupan na: `http://localhost:5173`

**Dostupne komande za frontend:**
```bash
npm run dev       # Start development server
npm run build     # Build za produkciju
npm run preview   # Pregled production build-a
npm run lint      # Provjera koda
```

## Korištenje API-ja

### POST /predict

Predviđa cijenu dionice za sutra i vraća sve podatke potrebne za frontend.

**Request:**
```json
{
  "ticker": "AAPL"
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "current_price": 174.50,
  "predicted_price_tomorrow": 175.23,
  "change_percent": 0.42,
  "is_positive": true,
  "model_metrics": {
    "r2_score": 0.98,
    "mape": 1.02
  },
  "chart_data": [
    {
      "date": "2024-12-01",
      "close": 170.25
    },
    {
      "date": "2024-12-02",
      "close": 171.50
    }
    // ... zadnjih 30 dana
  ],
  "explanation": "Model predviđa rast od 0.42% sutra na temelju volumena i trenda zadnja 2 dana."
}
```

**Response polja:**
- `current_price` - Trenutna cijena dionice (za lijevu karticu)
- `predicted_price_tomorrow` - Predviđena cijena za sutra (za srednju karticu)
- `change_percent` - Postotak promjene (koristi `is_positive` za boju)
- `is_positive` - `true` ako je rast (zeleno), `false` ako je pad (crveno)
- `model_metrics` - R² i MAPE metrike (za desnu karticu)
- `chart_data` - Podaci za graf (zadnjih 30 dana)
- `explanation` - Tekst objašnjenja za ispod grafa

### Primjer s curl:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"ticker\": \"AAPL\"}"
```

## Struktura projekta

```
Projekt-R/
├── backend/
│   ├── main.py                  # FastAPI aplikacija
│   ├── model.py                 # Model logika
│   ├── trainModel.py            # Treniranje modela
│   ├── stock_model.pkl          # Spremljeni model
│   └── requirements.txt         # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.js               # Glavna komponenta
│   │   ├── App.css              # Styling
│   │   ├── main.js              # React init
│   │   └── index.css            # Global CSS
│   ├── index.html               # HTML template
│   ├── package.json             # NPM dependencies (Vite)
│   ├── vite.config.js           # Vite konfiguracija
│   └── .gitignore
│
└── README.md
```

## Napomene

- Model treniran na AAPL podacima, ali može predviđati za bilo koji ticker
- Model koristi 3 features: Close_Lag1, Close_Lag2, Volume_Lag1
- API zahtijeva minimalno 3 dana podataka za predikciju
- API vraća sve podatke potrebne za frontend (cijene, metrike, graf, objašnjenje)
- CORS je omogućen za sve domene (promijeni u produkciji!)
- Nakon treniranja modela, ažuriraj `MODEL_R2` i `MODEL_MAPE` u `main.py` s vrijednostima iz `trainModel.py`
- Frontend koristi Vite za brzi development i optimizovani build