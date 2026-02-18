# Student Course & University Prediction

Simple ML prediction app using FastAPI (Python) backend and React Vite frontend.

## Quick Start

### Option 1: Run with Docker (Recommended)

Make sure [Docker](https://docs.docker.com/get-docker/) and Docker Compose are installed.

**Start both services:**
```bash
docker compose up --build
```

**Run in background:**
```bash
docker compose up --build -d
```

**Stop services:**
```bash
docker compose down
```

Open http://localhost:5173

---

### Option 2: Run Manually

#### 1. Setup Backend

**Linux/Mac:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Setup Frontend
```bash
cd frontend
npm install
```

#### 3. Run

**Terminal 1 - Backend:**

Linux/Mac: `./run-backend.sh` | Windows: `run-backend.bat` or:
```bash
cd backend && source venv/bin/activate && uvicorn app:app --reload --port 8000
```

**Terminal 2 - Frontend:**

Linux/Mac: `./run-frontend.sh` | Windows: `run-frontend.bat` or:
```bash
cd frontend && npm run dev
```

Open http://localhost:5173

## Project Structure

```
.
├── backend/          # FastAPI Python backend
│   ├── app.py       # Main API server
│   ├── dataset.csv  # Training data
│   └── requirements.txt
├── frontend/        # React Vite frontend
│   ├── src/
│   └── package.json
└── START.md        # Detailed setup guide
```

## Features

- ✅ Fast predictions (model loads once at startup)
- ✅ Simple architecture (FastAPI + React)
- ✅ Clean UI with dropdowns
- ✅ Handles invalid data (NQC values)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — confirms API is running |
| `GET` | `/api/options` | Returns available streams and districts for dropdowns |
| `GET` | `/api/model-info` | Returns model evaluation metrics, hyperparameters, and dataset stats |
| `POST` | `/api/predict` | Predicts top courses with confidence scores based on Z-Score, stream, and district |
| `POST` | `/api/explain` | Returns SHAP-based explanation for a prediction (feature contributions) |

### Request Body (`/api/predict` & `/api/explain`)

```json
{
  "zscore": 1.85,
  "stream": "Physical Science",
  "district": "Colombo"
}
```

### Sample Response (`/api/predict`)

```json
{
  "prediction": "Engineering - University of Moratuwa",
  "confidence": 0.82,
  "top_3": [
    { "course": "Engineering - University of Moratuwa", "confidence": 0.82 },
    { "course": "Engineering - University of Peradeniya", "confidence": 0.10 },
    { "course": "IT - University of Colombo", "confidence": 0.05 }
  ]
}
```
