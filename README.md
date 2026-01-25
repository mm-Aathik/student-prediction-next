# Student Course & University Prediction

Simple ML prediction app using FastAPI (Python) backend and React Vite frontend.

## Quick Start

### 1. Setup Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Setup Frontend
```bash
cd frontend
npm install
```

### 3. Run

**Terminal 1 - Backend:**
```bash
./run-backend.sh
# Or: cd backend && source venv/bin/activate && uvicorn app:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
./run-frontend.sh
# Or: cd frontend && npm run dev
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

- `GET /` - Health check
- `GET /api/options` - Get unique streams and districts
- `POST /api/predict` - Make prediction
