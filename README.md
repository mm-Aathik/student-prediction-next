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

- `GET /` - Health check
- `GET /api/options` - Get unique streams and districts
- `POST /api/predict` - Make prediction
