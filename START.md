# Quick Start Guide

## Option 1: Docker (Recommended — Works on Any OS)

### Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Run
```bash
docker compose up --build
```

That's it! Open **http://localhost:5173** in your browser.

To stop: press `Ctrl+C` or run `docker compose down`

---

## Option 2: Manual Setup

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Backend Setup

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

### 2. Frontend Setup
```bash
cd frontend
npm install
```

### Run the Application

**Terminal 1 — Start Backend:**

Linux/Mac: `./run-backend.sh` or:
```bash
cd backend
source venv/bin/activate
uvicorn app:app --reload --port 8000
```

Windows: `run-backend.bat` or:
```cmd
cd backend
venv\Scripts\activate
uvicorn app:app --reload --port 8000
```

**Terminal 2 — Start Frontend:**

Linux/Mac: `./run-frontend.sh` | Windows: `run-frontend.bat` or:
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## How It Works

1. **Backend (FastAPI)**: 
   - Loads model once at startup (fast!)
   - Runs on port 8000
   - Model stays in memory for instant predictions

2. **Frontend (React Vite)**:
   - Simple, fast UI
   - Runs on port 5173
   - Connects to backend API

## Troubleshooting

- **Docker: ports in use**: Make sure nothing else is running on ports 8000 or 5173
- **Backend won't start**: Make sure you activated the virtual environment
- **Frontend can't connect**: Make sure backend is running on port 8000
- **Model training slow**: First time only, then it's cached


