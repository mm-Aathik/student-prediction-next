# Quick Start Guide

## Setup (One Time)

### 1. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
cd frontend
npm install
```

## Run the Application

### Terminal 1 - Start Backend
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn app:app --reload --port 8000
```

### Terminal 2 - Start Frontend
```bash
cd frontend
npm run dev
```

## Access the App

Open http://localhost:5173 in your browser

## How It Works

1. **Backend (FastAPI)**: 
   - Loads model once at startup (fast!)
   - Runs on port 8000
   - Model stays in memory for instant predictions

2. **Frontend (React Vite)**:
   - Simple, fast UI
   - Runs on port 5173
   - Connects to backend API

## Why This is Better

- ✅ **Fast**: Model loads once, predictions are instant
- ✅ **Simple**: Just 2 commands to run
- ✅ **Reliable**: No timeouts, no hanging
- ✅ **Easy**: Clean separation of frontend/backend

## Troubleshooting

- **Backend won't start**: Make sure you activated the virtual environment
- **Frontend can't connect**: Make sure backend is running on port 8000
- **Model training slow**: First time only, then it's cached


