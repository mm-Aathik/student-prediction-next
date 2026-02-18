@echo off
cd backend
call venv\Scripts\activate
uvicorn app:app --reload --port 8000
