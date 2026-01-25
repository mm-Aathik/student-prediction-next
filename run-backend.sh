#!/bin/bash
cd backend
source venv/bin/activate
uvicorn app:app --reload --port 8000


