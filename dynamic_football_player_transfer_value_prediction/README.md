# Player Market Value Prediction System

## Tech Stack
- Machine Learning: Random Forest, LightGBM
- Backend: FastAPI
- Frontend: Streamlit
- Language: Python

## Architecture
Streamlit UI → FastAPI API → ML Model

## Steps
1. Train ML models offline
2. Save trained models
3. Serve predictions via FastAPI
4. User interacts via Streamlit UI

## How to Run
### Backend
cd backend  
uvicorn app:app --reload  

### Frontend
cd frontend  
streamlit run streamlit_app.py  