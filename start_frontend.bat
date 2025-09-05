@echo off
cd frontend
call ..\venv\Scripts\activate
streamlit run app.py --server.port 8501
pause
