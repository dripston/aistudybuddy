#!/bin/bash
cd frontend
source ../venv/bin/activate
streamlit run app.py --server.port 8501
