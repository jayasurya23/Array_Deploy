#!/bin/bash
# This script will start your Streamlit app and make it accessible

# Activate the Python environment if needed (optional, based on your setup)
# source /path/to/your/virtualenv/bin/activate

# Run Streamlit app
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
