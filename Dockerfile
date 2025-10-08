FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY streamlit_app.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8000", "--server.address=0.0.0.0"]
