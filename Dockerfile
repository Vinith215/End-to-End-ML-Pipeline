# Base Image: Lightweight Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some C++ libs used by LightGBM/Pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the trained model
COPY app.py .
COPY churn_model_lgb.pkl .

# Expose the API port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]