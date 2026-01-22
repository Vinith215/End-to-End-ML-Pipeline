**End-to-End ML Pipeline: Healthcare Churn Prediction**

This project implements a complete Machine Learning lifecycle for a Medical Imaging SaaS platform. It predicts whether a hospital will "churn" based on the quality and metadata of the DICOM images they upload.

**🏗 Architecture**

1.Ingestion Layer: Processes DICOM files using pydicom to extract image physics metrics (contrast, noise).

2.ETL Layer: Cleans and aggregates data using Polars (Rust-based DataFrame library) for high performance.

3.Model Layer: Trains a LightGBM classifier on hospital usage patterns.

4.Ops Layer: Tracks experiments and metrics using MLflow.

5.Serving Layer: Exposes a REST API via FastAPI & Docker, ready for AWS.

**🚀 Quick Start (Local)**

**1. Install Dependencies**

pip install -r requirements.txt


**2. Generate Synthetic Medical Data**

This script creates dummy .dcm files (DICOMs) and extracts features into a CSV.

python data_generator.py


Output: medical_churn_data.csv and raw_dicoms/ folder.

**3. Train the Model**

Runs the Polars ETL pipeline, trains LightGBM, and logs to MLflow.

python train_pipeline.py


Output: churn_model_lgb.pkl and MLflow runs.

**4. Run the API**

uvicorn app:app --reload


Test prediction:

curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
     -H "Content-Type: application/json" \
     -d '{"avg_img_mean": 450.5, "avg_img_contrast": 12.3, "primary_modality": 0, "scan_count": 120}'


**☁️ AWS Deployment Guide**

To serve this on AWS App Runner or ECS:

1.Build the Image:

docker build -t medical-churn-api .


2.Run Locally (Test):

docker run -p 8000:8000 medical-churn-api


3.Deploy:

Push to AWS ECR.

Create an AWS App Runner service linked to the ECR image.

App Runner handles the SSL and scaling automatically.
