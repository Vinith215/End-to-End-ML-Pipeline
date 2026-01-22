import polars as pl
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# --- 1. CLEANING & ETL LAYER (POLARS) ---
def process_data(csv_path: str):
    print("Step 1: Ingesting and Cleaning data with Polars...")
    
    # Lazy evaluation for performance on large datasets
    q = (
        pl.scan_csv(csv_path)
        # Handle Missing Values
        .with_columns(pl.col("slice_thickness").fill_null(0.0))
        # Feature Engineering: Encode Modality (CT=0, MR=1)
        .with_columns(
            pl.when(pl.col("modality") == "CT").then(0)
            .when(pl.col("modality") == "MR").then(1)
            .otherwise(-1).alias("modality_encoded")
        )
        # Aggregation: We predict churn per HOSPITAL, not per image.
        # We aggregate image stats to get hospital-level profile.
        .group_by("hospital_id")
        .agg([
            pl.col("churn_label").max().alias("target"), # If they churned, they churned
            pl.col("img_mean").mean().alias("avg_img_mean"),
            pl.col("img_contrast").mean().alias("avg_img_contrast"),
            pl.col("modality_encoded").mode().first().alias("primary_modality"),
            pl.count().alias("scan_count")
        ])
    )
    
    # Execute query
    df_clean = q.collect()
    print(f"Data processed. Shape: {df_clean.shape}")
    return df_clean.to_pandas()

# --- 2. TRAINING LAYER (LightGBM + MLflow) ---
def train_model():
    # Setup MLflow
    mlflow.set_experiment("Medical_Churn_Prediction")
    mlflow.lightgbm.autolog() # Automatic logging of params and metrics

    # Load & Split
    df = process_data("medical_churn_data.csv")
    X = df.drop(columns=["hospital_id", "target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        print("Step 2: Training LightGBM Model...")
        
        # Create LightGBM Dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Hyperparameters
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9
        }

        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data]
        )

        # Evaluate
        preds_prob = model.predict(X_test)
        preds_class = [1 if x > 0.5 else 0 for x in preds_prob]
        
        acc = accuracy_score(y_test, preds_class)
        f1 = f1_score(y_test, preds_class)
        
        print(f"Training Complete. Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        # Log custom metrics manually if needed (autolog does most)
        mlflow.log_metric("custom_accuracy", acc)

        # Save Model locally for Docker
        joblib.dump(model, "churn_model_lgb.pkl")
        print("Model saved to churn_model_lgb.pkl")

if __name__ == "__main__":
    train_model()