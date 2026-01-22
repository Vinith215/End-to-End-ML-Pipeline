import os
import random
import pandas as pd
from medical_imaging import DicomProcessor

# Configuration
NUM_HOSPITALS = 50
SCANS_PER_HOSPITAL = 10
DATA_DIR = "raw_dicoms"

def generate_synthetic_dataset():
    """
    Simulates a production environment where hospitals upload scans.
    Generates DICOM files and creates a 'usage_log.csv'.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    records = []
    processor = DicomProcessor()

    print(f"Generating synthetic DICOMs for {NUM_HOSPITALS} hospitals...")

    for i in range(NUM_HOSPITALS):
        hosp_id = f"HOSP_{i:03d}"
        
        # Determine if this hospital is a 'churner' (bad stats) or 'loyal'
        is_churner = random.random() < 0.2
        
        for j in range(SCANS_PER_HOSPITAL):
            file_name = os.path.join(DATA_DIR, f"{hosp_id}_scan_{j}.dcm")
            
            # Churners tend to have older machines (CT only) or weird image stats
            modality = "CT" if is_churner and random.random() > 0.3 else random.choice(["CT", "MR"])
            
            # Generate the physical file
            processor.generate_dummy_dicom(file_name, hosp_id, modality)
            
            # Ingest: Read back the file to extract features (Simulating the ETL)
            ds, pixels = processor.read_dicom(file_name)
            features = processor.extract_image_features(ds, pixels)
            
            # Append target label logic (Synthetic Ground Truth)
            # If contrast is low (bad image) -> higher chance of churn
            features['churn_label'] = 1 if is_churner else 0
            records.append(features)

    # Save raw extracted features to CSV
    df = pd.DataFrame(records)
    df.to_csv("medical_churn_data.csv", index=False)
    print("Dataset generated: medical_churn_data.csv")

if __name__ == "__main__":
    generate_synthetic_dataset()