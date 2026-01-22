import pydicom
import numpy as np
import os
from pathlib import Path

class DicomProcessor:
    """
    Handles medical image processing tasks for the ML pipeline.
    """
    
    @staticmethod
    def read_dicom(file_path: str):
        """Reads a DICOM file and returns the dataset and pixel array."""
        try:
            ds = pydicom.dcmread(file_path)
            # Handle standard DICOM pixel extraction
            pixel_array = ds.pixel_array
            return ds, pixel_array
        except Exception as e:
            print(f"Error reading DICOM {file_path}: {e}")
            return None, None

    @staticmethod
    def extract_image_features(ds, pixel_array):
        """
        Extracts technical features that might indicate 'usage quality'.
        Low quality scans might correlate with customer dissatisfaction (churn).
        """
        if pixel_array is None:
            return {}

        # 1. Image Quality Metrics
        mean_intensity = np.mean(pixel_array)
        std_intensity = np.std(pixel_array)
        
        # Simple contrast metric (RMS contrast)
        contrast = np.sqrt(np.mean((pixel_array - mean_intensity)**2))

        # 2. Metadata Extraction (DICOM Tags)
        # 0008,0060 is Modality (CT, MR, XA, etc.)
        modality = ds.get("Modality", "Unknown")
        # 0018,0050 is Slice Thickness
        slice_thickness = float(ds.get("SliceThickness", 0.0))
        # 0008,0080 is Institution Name (simulated as Hospital ID here)
        hospital_id = ds.get("InstitutionName", "Unknown")

        return {
            "hospital_id": hospital_id,
            "modality": modality,
            "slice_thickness": slice_thickness,
            "img_mean": mean_intensity,
            "img_std": std_intensity,
            "img_contrast": contrast
        }

    @staticmethod
    def generate_dummy_dicom(filename: str, hospital_id: str, modality: str = "CT"):
        """
        Generates a valid dummy DICOM file for testing the pipeline.
        """
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import UID

        # Create basic metadata
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
        file_meta.MediaStorageSOPInstanceUID = UID('1.2.3.4.5.6.7')
        file_meta.TransferSyntaxUID = UID('1.2.840.10008.1.2.1')

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Add required DICOM tags
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.Modality = modality
        ds.InstitutionName = hospital_id
        ds.SliceThickness = 2.5
        
        # Generate random "medical" noise image
        img_data = np.random.randint(0, 1000, (512, 512), dtype=np.uint16)
        ds.PixelData = img_data.tobytes()
        ds.Rows, ds.Columns = 512, 512
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        ds.save_as(filename)
        return filename