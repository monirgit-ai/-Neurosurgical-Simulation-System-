import os
import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def load_image(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == '.dcm':
        return load_dicom(file_path)
    elif ext in ['.jpg', '.jpeg', '.png']:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        return image, {'FileType': 'Image', 'Filename': os.path.basename(file_path)}
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_dicom(file_path):
    dcm = pydicom.dcmread(file_path)
    image = apply_voi_lut(dcm.pixel_array, dcm)

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        image = np.max(image) - image

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    metadata = {
        'PatientName': str(getattr(dcm, 'PatientName', 'N/A')),
        'StudyDate': getattr(dcm, 'StudyDate', 'N/A'),
        'Modality': getattr(dcm, 'Modality', 'N/A'),
        'InstitutionName': getattr(dcm, 'InstitutionName', 'N/A'),
        'Filename': os.path.basename(file_path),
        'FileType': 'DICOM'
    }

    return image, metadata
