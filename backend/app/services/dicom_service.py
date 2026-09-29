import io
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_dicom(source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read DICOM file or bytes and return:
    (normalized_uint8_array [H, W], sanitized_metadata_dict)
    Handles MONOCHROME1 / MONOCHROME2, RescaleSlope/Intercept, VOI LUT.
    """
    try:
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"DICOM file not found: {p}")
            ds = pydicom.dcmread(str(p), force=True)
        elif isinstance(source, bytes):
            ds = pydicom.dcmread(io.BytesIO(source), force=True)
        else:
            ds = pydicom.dcmread(source, force=True)
    except Exception as e:
        raise ValueError(f"Failed to parse DICOM: {str(e)}")

    if not hasattr(ds, "pixel_array"):
        raise ValueError("DICOM object contains no pixel data.")

    try:
        # Try applying VOI LUT if present for optimal medical contrast
        try:
            arr = apply_voi_lut(ds.pixel_array, ds)
        except Exception:
            arr = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope & Intercept if not already applied
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        if slope != 1.0 or intercept != 0.0:
            arr = arr * float(slope) + float(intercept)

        # Check Photometric Interpretation
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        if photometric == "MONOCHROME1":
            # MONOCHROME1: 0 is white, max is black -> invert to standard medical view
            arr = np.max(arr) - arr

        # Normalize to 0-255 uint8
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max > arr_min:
            arr_norm = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
        else:
            arr_norm = np.zeros_like(arr, dtype=np.uint8)

        # Extract strictly safe, non-identifying technical metadata
        safe_metadata = {
            "modality": str(getattr(ds, "Modality", "CR")),
            "photometric_interpretation": str(photometric),
            "rows": int(getattr(ds, "Rows", arr.shape[0])),
            "columns": int(getattr(ds, "Columns", arr.shape[1])),
            "bits_allocated": int(getattr(ds, "BitsAllocated", 8)),
            "bits_stored": int(getattr(ds, "BitsStored", 8)),
            "pixel_spacing": [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])] if hasattr(ds, "PixelSpacing") else [1.0, 1.0],
            "body_part_examined": str(getattr(ds, "BodyPartExamined", "CHEST")),
            "view_position": str(getattr(ds, "ViewPosition", "PA")),
            "study_description": str(getattr(ds, "StudyDescription", "Chest X-Ray")),
        }

        return arr_norm, safe_metadata

    except Exception as e:
        raise ValueError(f"Error processing DICOM pixel data: {str(e)}")

def dicom_to_pil(source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[Image.Image, Dict[str, Any]]:
    """Convert DICOM directly to PIL Image (RGB) and safe metadata"""
    arr_uint8, meta = read_dicom(source)
    pil_img = Image.fromarray(arr_uint8).convert("RGB")
    return pil_img, meta

def dicom_to_base64_png(source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[str, Dict[str, Any]]:
    """Convert DICOM to Base64 PNG data URL for direct web display"""
    pil_img, meta = dicom_to_pil(source)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}", meta
