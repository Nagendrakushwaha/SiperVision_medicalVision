import io
import base64
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_bounding_boxes(
    image: Image.Image,
    boxes: List[Dict[str, float]],
    box_color: str = "#EF4444",  # Crimson Red
    line_width: int = 4
) -> Tuple[Image.Image, str]:
    """
    Draws ground truth pneumonia bounding boxes onto an image.
    Each box dict contains: {"x": float, "y": float, "width": float, "height": float}
    Returns: (PIL.Image with drawn boxes, base64_png_data_url)
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a clean font or default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, box in enumerate(boxes, start=1):
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("width", 0))
        h = float(box.get("height", 0))

        if w <= 0 or h <= 0:
            continue

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # Draw bounding rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        # Label tag
        label = f"Pneumonia Opacity #{idx} [{int(w)}x{int(h)}]"
        text_bbox = draw.textbbox((x1, max(0, y1 - 20)), label, font=font) if hasattr(draw, "textbbox") else (x1, y1 - 18, x1 + 140, y1)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2], fill=box_color)
        draw.text((x1 + 2, max(0, y1 - 18)), label, fill="#FFFFFF", font=font)

    # Convert to base64
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return canvas, f"data:image/png;base64,{b64_str}"

def format_rsna_boxes(patient_rows) -> List[Dict[str, float]]:
    """
    Extracts valid bounding boxes from RSNA dataframe rows.
    NaN boxes (Normal cases) are ignored.
    """
    boxes = []
    for _, row in patient_rows.iterrows():
        if int(row.get("Target", 0)) == 1 and not np.isnan(row.get("x", np.nan)):
            boxes.append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "width": float(row["width"]),
                "height": float(row["height"])
            })
    return boxes
