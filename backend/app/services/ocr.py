"""OCR helpers built on top of pytesseract."""

from __future__ import annotations

from io import BytesIO
from typing import Optional

from PIL import Image

try:
    import pytesseract
    from pytesseract import Output, TesseractNotFoundError
except ImportError:  # pragma: no cover - exercised when pytesseract missing locally
    pytesseract = None
    Output = None
    TesseractNotFoundError = RuntimeError

from app.schemas import OcrResult


class OcrServiceError(RuntimeError):
    """Raised when OCR fails because dependencies are missing or input is invalid."""


def extract_text(image_bytes: bytes) -> OcrResult:
    """Run OCR over the provided image bytes."""
    if pytesseract is None or Output is None:
        raise OcrServiceError(
            "pytesseract is not installed. Install Tesseract and the Python bindings."
        )

    try:
        image = Image.open(BytesIO(image_bytes))
    except Exception as exc:  # noqa: BLE001 - broad to wrap Pillow errors
        raise OcrServiceError(f"Could not open image: {exc}") from exc

    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
    except TesseractNotFoundError as exc:
        raise OcrServiceError(
            "Tesseract is not installed. Follow the backend README to install it."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise OcrServiceError(f"OCR failed: {exc}") from exc

    text_blocks = [block for block in data["text"] if block.strip()]
    text = "\n".join(text_blocks)

    confidences = [float(conf) for conf in data["conf"] if conf not in ("-1", "")]
    confidence: Optional[float] = None
    if confidences:
        confidence = round(sum(confidences) / len(confidences), 2)

    return OcrResult(text=text.strip(), confidence=confidence)
