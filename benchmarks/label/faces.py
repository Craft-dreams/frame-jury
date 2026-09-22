"""Face detection and crop helpers for the identity labelling API."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from frame_jury.backends.base import FaceBackend

_FACE_BOX_CACHE: dict[str, list[list[int]]] = {}


def face_boxes(image_path: str, backend: FaceBackend) -> list[list[int]]:
    """Return stable, left-to-right face boxes, caching by image path."""

    if image_path not in _FACE_BOX_CACHE:
        boxes = [list(map(int, detection.box)) for detection in backend.detect_faces(image_path)]
        boxes.sort(key=lambda box: box[0])
        _FACE_BOX_CACHE[image_path] = boxes
    return [box.copy() for box in _FACE_BOX_CACHE[image_path]]


def crop(image_path: str, box: list[int], *, pad: float = 0.35, size: int = 256) -> bytes:
    """Return a padded, aspect-preserving face crop encoded as PNG."""

    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    with Image.open(image_path) as image:
        left = max(0, int(x0 - pad * width))
        top = max(0, int(y0 - pad * height))
        right = min(image.width, int(x1 + pad * width + 0.999999))
        bottom = min(image.height, int(y1 + pad * height + 0.999999))
        face = image.crop((left, top, right, bottom))
        if face.width <= 0 or face.height <= 0:
            raise ValueError("box must describe a non-empty region inside the image")
        scale = size / max(face.size)
        resized = face.resize(
            (max(1, round(face.width * scale)), max(1, round(face.height * scale))),
            Image.Resampling.LANCZOS,
        )
        output = BytesIO()
        resized.save(output, format="PNG")
    return output.getvalue()
