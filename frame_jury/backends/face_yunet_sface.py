"""frame_jury.backends.face_yunet_sface — face detector and embedder via OpenCV Zoo YuNet + SFace.

Licences:
  - YuNet face detector: MIT (opencv_zoo / Shiqi Yu)
  - SFace face recognizer: Apache-2.0 (opencv_zoo / Zhong et al.)
  - opencv-python: Apache-2.0
Recorded in THIRD_PARTY_NOTICES.md.

Models:
  - YuNet: Yu et al., lightweight CNN for face detection with 5 facial landmarks.
    Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  - SFace: Zhong et al., "Can CNNs Be More Robust Than Humans to Face Recognition?",
    source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

SPEC.md §6 guarantees:
  - No network at inference: models are cached locally in ~/.cache/frame_jury/weights/
    (or FRAME_JURY_CACHE_DIR), pinned by sha256.
  - CPU path: runs on CPU via cv2.dnn.
  - Deterministic: same image produces same detections and embeddings.
"""

from __future__ import annotations

import hashlib
import math
import os
import urllib.request
from pathlib import Path

from frame_jury.backends.base import Detection, FaceBackend

# ── Pinned weights metadata ─────────────────────────────────────────────────
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

_SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
_SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
_SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"

_PINNED_SHA256 = f"yunet:{_YUNET_SHA256},sface:{_SFACE_SHA256}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_weights(cache_dir: Path) -> tuple[Path, Path]:
    """Download and verify model weights if not already present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    yunet_path = cache_dir / _YUNET_FILENAME
    sface_path = cache_dir / _SFACE_FILENAME

    for target_path, url, expected_sha256 in [
        (yunet_path, _YUNET_URL, _YUNET_SHA256),
        (sface_path, _SFACE_URL, _SFACE_SHA256),
    ]:
        if not target_path.is_file():
            req = urllib.request.Request(url, headers={"User-Agent": "frame-jury/2.0"})
            with urllib.request.urlopen(req) as resp, target_path.open("wb") as out:
                out.write(resp.read())

        actual_sha256 = _sha256_file(target_path)
        if actual_sha256 != expected_sha256:
            target_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Weights sha256 mismatch for {target_path.name}: "
                f"expected {expected_sha256!r}, got {actual_sha256!r}."
            )

    return yunet_path, sface_path


class YuNetSFaceBackend(FaceBackend):
    """OpenCV Zoo YuNet face detector + SFace face recognizer, CPU.

    Lazy import: ``cv2`` and model weights are only loaded on first inference call.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
        else:
            env_dir = os.environ.get("FRAME_JURY_CACHE_DIR")
            self._cache_dir = (
                Path(env_dir) if env_dir else Path.home() / ".cache" / "frame_jury" / "weights"
            )
        self._detector = None
        self._recognizer = None

    def name(self) -> str:
        return "yunet-sface"

    def weights_sha256(self) -> str:
        return _PINNED_SHA256

    def _load(self):
        if self._detector is not None and self._recognizer is not None:
            return self._detector, self._recognizer

        import cv2

        yunet_path, sface_path = _ensure_weights(self._cache_dir)
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (300, 300),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        return self._detector, self._recognizer

    def _read_image(self, image_path: str | Path):
        import cv2
        import numpy as np

        p = Path(image_path)
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {p}")
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image: {p}")
        return img

    def detect_faces(self, image_path: str | Path) -> list[Detection]:
        detector, _ = self._load()
        img = self._read_image(image_path)
        h, w = img.shape[:2]

        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        if faces is None or len(faces) == 0:
            return []

        detections: list[Detection] = []
        for face in faces:
            x, y, fw, fh = face[:4]
            conf = float(face[14])
            x_min = max(0, int(round(x)))
            y_min = max(0, int(round(y)))
            x_max = min(w, int(round(x + fw)))
            y_max = min(h, int(round(y + fh)))
            detections.append(
                Detection(
                    label="face",
                    confidence=conf,
                    box=(x_min, y_min, x_max, y_max),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def embed(
        self,
        image_path: str | Path,
        box: tuple[int, int, int, int],
    ) -> list[float]:
        import cv2
        import numpy as np

        detector, recognizer = self._load()
        img = self._read_image(image_path)
        h, w = img.shape[:2]

        x_min, y_min, x_max, y_max = box
        bw = max(1, x_max - x_min)
        bh = max(1, y_max - y_min)

        # Detect faces to find 5 facial landmarks for canonical face alignment.
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        best_face = None
        best_iou = 0.0

        if faces is not None and len(faces) > 0:
            for f in faces:
                fx, fy, fw, fh = f[:4]
                ix1 = max(x_min, fx)
                iy1 = max(y_min, fy)
                ix2 = min(x_max, fx + fw)
                iy2 = min(y_max, fy + fh)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = bw * bh + fw * fh - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_face = f

        if best_face is not None and best_iou > 0.3:
            aligned = recognizer.alignCrop(img, best_face)
        else:
            face_box = np.array([x_min, y_min, bw, bh], dtype=np.float32)
            aligned = recognizer.alignCrop(img, face_box)

        feat = recognizer.feature(aligned)
        return feat.flatten().tolist()

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two face embeddings.

        Parameters
        ----------
        a:
            First embedding vector.
        b:
            Second embedding vector.

        Returns
        -------
        float
            Cosine similarity in [-1.0, 1.0], where 1.0 means identical,
            0.0 means orthogonal, -1.0 means opposite, and higher means
            more alike.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        val = dot / (norm_a * norm_b)
        return max(-1.0, min(1.0, float(val)))


# Alias for convenience
YunetSfaceBackend = YuNetSFaceBackend
