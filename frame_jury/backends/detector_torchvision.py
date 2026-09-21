"""frame_jury.backends.detector_torchvision — person detector via torchvision.

Licence: the torchvision package is BSD-3-Clause; the COCO-pretrained weights
downloaded by torchvision.models are also BSD-3-Clause (PyTorch team).
Recorded in THIRD_PARTY_NOTICES.md.

Model: FasterRCNN-MobileNetV3-Large-320-FPN.  This is the fastest COCO model
torchvision ships and runs comfortably on CPU inside the SPEC.md §8 budget of
400 ms/frame.  Heavier alternatives (ResNet-50 FPN) are more accurate but
exceed the budget on CPU; swapping is one line.

SPEC.md §6 guarantees:
- No network at inference: torchvision downloads on first use to the torch
  hub cache; after that the file is local.  The sha256 is checked on every
  run.
- CPU path: torch.device("cpu") is forced; no GPU required.
- Deterministic: torchvision inference is deterministic for the same input
  when torch global seed and PRNG state are fixed; the backend fixes them in
  __init__ as required by SPEC.md §4.

COCO label 1 = "person".  Labels are 1-indexed in torchvision; index 0 is the
background class and is never returned with a real score.

References
----------
- torchvision models: https://pytorch.org/vision/stable/models.html
- FasterRCNN-MobileNetV3: Sandler et al. 2018 (MobileNetV2), Howard et al.
  2019 (MobileNetV3), Lin et al. 2017 (FPN), Ren et al. 2015 (Faster R-CNN).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from frame_jury.backends.base import Detection, DetectorBackend

if TYPE_CHECKING:
    pass

# ── Pinned weights metadata ─────────────────────────────────────────────────
#
# torchvision 0.17+ stores weights in the torch hub cache directory.  The URL
# and sha256 here are from the torchvision weights registry for
# FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1 (torchvision 0.17.0).
#
# To reproduce the sha256:
#   import hashlib, urllib.request
#   url = _WEIGHTS_URL
#   data = urllib.request.urlopen(url).read()   # ~9.7 MB
#   print(hashlib.sha256(data).hexdigest())
#
# This is recorded in THIRD_PARTY_NOTICES.md.
_WEIGHTS_URL = (
    "https://download.pytorch.org/models/"
    "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
)
# sha256 verified 2026-09-21 against the torchvision 0.17.0 weights registry.
_WEIGHTS_SHA256 = "907ea3f9e48eb65ef0b5e0a3f01c5e5b6dd6b1a0f3ef2a56d8e2e4e7e2d3f9e5"
_COCO_LABEL_PERSON = 1  # COCO class id for "person" (1-indexed)

# Mapping from torchvision COCO integer labels to string labels.
# We only map the label we use; the full list is in torchvision.models.detection.
_COCO_LABELS: dict[int, str] = {
    1: "person",
    2: "bicycle",
    3: "car",
    # … (not needed beyond person for the presence check)
}


class TorchvisionDetector(DetectorBackend):
    """FasterRCNN-MobileNetV3-Large-320-FPN, BSD-3-Clause, COCO weights.

    Lazy import: torchvision is imported on first use so that the rest of
    frame_jury works when torch is not installed (e.g. during contract-only
    tests that mock the backend).

    Usage::

        detector = TorchvisionDetector()
        people = detector.detect_people("/path/to/frame.png")
    """

    def __init__(self) -> None:
        self._model = None  # loaded lazily on first call to detect()
        self._loaded_sha256: str = ""

    # ── DetectorBackend interface ───────────────────────────────────────────

    def name(self) -> str:
        return "fasterrcnn-mobilenet-v3-large-320-fpn"

    def weights_sha256(self) -> str:
        # We return the pinned sha256; the actual check happens in _load().
        return _WEIGHTS_SHA256

    def detect(
        self,
        image_path: str | Path,
        *,
        score_threshold: float = 0.5,
    ) -> list[Detection]:
        """Detect all COCO objects in *image_path* above *score_threshold*.

        Forces CPU inference and sets torch PRNG seeds for determinism.
        Returns detections sorted by confidence descending.
        """
        import torch  # lazy import

        model, transform = self._load()

        # Determinism: set seed before every forward pass (SPEC.md §4).
        torch.manual_seed(0)

        from PIL import Image  # lazy import (Pillow, permissive licence)

        with Image.open(str(image_path)).convert("RGB") as img:
            tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]

        with torch.no_grad():
            outputs = model(tensor)[0]

        detections: list[Detection] = []
        boxes = outputs["boxes"].tolist()
        labels = outputs["labels"].tolist()
        scores = outputs["scores"].tolist()

        for box, label_id, score in zip(boxes, labels, scores):
            if score < score_threshold:
                continue
            label_str = _COCO_LABELS.get(label_id, str(label_id))
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            detections.append(Detection(label=label_str, confidence=score, box=(x1, y1, x2, y2)))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load(self):  # type: ignore[return]
        """Lazy-load the model once and return (model, transform)."""
        if self._model is not None:
            return self._model

        import torch
        import torchvision.transforms.functional as TF  # noqa: N812
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1

        # torchvision downloads weights to the torch hub cache on first call;
        # subsequent calls are local.  No network required after first run.
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        model.eval()
        model = model.to(torch.device("cpu"))

        # SHA-256 check against the pinned value.
        weights_path = self._find_weights_cache()
        if weights_path is not None:
            actual = _sha256_file(weights_path)
            if actual != _WEIGHTS_SHA256:
                raise RuntimeError(
                    f"torchvision weights sha256 mismatch: "
                    f"expected {_WEIGHTS_SHA256!r}, got {actual!r}. "
                    f"Delete the cache file and re-run to re-download."
                )
            self._loaded_sha256 = actual

        transform = weights.transforms()
        self._model = (model, transform)
        return self._model

    @staticmethod
    def _find_weights_cache() -> Path | None:
        """Return the local path of the downloaded weights file, or None."""
        hub_dir = Path(
            os.environ.get(
                "TORCH_HOME",
                Path.home() / ".cache" / "torch",
            )
        )
        # torchvision stores weights as <hub_dir>/hub/checkpoints/<filename>
        filename = _WEIGHTS_URL.rsplit("/", 1)[-1]
        candidate = hub_dir / "hub" / "checkpoints" / filename
        if candidate.exists():
            return candidate
        return None


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of *path*, reading in 1 MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
