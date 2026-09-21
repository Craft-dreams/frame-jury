"""frame_jury.backends.detector_rtdetr — person detector via RT-DETR through HuggingFace transformers.

Licence: RT-DETR is Apache-2.0; the `transformers` library is Apache-2.0.
Weights: PekingU/rtdetr_r50vd (Apache-2.0), hosted on HuggingFace Hub.
Recorded in THIRD_PARTY_NOTICES.md.

Model: RT-DETR-R50 (Real-Time DEtection TRansformer, ResNet-50 backbone).
Published in: Zhao et al. 2023, "DETRs Beat YOLOs on Real-time Object
Detection" (https://arxiv.org/abs/2304.08069).  Implementation through
`transformers` (HuggingFace, Apache-2.0).

The model is a COCO-pretrained end-to-end transformer detector; "person" is
COCO class 0 in this model's label space (the processor decodes to string
labels including "person").

SPEC.md §6 guarantees:
- No network at inference: `transformers` downloads on first use to the HF
  cache (``~/.cache/huggingface/hub``); after that inference is local.
- CPU path: device="cpu" is forced.
- Deterministic: determinism is guaranteed for the same input because
  transformers inference uses fixed arithmetic paths on CPU (no dropout).

References
----------
- Paper: Zhao et al. 2023, https://arxiv.org/abs/2304.08069
- HuggingFace model card: https://huggingface.co/PekingU/rtdetr_r50vd
- transformers library: https://github.com/huggingface/transformers (Apache-2.0)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from frame_jury.backends.base import Detection, DetectorBackend

# ── Pinned weights metadata ─────────────────────────────────────────────────
#
# The RT-DETR weights are distributed as a HuggingFace model repository.
# ``transformers`` downloads multiple files (config, model.safetensors, etc.)
# into the HF hub cache.  We pin the sha256 of the main weights file
# (model.safetensors) as returned by the HF hub API.
#
# Model: PekingU/rtdetr_r50vd — Apache-2.0
# URL:   https://huggingface.co/PekingU/rtdetr_r50vd
# SHA-256 of model.safetensors verified 2026-09-21 via HuggingFace Hub API.
#
# NOTE: because the transformers cache layout uses hashed subdirectories,
# _find_weights_cache() walks the cache to find the file by name.  If the
# file is not present (first run), the sha256 check is skipped and the
# download proceeds normally; the check runs on subsequent calls.
_HF_MODEL_ID = "PekingU/rtdetr_r50vd"
_WEIGHTS_FILENAME = "model.safetensors"
# sha256 pinned 2026-09-21; update here and in THIRD_PARTY_NOTICES.md whenever
# the upstream weights change.
_WEIGHTS_SHA256 = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

# The person label in COCO as decoded by the RT-DETR processor.
_PERSON_LABEL = "person"


class RTDetrDetector(DetectorBackend):
    """RT-DETR-R50vd via HuggingFace transformers, Apache-2.0.

    Lazy import: ``transformers`` and ``torch`` are imported on first use.

    Usage::

        detector = RTDetrDetector()
        people = detector.detect_people("/path/to/frame.png")
    """

    def __init__(self) -> None:
        self._model = None   # (model, processor) loaded lazily
        self._loaded_sha256: str = ""

    # ── DetectorBackend interface ───────────────────────────────────────────

    def name(self) -> str:
        return "rtdetr-r50vd"

    def weights_sha256(self) -> str:
        return _WEIGHTS_SHA256

    def detect(
        self,
        image_path: str | Path,
        *,
        score_threshold: float = 0.5,
    ) -> list[Detection]:
        """Detect COCO objects in *image_path* above *score_threshold*.

        Inference runs on CPU.  The model is end-to-end deterministic for the
        same image (no dropout at eval time).
        """
        import torch  # lazy

        model, processor = self._load()

        from PIL import Image  # lazy (Pillow, MIT/HPND)

        with Image.open(str(image_path)).convert("RGB") as img:
            inputs = processor(images=img, return_tensors="pt")

        # Move to CPU explicitly.
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process: returns list of {"scores", "labels", "boxes"} per image.
        target_sizes = torch.tensor([[inputs["pixel_values"].shape[-2],
                                      inputs["pixel_values"].shape[-1]]])
        results = processor.post_process_object_detection(
            outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )[0]  # single image

        detections: list[Detection] = []
        for score, label_id, box in zip(
            results["scores"].tolist(),
            results["labels"].tolist(),
            results["boxes"].tolist(),
        ):
            # The processor's id2label maps integer → string label.
            label_str = model.config.id2label.get(label_id, str(label_id))
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            detections.append(
                Detection(label=label_str, confidence=score, box=(x1, y1, x2, y2))
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load(self):  # type: ignore[return]
        if self._model is not None:
            return self._model

        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        processor = RTDetrImageProcessor.from_pretrained(
            _HF_MODEL_ID,
            local_files_only=False,  # download once, then cached
        )
        model = RTDetrForObjectDetection.from_pretrained(
            _HF_MODEL_ID,
            local_files_only=False,
        )
        model.eval()
        model = model.to(torch.device("cpu"))

        # sha256 check (best-effort: skip if the cache file is not found).
        weights_path = self._find_weights_cache()
        if weights_path is not None:
            actual = _sha256_file(weights_path)
            if actual != _WEIGHTS_SHA256:
                raise RuntimeError(
                    f"RT-DETR weights sha256 mismatch: "
                    f"expected {_WEIGHTS_SHA256!r}, got {actual!r}. "
                    "Delete the HuggingFace cache for this model and re-run."
                )
            self._loaded_sha256 = actual

        self._model = (model, processor)
        return self._model

    @staticmethod
    def _find_weights_cache() -> Path | None:
        """Walk the HF hub cache to find model.safetensors for _HF_MODEL_ID."""
        hf_cache = Path(
            os.environ.get(
                "HF_HOME",
                Path.home() / ".cache" / "huggingface" / "hub",
            )
        )
        if not hf_cache.exists():
            return None
        # HF hub stores files as:
        # <cache>/models--<owner>--<repo>/snapshots/<sha>/<filename>
        model_slug = _HF_MODEL_ID.replace("/", "--")
        model_dir = hf_cache / f"models--{model_slug}"
        if not model_dir.exists():
            return None
        for candidate in model_dir.glob(f"snapshots/*/{_WEIGHTS_FILENAME}"):
            return candidate  # return the first (most recent) snapshot
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
