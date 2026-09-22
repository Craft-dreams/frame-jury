"""frame_jury.backends.vlm_qwen3 — local VLM yes/no scorer via Qwen3-VL-8B-Instruct.

Licence: Qwen3-VL-8B-Instruct is Apache-2.0; transformers is Apache-2.0;
bitsandbytes is MIT / Apache-2.0. Recorded in THIRD_PARTY_NOTICES.md.

Technique: VQAScore (Lin et al. 2024, "Evaluating Very Large Language Models Using
Visual Question Answering"). One yes/no question is evaluated via a single forward
pass, measuring P(Yes) / (P(Yes) + P(No)) over the logits of the first output token.
No autoregressive generation or JSON parsing is performed.

Dependencies:
- torch (BSD-3-Clause)
- transformers (Apache-2.0)
- bitsandbytes (MIT/Apache-2.0)
- Pillow (HPND/MIT-compatible)

All heavy dependencies are imported lazily on first load so that the rest of
frame-jury functions without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from frame_jury.backends.base import VlmScorerBackend

# Model ID on Hugging Face Hub (Apache-2.0)
MODEL_ID: str = "Qwen/Qwen3-VL-8B-Instruct"

# Pinned model revision / commit hash.
# Revision measured in the lab (T3b) and verified against the Hugging Face API, 2026-09-22.
# Do NOT invent a hash.
MODEL_REVISION: str = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


class Qwen3VlmScorer(VlmScorerBackend):
    """Qwen3-VL-8B-Instruct 4-bit NF4 yes/no scorer.

    Lazy import: torch, transformers, and bitsandbytes are loaded on first use.
    """

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        device_map: str = "cuda",
        max_pixels: int = 1280 * 28 * 28,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._device_map = device_map
        self._max_pixels = max_pixels

        self._model: Any = None
        self._processor: Any = None
        self._yes_token_ids: set[int] = set()
        self._no_token_ids: set[int] = set()

    def name(self) -> str:
        return "qwen3-vl-8b-instruct"

    def revision(self) -> str:
        return self._revision

    def _load(self) -> None:
        """Lazily initialize model, processor, and token sets."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        load_kwargs: dict[str, Any] = {
            "device_map": self._device_map,
            "quantization_config": quant_config,
        }
        if not self._revision.startswith("TODO"):
            load_kwargs["revision"] = self._revision

        processor_kwargs: dict[str, Any] = {
            "max_pixels": self._max_pixels,
        }
        if not self._revision.startswith("TODO"):
            processor_kwargs["revision"] = self._revision

        self._processor = AutoProcessor.from_pretrained(self._model_id, **processor_kwargs)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_id,
            **load_kwargs,
        ).eval()

        # Resolve Yes / No token IDs from the tokenizer
        tokenizer = self._processor.tokenizer
        self._yes_token_ids = {
            tokenizer.encode(w, add_special_tokens=False)[0]
            for w in ("Yes", "yes", " Yes")
        }
        self._no_token_ids = {
            tokenizer.encode(w, add_special_tokens=False)[0]
            for w in ("No", "no", " No")
        }

    def score_yes_no(self, image_path: str | Path, question: str) -> float:
        """Compute P(Yes) / (P(Yes) + P(No)) at the first answer token."""
        self._load()

        import torch
        from PIL import Image

        image_path = Path(image_path)
        with Image.open(image_path).convert("RGB") as img:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"{question}\nAnswer with only Yes or No."},
                    ],
                }
            ]

            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0, -1].float()

            probs = torch.softmax(logits, dim=-1)
            p_yes = sum(probs[tok_id].item() for tok_id in self._yes_token_ids)
            p_no = sum(probs[tok_id].item() for tok_id in self._no_token_ids)

            total = p_yes + p_no
            if total > 0.0:
                return float(p_yes / total)
            return 0.5
