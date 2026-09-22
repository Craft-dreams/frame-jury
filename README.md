# frame-jury

Judges a generated frame against the contract that asked for it.

Every public tool for this judges an image blind: one compares two faces,
another counts what it recognises, another finds broken limbs, a vision model
gives an opinion. None of them knows what the image was *supposed* to show — so
each one has to guess whether two people in a frame are a mistake.

frame-jury starts from the declaration instead. A shot says which entities must
be visible, what kind each one is, what the character looks like (an approved
reference image), how it is framed, and what the prompt asked for and forbade.
Against that, "two people where one was declared" is not a guess: it is a
contract violation, with the numbers to prove it.

It uses detectors as evidence, not as opinions, and it says `unsure` when the
evidence does not settle the question — a judge that guesses is worse than one
that abstains.

## Status

Milestone M1 provides the corpus builder, strict case and label schemas, a local
keyboard-first labelling page, and deterministic run-level dev/test splits.

Milestone M2 provides the public JSON contract, the presence check
(`duplicated_character`, `extra_person`, `missing_entity`), two detector
backends (torchvision FasterRCNN and RT-DETR), the jury router, and calibration
thresholds as data.

Milestone M3 provides the face identity check (`wrong_identity`), the YuNet face
detector and SFace face embedding backend (`cv2.FaceDetectorYN` and
`cv2.FaceRecognizerSF`), per-framing similarity thresholds with ambiguous band
abstention, and cheap-first routing.

## Build and label the corpus

Requirements: Python 3.12 or newer. M1 uses only the standard library; it needs
no package installation, network access, model weights, or GPU. Run commands
from the repository root.

Build cases from the Content Factory runs. The builder only reads the run tree;
each case stores absolute paths to the existing images and no image is copied:

```powershell
python -m benchmarks.corpus.build `
  --runs C:\path\to\content-factory\build\runs `
  --out benchmarks\cases\cases.jsonl
```

`python -m benchmarks.corpus` is an equivalent shorter entry point. The command
prints the number of runs seen and usable, cases emitted, and every counted skip
reason. A usable run is one with at least one emitted case, regardless of what
other stages it contains. Output is deterministic for an unchanged run tree.

Start the labelling page (the label file is created on the first decision):

```powershell
python -m benchmarks.label `
  --cases benchmarks\cases\cases.jsonl `
  --labels benchmarks\labels\labels.jsonl `
  --labeller your-name
```

The page opens at `http://127.0.0.1:8765/`. Keys `1`–`9` and `0` toggle the ten
defect labels; each button shows its shortcut, with `duplicated_character` on
`1` and `broken_hands` on `2`. `Enter` saves, `C` records `clean`, `U` records
`uncertain`, and `N` focuses notes. The progress header includes the positive
count for every defect.

The labelling queue is deterministic and designed to surface useful positives:
shots declaring two or more characters come first, then one-character shots;
inside each character-count band, close-ups and medium shots precede wide shots,
then other framings. Cases are grouped by run inside each band, with case id as
the final stable tie-breaker. This is only a presentation order: it does not
change case ids or the dev/test split.

Each decision is appended as one JSON line. New lines use label schema 2.0 and
explicitly record `taxonomy_version: "2.0"`. Existing schema-1.0 lines are not
rewritten: an old `broken_anatomy` remains “body or hands, unspecified” for
later scoring, never silently mapped to `broken_hands` or `broken_body`.
Restarting the server validates both current and legacy lines and resumes at the
first unlabelled case. Stop and restart any labelling server that was already
running when you update to this version, so its API and page use the same schema.

Create the deterministic split only after the corpus is built. Cases from one
run always stay on one side:

```powershell
python -m benchmarks.corpus.split `
  --cases benchmarks\cases\cases.jsonl `
  --out benchmarks\cases\splits.json `
  --seed 20260919
```

Run the offline test suite on a clean checkout with:

```powershell
python -m unittest discover -s tests -v
```

## What it does

```text
how many people are in this frame     -> permissive detector, on CPU, ~100 ms
is this the same character as before  -> reference image vs frame
are the hands and limbs broken        -> research track
does the frame show what was asked    -> a vision model, last and dearest
```

Cheap and deterministic checks run first and gate the expensive ones. Thresholds
are calibrated data per camera framing, not numbers typed into the code.

## Judging a frame (M2 presence, M3 identity)

Install the required packages (CPU only; no GPU required):

```powershell
pip install torch torchvision Pillow opencv-python-headless --index-url https://download.pytorch.org/whl/cpu
```

Then call `frame_jury.jury.judge` with a parsed request:

```python
import json
from frame_jury.contract import JuryRequest
from frame_jury.jury import judge

request = JuryRequest.from_json(json.dumps({
    "schema_version": "2.0",
    "image_path": "/path/to/shot-scene-001-004.png",
    "shot": {
        "shot_id": "shot-scene-001-004",
        "framing": "close-up",
        "declared_entities": [
            {
                "entity_id": "char-vigia",
                "kind": "character",
                "display_name": "O vigia",
                "aliases": ["guarda-noturno"],
                "reference_images": ["/path/to/reference-sheet/char-vigia.png"]
            }
        ],
        "staging": {
            "purpose": "O vigia descobre a anotação decisiva.",
            "must_render": [],
            "composition": []
        },
        "positive_prompt": "…",
        "negative_prompt": "…"
    },
    "checks": ["presence", "identity"],
    "budget": "cheap"
}))

verdict = judge(request)   # downloads weights on first run, then runs locally
print(verdict.to_json())
```

The verdict JSON matches the schema in `SPEC.md §4` exactly. The `verdict`
field is `"accept"`, `"reject"` or `"unsure"`. Every finding in `findings`
carries the evidence numbers that produced it (counts, boxes, similarities) and a
`prompt_hint` — a suggestion for the operator, never an automatic edit.

On a first run, detectors download their weights (~10 MB for torchvision FasterRCNN,
~232 KB for YuNet, ~38 MB for SFace). Weights are cached locally and verified by
sha256 on subsequent runs. No GPU and no network access are needed after the
first download.

To use the RT-DETR backend for presence instead:

```python
from frame_jury.backends.detector_rtdetr import RTDetrDetector
verdict = judge(request, detector=RTDetrDetector())
```

To supply a custom face backend for identity:

```python
from frame_jury.backends.face_yunet_sface import YuNetSFaceBackend
verdict = judge(request, face_backend=YuNetSFaceBackend())
```

### Local VLM scoring (budget="full")

For scene-level semantic defects (`duplicated_character`, `missing_entity`), install the optional `vlm` dependency group:

```powershell
pip install .[vlm]
```

Then request `budget="full"` (or include `"vlm_scene"` in `checks`):

```python
request = JuryRequest.from_json(json.dumps({
    ...
    "checks": ["presence", "identity", "vlm_scene"],
    "budget": "full"
}))
verdict = judge(request)
```

The VLM backend (`Qwen3VlmScorer`, Qwen3-VL-8B-Instruct, Apache-2.0) uses 4-bit NF4 quantization and evaluates yes/no questions via VQAScore (one forward pass at the first answer token, no generation). When the backend is unavailable, it records a first-class `abstention` (`verdict="unsure"`) rather than silently passing.

## Calibration

Thresholds are stored in `frame_jury/calibration/defaults.json`, keyed by
camera framing (e.g. `"close-up"`, `"wide shot"`). They are **not fitted**
yet — fitting requires the label campaign (M7). The `"fitted": false` flag
in the JSON is the authoritative marker.

Do not claim precision, recall or F1 numbers against these defaults; they are
documentation values, not evidence (`AGENTS.md §judging-honestly`).

To override thresholds for a specific run (e.g. a custom calibration file):

```python
from frame_jury.calibration.thresholds import CalibrationFile
cal = CalibrationFile.load("/path/to/my-thresholds.json")
verdict = judge(request, calibration=cal)
```

The JSON schema for a calibration file is the same as `defaults.json`.


## Licence

**AGPL-3.0-or-later**, with a commercial licence available.

Using frame-jury inside software you offer to others — including over a network —
places that software under the AGPL. If that does not suit your product, a
commercial licence removes the obligation: see `COMMERCIAL-LICENSE.md`.

The commercial licence covers the `frame_jury` package, whose dependencies and
model weights are deliberately permissive (MIT, BSD, Apache-2.0) so that it can
be relicensed. It does **not** cover `benchmarks/competitors/`, which exists to
measure other projects against ours and imports AGPL-licensed code; those
adapters are AGPL-only and are never part of a release.

Contributions require signing the CLA in `CLA.md`, which is what keeps the dual
licence possible.

Every model the package downloads is recorded in `THIRD_PARTY_NOTICES.md` with
its licence and the checksum of its weights.
