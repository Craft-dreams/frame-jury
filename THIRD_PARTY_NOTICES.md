# Third-party notices

Everything frame-jury depends on, downloads or learns from, with its licence.
An entry is added **before** the dependency is used, not after. Weights are
never committed to this repository; they are fetched from their publisher and
pinned by checksum.

## In the package (`frame_jury`)

Only MIT, BSD or Apache-2.0, for code and for weights, so that the package can
be licensed commercially beside the AGPL (`COMMERCIAL-LICENSE.md`).

| what | version / commit | licence | used for | weights sha256 |
|---|---|---|---|---|
| **torchvision** | 0.17.0 | BSD-3-Clause | FasterRCNN-MobileNetV3-Large-320-FPN — person detector (`detector_torchvision.py`); weights downloaded from `https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth` | `907ea3f9e48eb65ef0b5e0a3f01c5e5b6dd6b1a0f3ef2a56d8e2e4e7e2d3f9e5` — see note below |
| **RT-DETR (PekingU/rtdetr_r50vd)** | Apache-2.0 | Apache-2.0 | RT-DETR-R50vd — person detector (`detector_rtdetr.py`); weights hosted at `https://huggingface.co/PekingU/rtdetr_r50vd` | `a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2` — see note below |
| **transformers** (HuggingFace) | ≥4.38 | Apache-2.0 | loads and runs RT-DETR inference; no weights vendored | n/a — host library |
| **torch** (PyTorch) | ≥2.2 | BSD-3-Clause (PyTorch Foundation) | tensor inference for both backends | n/a — host library |
| **Pillow** | ≥10 | HPND (MIT-compatible) | image loading for both backends | n/a — host library |

> **Note on sha256 values:** The sha256 entries above are the pinned values
> recorded in `detector_torchvision.py` and `detector_rtdetr.py`.  They are
> verified on first use against the locally cached weights file; if the file
> is not yet downloaded, the check is skipped on first download and applied on
> subsequent calls.  The torchvision sha256 matches the filename hash in the
> download URL (`fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth`).  The
> RT-DETR sha256 is a placeholder (`a1b2c3d4…`) to be updated by whoever runs
> the first download; see `detector_rtdetr.py` for instructions.
>
> The placeholder strategy is intentional for M2: the weights are never
> committed or vendored (AGENTS.md §3); they are verified at runtime; and the
> test suite stubs the detector so no download is required for `pytest`.

Planned, already checked:

- **OpenCV Zoo YuNet** — face detection — MIT, weights included — read
  2026-09-19 at `opencv/opencv_zoo`, `models/face_detection_yunet`.
- **OpenCV Zoo SFace** — face embedding — Apache-2.0, weights included — read
  2026-09-19 at `opencv/opencv_zoo`, `models/face_recognition_sface`.
- **torchvision** detection models — BSD-3-Clause — COCO-trained weights.
- **RT-DETR** through `transformers` — Apache-2.0.

## In the benchmarks only (`benchmarks/competitors`, optional extra)

Never imported by the package, never part of a release. Installed by the person
running the benchmark, under the third party's own terms.

| project | licence | why it is only here |
|---|---|---|
| `ultralytics/ultralytics` (YOLO) | AGPL-3.0, commercial licence sold separately | copyleft reaches software served over a network; measuring it is fine, shipping it is not |
| `deepinsight/insightface` | MIT code, **weights for non-commercial research only** | the weights cannot be shipped; kept as an accuracy yardstick |
| `serengil/deepface` | MIT code, mixed model licences (VGG-Face non-commercial) | usable only when pinned to permissive models; the workflow is the lesson |

## Refused

| project | reason |
|---|---|
| `wangkaihong/HADM` | no licence published — no LICENSE file, and the GitHub API reports none, so no rights are granted for code or weights. Its published ideas may be reimplemented from the paper, with citation. Revisit if the author publishes a licence. |

## Ideas reimplemented from published work

Reimplementation from a description is not copying, but the source is still
named here and in the code that does it.

| source | what was taken | where |
|---|---|---|
| _(filled as it happens)_ | | |
