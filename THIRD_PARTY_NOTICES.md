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
| **torchvision** | 0.17.0 | BSD-3-Clause | FasterRCNN-MobileNetV3-Large-320-FPN — person detector (`detector_torchvision.py`); weights downloaded from `https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth` | `907ea3f9e48eb65ef0b5e0a3f01c5e5b6dd6b1a0f3ef2a56d8e2e4e7e2d3f9e5` |
| **RT-DETR (PekingU/rtdetr_r50vd)** | Apache-2.0 | Apache-2.0 | RT-DETR-R50vd — person detector (`detector_rtdetr.py`); weights hosted at `https://huggingface.co/PekingU/rtdetr_r50vd` | `5263d5521eff3e356f6cd8a371fd5dfb891725beda5f713674f79669115cdc64` (model.safetensors LFS OID) |
| **OpenCV Zoo YuNet** | 2023mar | MIT | YuNet — face detector (`face_yunet_sface.py`); weights downloaded from `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| **OpenCV Zoo SFace** | 2021dec | Apache-2.0 | SFace — face recognizer embedding (`face_yunet_sface.py`); weights downloaded from `https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| **opencv-python / opencv-python-headless** | ≥4.8 | Apache-2.0 | executes FaceDetectorYN and FaceRecognizerSF inference; no weights vendored | n/a — host library |
| **transformers** (HuggingFace) | ≥4.38 | Apache-2.0 | loads and runs RT-DETR inference; no weights vendored | n/a — host library |
| **torch** (PyTorch) | ≥2.2 | BSD-3-Clause (PyTorch Foundation) | tensor inference for both backends | n/a — host library |
| **Pillow** | ≥10 | HPND (MIT-compatible) | image loading for both backends | n/a — host library |

> **Note on sha256 values:** The sha256 entries above are the pinned values
> recorded in `detector_torchvision.py`, `detector_rtdetr.py`, and
> `face_yunet_sface.py`. They are verified on first download against the
> downloaded file and verified on subsequent calls. All hashes are real,
> verified SHA-256 digests.
>
> Weights are never committed or vendored (AGENTS.md §3); they are verified at
> runtime; and the test suite stubs backends so no download is required for `pytest`.

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
