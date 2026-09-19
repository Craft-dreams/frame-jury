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
| _(none yet — filled as each backend lands)_ | | | | |

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
