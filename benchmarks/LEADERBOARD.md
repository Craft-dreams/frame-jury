# Benchmark Leaderboard

- **Split**: test
- **Cases**: 148
- **Labelled**: 0
- **Excluded legacy (`broken_anatomy`)**: 0
- **Timestamp**: 2026-09-22T02:28:20.440593+00:00
- **Pinned versions**:
  - Python: 3.12.10
  - numpy: 2.5.3
  - opencv-python: unknown
  - opencv-python-headless: 5.0.0.93
  - pillow: 12.3.0
  - torch: 2.14.0+cpu
  - torchvision: 0.29.0+cpu

| system | defect | precision | recall | F1 | abstain % | ms/frame | peak RAM (py) | licence |
|---|---|---|---|---|---|---|---|---|
| frame-jury | broken_body | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | broken_face | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | broken_hands | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | duplicated_character | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | empty_or_flat | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | extra_person | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | fused_objects | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | garbled_text | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | missing_entity | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | wrong_identity | — | — | — | 10.135 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | wrong_interaction | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| frame-jury | wrong_scale | — | — | — | 0.000 | 233.275 | 135.906 | AGPL-3.0-or-later |
| null | broken_body | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | broken_face | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | broken_hands | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | duplicated_character | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | empty_or_flat | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | extra_person | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | fused_objects | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | garbled_text | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | missing_entity | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | wrong_identity | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | wrong_interaction | — | — | — | 0.000 | 0.002 | 0.154 | n/a |
| null | wrong_scale | — | — | — | 0.000 | 0.002 | 0.154 | n/a |

*peak RAM (py) measures Python allocations only (stdlib tracemalloc in MB).*
