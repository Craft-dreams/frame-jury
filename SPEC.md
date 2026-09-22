# frame-jury — specification

Status: specification, 2026-09-19. Built milestone by milestone under review,
benchmarked against the public projects that solve parts of this, and plugged
into a production pipeline only once it clears the bar in §8.

`AGENTS.md` holds the working rules — licensing boundaries, how the corpus is
borrowed, what a pull request must carry. Read it before writing code.

An upstream pipeline renders the frames and holds the declarations; it consumes
frame-jury through its own port, and neither project imports the other.

---

## 1. Why this exists

A generated frame can be wrong in ways nobody notices until the film is cut: the
same character appears twice, a declared prop is missing, hands have six
fingers, or a character's face does not match their approved reference.
Generators render and hope.

frame-jury judges a frame against the declaration that asked for it. Instead of
evaluating an image blind without knowing what was intended, it starts from the
shot contract:

- `required_visible_entity_ids` — exactly which entities must be in frame;
- each entity's kind (character, object, environment) from the canonical world;
- an approved reference image per entity (`reference_images`);
- the staging prose, the positive and the negative prompt;
- the shot's camera framing (a close-up and a wide shot fail differently).

**That declaration is the product's foundation.** An isolated detector must guess
whether two people in a frame are a defect; frame-jury knows whether the shot
declared one. An identity check needs an anchor; frame-jury evaluates against an
approved reference. So frame-jury is not another detector: it is a *contract
checker* that uses detectors as evidence.

Second advantage: the corpus raw material exists already. The current run-tree
rebuild emits 782 cases from 41 usable runs; its 73 run directories contain 370
reference images. Every emitted frame is joinable to its declaration, but it is
not a labelled corpus until the human campaign records ground truth.

## 2. Scope

**In scope.** Judging one still frame against one shot declaration, returning a
verdict with evidence, and a benchmark harness that measures ours against the
donors on our own corpus.

**Out of scope.** Generating or repairing images; video; deciding what to do
about a verdict (the caller's pipeline or run control decides that); anything that writes
into a run directory.

## 3. Hard licensing rules

The judge ships inside a product that is served over a network. Therefore:

1. The installable package (`frame_jury`) depends only on permissive licences —
   MIT, BSD, Apache-2.0 — for **code and weights**. No AGPL, no
   research-only weights, no "non-commercial" anything.
2. The benchmark may install anything, including AGPL-3.0 (Ultralytics) and
   research-only weights (InsightFace, HADM if it ever gets a licence), but only
   under `benchmarks/competitors/`, as an **optional extra**
   (`pip install frame-jury[competitors]`), never imported by `frame_jury`, and
   never vendored into the repository. A test asserts that no module under
   `frame_jury/` imports a competitor package.
3. Every model the package downloads is recorded in `THIRD_PARTY_NOTICES.md`
   with its licence, its URL and the sha256 of the weights file.
4. HADM publishes no licence at all: no code, no weights, not even for the
   benchmark, until its author publishes one. Its *idea* may be reimplemented
   from the paper.

Known-good starting set: OpenCV Zoo **YuNet** (MIT) and **SFace** (Apache-2.0)
for faces; **torchvision** detection weights (BSD-3-Clause) or **RT-DETR**
through `transformers` (Apache-2.0) for presence and counting.

## 4. The contract

One call, one frame. This JSON is the public interface, and any pipeline
adapter will speak exactly it.

```jsonc
// request
{
  "schema_version": "2.0",
  "image_path": "…/shot-scene-001-004.png",
  "shot": {
    "shot_id": "shot-scene-001-004",
    "framing": "close-up",            // the shot's camera framing
    "other_people_allowed": true,     // optional, default true: background people are fine (§5)
    "expected_people_count": 1,       // optional non-negative integer; defaults to derived character count
    "declared_entities": [
      { "entity_id": "char-vigia", "kind": "character", "display_name": "O vigia",
        "aliases": ["guarda-noturno"], "visual_identity": "Homem grisalho…",
        "relative_scale": "adulto alto", "approximate_dimensions": "1,85 m",
        "reference_images": ["…/reference-sheet/char-vigia.png"] },
      { "entity_id": "obj-caderno", "kind": "object", "display_name": "Caderno",
        "aliases": [], "visual_identity": "Caderno de capa preta…",
        "relative_scale": "cabe em uma mão", "approximate_dimensions": "20 × 14 cm",
        "reference_images": [] }
    ],
    "staging": {
      "purpose": "O vigia descobre a anotação decisiva…",
      "must_render": ["O caderno deve estar aberto"],
      "composition": ["O rosto e o caderno permanecem em foco"]
    },
    "positive_prompt": "…",
    "negative_prompt": "…"
  },
  "checks": ["presence", "identity", "anatomy", "legibility"],  // optional subset
  "budget": "cheap"                   // cheap = local only; full = may call a VLM
}
```

`shot.expected_people_count` optionally states the exact number of people the
frame should contain and overrides the count derived from declared character
entities. This lets one collective character entity such as “dois vizinhos”
require two people, while a non-person entity such as an octopus can require
zero, without inventing entity ids and corrupting identity. When omitted, the
derived character count remains unchanged; `other_people_allowed` still decides
whether people beyond the expected count are defects.

```jsonc
// verdict
{
  "schema_version": "2.0",
  "shot_id": "shot-scene-001-004",
  "verdict": "accept" | "reject" | "unsure",
  "confidence": 0.0,
  "findings": [
    { "check": "presence", "defect": "extra_person",
      "severity": "blocking", "confidence": 0.93,
      "evidence": { "people_detected": 2, "declared_characters": 1,
                    "other_people_allowed": false,
                    "boxes": [[12,40,180,420],[300,44,470,430]] },
      "explanation": "the shot declares one character and allows nobody else, and two people are in frame",
      "prompt_hint": "state that nobody but the declared character is in frame; add people to the negative prompt" }
  ],
  "abstentions": [
    { "check": "identity", "reason": "no_face_in_frame",
      "entity_id": "char-vigia",
      "explanation": "no face detected in frame; cannot verify identity for 'O vigia'" }
  ],
  "measurements": { "faces": 2, "people": 2, "identity_similarity": 0.41 },
  "timings_ms": { "presence": 120, "identity": 90 },
  "detectors": [ { "check": "presence", "backend": "rtdetr-r50", "weights_sha256": "…" } ]
}
```

Rules the verdict must obey:

- **every finding carries its evidence** — the numbers that produced it, not
  only a label. A verdict nobody can audit is worthless in an automated pipeline,
  where a failed gate must preserve enough evidence to diagnose it;
- **`unsure` is a first-class answer, and an abstention is not a finding.**
  A defect is something wrong with the frame; an abstention is something the
  judge could not determine (e.g. no face found, ambiguous similarity, or fewer
  faces than characters). Abstentions are recorded in a first-class `abstentions`
  field. Any `blocking` finding produces `reject`; otherwise any abstention
  produces `unsure`; otherwise `accept`. A `warning` finding on its own does not
  produce `unsure` — it is a real but non-blocking defect, and an `accept` may
  carry warnings;
- **identity assignment is bipartite, one-to-one and optimal.** Characters are
  assigned to detected faces by maximising total similarity (Hungarian algorithm),
  not greedily or independently. Each declared character is judged against its
  assigned face only. Surplus characters left unassigned when there are fewer
  faces than characters result in an abstention, not a `wrong_identity`;
- **`prompt_hint` is a suggestion for a human or a station, never an edit.**
  frame-jury never rewrites a prompt;
- deterministic: the same image and declaration give the same verdict, with
  pinned weights and fixed seeds.

## 5. Defect taxonomy (v2.1)

| defect | check | how it is decided |
|---|---|---|
| `duplicated_character` | identity | the same declared character appears more than once — decided by face identity, never by counting people (not implemented yet; see below) |
| `extra_person` | presence | more people than the declared characters, in a shot whose declaration sets `other_people_allowed: false` |
| `missing_entity` | presence | a declared character/object not found |
| `wrong_identity` | identity | face embedding distance to the entity's reference above threshold |
| `broken_hands` | anatomy | hands or fingers have the wrong count, are fused or are malformed; this common defect may need a dedicated detector |
| `broken_body` | anatomy | limbs, joints or poses are impossible (the face is `broken_face` since v2.1) |
| `broken_face` | anatomy | a face is deformed, melted, asymmetric beyond style, or has wrong or misplaced features |
| `wrong_scale` | contract | an entity has the wrong size or proportion relative to the scene, another entity or its bible declaration |
| `fused_objects` | anatomy | two distinct objects/entities have collapsed into one connected body or lost their boundary |
| `wrong_interaction` | interaction | a character handles an object in a physically impossible or wrong way: the wrong grip, holding it by the wrong part, a hand passing through it, an object floating beside the hand meant to hold it |
| `garbled_text` | legibility | text-like regions that are not words |
| `empty_or_flat` | legibility | blank, near-uniform or detail-starved frame |

`wrong_scale` is contract-checkable, not a matter of taste: the Production
Bible already declares `relative_scale` and `approximate_dimensions` for an
entity, and the case carries both for comparison. `fused_objects` is about two
entities losing their boundary, unlike `wrong_scale`, where the entity remains
distinct but has the wrong size. `broken_body` is malformed human anatomy,
unlike `wrong_scale`, which compares an otherwise recognizable entity's size
against its declared context.

### Background people are not a defect

Decided with the operator on 2026-09-22: **a background figure is never a
defect, unless the shot says there is nobody else in it.** A library has
readers and a street has pedestrians; a frame that shows them is correct.
Image prompts often state this ("any other people the place calls for appear
only as ordinary background figures"), so a judge that punished it would
contradict the prompt that asked for it.

The first reading of the labels forced it. Counting every detected person
against the declared cast produced **31 false `duplicated_character` findings
against 1 true one** on 67 labelled frames, and 29 of the 35 frames labelled
clean were flagged — a child and an octopus in a library, with four readers
behind them, detected at 0.93 and above.

So the rule is structured, never read from prose:

- `shot.other_people_allowed` is a boolean in the request, **default `true`**.
  Only an explicit `false` makes surplus people a defect. frame-jury never infers
  it from `positive_prompt`, `negative_prompt` or staging text — a prompt is a
  projection, not the declaration.
- Counting cannot tell a duplicate from a figurant, so **presence no longer
  emits `duplicated_character`.** With `other_people_allowed: false`, any person
  beyond the declared characters is `extra_person`, whether the shot declares
  none or several. With `true`, surplus people are measured and reported in
  `measurements`, and are not a finding.
- `duplicated_character` belongs to identity: two faces that both match one
  declared character's reference. It waits until identity's threshold is fitted
  — on the same 67 labels, correct pairs of stylised characters score only
  0.25–0.49 against their reference sheets, so identity is not yet reliable
  enough to claim a duplicate.

Until upstream shot declarations carry a structured "nobody else here"
fact, current corpus cases have `other_people_allowed: true`, and `extra_person`
will not fire on them. That is honest: the fact does not exist upstream
yet, and inventing it from prompt prose is exactly what this rule forbids.

### What v2.1 added, and why

Both came from the operator labelling the first 67 frames and finding nowhere
to put what they saw.

- **`broken_face`**, split out of `broken_body`. A deformed face was already a
  defect, inside `broken_body`'s definition, but no labeller could find it
  behind a button reading "broken body". It is also a different check: faces
  are already located by the face backend identity uses, so a face-specific
  check has a natural home that a whole-body pose check does not.
- **`wrong_interaction`**, new. A hand holding a sword by the blade, or a cup
  hovering beside the fingers, is none of the v2 defects: the hand itself may be
  anatomically fine (`broken_hands`), and the two objects keep their boundary
  (`fused_objects`). It is judged by no cheap check; it is a research track and
  a candidate for the `full` budget's VLM.

A taxonomy-2.0 `broken_body` label means "body or face, unspecified", exactly as
legacy `broken_anatomy` means "body or hands": it is excluded from scoring
`broken_face`, and scoring must not reinterpret it. (None exists at the time of
the split.)

The first detector release must ship `presence` and `identity` well. `anatomy`,
`wrong_scale` and `legibility` are research tracks whose baselines the benchmark
measures first.

## 6. Architecture

```text
frame_jury/
  contract.py        request/verdict dataclasses, JSON in and out, versioned
  jury.py            the router: runs checks in order, stops early, merges findings
  checks/
    presence.py      counting people/objects against the declaration
    identity.py      reference-vs-frame face and appearance similarity
    anatomy.py       hands/limbs plausibility
    legibility.py    text and flatness
  backends/          one adapter per model, all permissive
    detector_rtdetr.py, detector_torchvision.py
    face_yunet_sface.py
    vlm_openai_compatible.py      # optional, only when budget = "full"
  calibration/       thresholds as data, per framing, fitted on the corpus
  evidence.py        the ledger every verdict writes
benchmarks/
  corpus/            builder that joins our runs into cases (no image is copied
                     into the repo; it points at paths)
  labels/            human labels, JSONL, versioned
  competitors/       YOLO / InsightFace / VLM-only wrappers (optional extra)
  run.py             the harness; writes a leaderboard
```

Core architectural principles:

1. **Cheap gates first.** Counting runs on CPU in ~100 ms; a VLM call costs money
   and seconds. The router runs deterministic checks first and only escalates
   what they cannot settle.
2. **Backends are replaceable.** A check states what it needs; a backend
   provides it. Swapping RT-DETR for something better must touch one file.
3. **Thresholds are data, not code.** They are fitted per framing on the corpus
   and stored in `calibration/`, with the fit reproducible from the labels.
4. **No network at inference.** Weights are downloaded once, pinned by sha256,
   cached locally. The optional VLM backend is the single exception, and it is
   off by default.

## 7. Corpus and ground truth

The corpus builder reads external generator runs and emits one case per rendered
frame:

```jsonc
{ "schema_version": "2.0",
  "case_id": "run-suspense-stakeout-20260919-181850/shot-scene-001-004",
  "image_path": "…",
  "shot": {
    "shot_id": "shot-scene-001-004", "framing": "close-up",
    "declared_entities": [
      { "entity_id": "char-vigia", "kind": "character", "display_name": "O vigia",
        "aliases": ["guarda-noturno"], "visual_identity": "Homem grisalho…",
        "relative_scale": "adulto alto", "approximate_dimensions": "1,85 m",
        "reference_images": ["…/reference-sheet/char-vigia.png"] }
    ],
    "staging": {
      "purpose": "O vigia descobre a anotação decisiva…",
      "must_render": ["O caderno deve estar aberto"],
      "composition": ["O rosto e o caderno permanecem em foco"]
    },
    "positive_prompt": "…", "negative_prompt": "…"
  },
  "reference_images": { "char-vigia": ["…"] },
  "provenance": { "run_id": "…", "model": "…", "seed": 1234 } }
```

Joins, all present in a run: the image file name is the `shot_id`; the
declaration is in `07-direction/outcome.json`; `staging.purpose`,
`staging.must_render` and `staging.composition` are respectively the matched
shot's `purpose`, `must_render_constraints` and `composition_constraints`; the
world gives each entity's kind, display name and aliases;
`05-production-bible/reference-sheet/` gives reference images; optional
`05-production-bible/bible.json` supplies `visual_identity`, `relative_scale`
and `approximate_dimensions` by `entity_id`; and the `*.visual.json` sidecar
gives prompts, model and seed. A **usable run** is a run from which at least one
case is emitted, whatever other stages that run happens to hold.

**Labels.** A minimal labelling tool (a local HTML page or a terminal loop)
shows the frame beside its declaration and records one line per case:

```jsonc
{ "schema_version": "2.0", "taxonomy_version": "2.0",
  "case_id": "…", "defects": ["duplicated_character"], "notes": "",
  "labeller": "your-name", "at": "2026-09-20T…" }
```

New labels carry `"taxonomy_version": "2.1"`. The labelling page shows each
defect with a one-line description in Portuguese, the operator's language,
because a bare slug hid `broken_face` for 67 frames.

Schema-1.0 label lines are historical records and remain valid as written. In
particular, legacy `broken_anatomy` means “body or hands, unspecified”; scoring
must not silently reinterpret it as either `broken_body` or `broken_hands`.

Targets: **300 labelled cases** for v1, stratified by run, framing and by
whether a character is declared; at least 40 positives for each defect v1 ships.
A case nobody is sure about is labelled `uncertain` and excluded from scoring,
never guessed.

**Identity labels are a separate pass.** A frame-level label does not certify
identity: the operator labelled the first 67 frames without comparing faces to
the references in detail, so a frame labelled `clean` says nothing about
whether each character is the right person. Fitting identity's threshold on
those labels would fit it to a negative nobody checked.

So identity is labelled per **(case, declared character)**, in its own
append-only file, `benchmarks/labels/identity.jsonl` (gitignored, like the
frame labels):

```jsonc
{ "schema_version": "1.0", "case_id": "…", "entity_id": "char-vigia",
  "decision": "same" | "different" | "not_visible" | "unsure",
  "faces": [0],                        // indices into face_boxes the labeller picked
  "face_boxes": [[x0,y0,x1,y1], …],     // every face the backend found in the frame
  "labeller": "your-name", "at": "…" }
```

The page shows the character's reference face, cropped and enlarged, beside
every face found in the frame, also cropped, and asks one thing: which of these
faces is this character? Picking faces and confirming means `same`; marking the
character present but looking different means `different`; `not_visible` when
their face is not in the frame; `unsure` otherwise. Picking **two** faces as the
same character is how a duplicate is recorded — the ground truth identity-based
`duplicated_character` will need.

Findings from the identity labelling pass and duplicate investigation:

- **Collective characters.** The corpus has characters that are a group —
  "agentes", three men in suits on one reference sheet. For them several faces
  are correct, not a duplicate, and the labeller picks all of them as `same`.
  Nothing structured says an entity is a group: only its plural name, which is
  prose. Until the declaration carries that fact (an entity-level field, set
  upstream by the generator pipeline), duplicate scoring cannot tell a group from a clone
  and must not score entities it cannot classify.
- **A reference sheet can itself be defective.** Identity measures a frame's
  distance to the reference, so a cloned or deformed reference poisons every
  shot of that character, and no frame-level check can see it. Judging the
  reference sheets themselves — one declared character, exactly one person —
  is a natural use of the same presence and identity machinery, and belongs to
  the upstream reference generation stage as much as to this repository.

### Measured signals: clones and anatomy

All evaluations run on the same labelled set (172 frames judged: 22 broken-hands
positives, 4 duplicate positives, ~150 negatives; plus 100 synthetic clones vs
100 two-person frames), so numbers are directly comparable:

1. **Overlapping box bug invalidated initial appearance results:** The person
   detector returned overlapping boxes for a single person (17 of 70 multi-person
   frames). Every clone signal now compares only distinct people (`box IoU < 0.2`).
2. **Appearance signals for clones (most similar distinct pair per frame, post-fix):**
   Colour histogram AUC 0.75, OSNet-AIN 0.72, OSNet 0.69, DINOv2 0.57. Face
   similarity (SFace) does not separate (0.28–0.72 vs up to 0.66 on clean pairs).
3. **"Two people look alike" is the wrong question:** Top-scoring clean pairs were
   groups that look alike by design (agents in suits, hooded figures, generic
   extras). A clone is one declared character appearing twice. Reframed per
   character against its reference sheet (clone score = best character's
   second-highest match): OSNet-AIN AUC 0.76, OSNet 0.63, DINOv2 0.59 (only 4
   corpus positives).
4. **Local VLM, verbalised JSON probability (8B instruct, 4-bit):** AUC 0.50 on every
   defect — it answered "no defect" with confident wrong evidence even on all 22
   broken hands. Verbalised probabilities carry no ranking signal.
5. **Local VLM, yes/no first-token logit ratio (VQAScore, 3.9 s/frame):** Duplicated
   character **AUC 0.94** (recall 3/4 at 10% FPR, 4/4 at 20%), `missing_entity` 0.78,
   `fused_objects` 0.51, `broken_hands` 0.27 full frame / 0.42 on enlarged person crops.
   On 100 synthetic clones vs 100 two-person frames: **AUC 1.00** — but those are easy
   cases (posed, side by side), so treat it as an upper bound.
6. **Contrastive in-context examples for hands (2 broken + 2 clean crops, leave-one-out):**
   AUC 0.51. Three VLM attempts at chance: an 8B VLM does not perceive finger defects.
7. **Hand-landmark instability (multi-transform landmarker):** Multi-transform (flip,
   ±8°, 0.9/1.1 scale; features: presence, handedness, landmark variance, bone ratios,
   joint angles, finger crossing): best single feature AUC 0.634; 5-fold out-of-fold
   logistic fusion drops to **0.47**. Hand detected in only 70% of frames.
8. **Finger counting by contour and convex hull:** AUC 0.645 but recall 0 at 5–10%
   FPR, and segmentation fails on 30% of frames — not shippable as a standalone signal.
9. **Line-art (Canny) crops vs colour crops (HOG + logistic, 5-fold, 22 positives):**
   Colour **0.690** (95% CI 0.568–0.804), line art 0.527. Line art loses the shading
   and continuity cues the defect lives in. The colour-crop classifier is the first
   signal to pass the 0.66 reference of a rejected research-only detector, with 22
   positives and no deep training — so the bottleneck for hands is labelled data, not
   method.

Scoring `wrong_identity` uses **only** identity records. A case is positive when
any of its characters is `different`, negative when every character with a
reference is `same`, and excluded otherwise — including every case with no
identity record at all. A frame-level label never counts as an identity
negative.

Splits: `dev` (fit thresholds) and `test` (never used for fitting), split by
**run**, so frames of the same film cannot leak between them.

## 8. The benchmark, which is the point of the separate repo

`benchmarks/run.py` evaluates every competitor on the same cases and writes a
leaderboard:

| system | defect | precision | recall | F1 | ms/frame | peak RAM | licence |
|---|---|---|---|---|---|---|---|

Competitors for v1:

- `frame-jury` (ours, `budget=cheap`) and `frame-jury-full` (with the VLM);
- `yolo-count` — Ultralytics person counting (AGPL, benchmark only);
- `insightface-identity` — ArcFace identity (research weights, benchmark only);
- `vlm-only` — a vision model asked the question in prose, no detectors;
- `null` — always accept, as the floor everything must beat.

Rules: same cases, same split, pinned versions recorded in the report, three
runs for timing, median reported. The leaderboard is committed on every change,
so a regression is visible in a diff.

**The bar for production deployment:** on the held-out split, for
`duplicated_character` and `wrong_identity`, recall ≥ 0.90 with precision ≥ 0.95
at `budget=cheap`, under 400 ms per frame on CPU. Precision is the strict one on
purpose: a judge that rejects good frames burns GPU hours and trust.

## 9. Milestones for the delegated agent

Each milestone is a PR. See `DELEGATION.md` for who builds each one, on which
model, and what they may do without asking.

**Labels are not on the critical path of the code.** The operator has no desk
time to label, so the 300 labels M1 asks for do not exist yet and will not for a
while. Every milestone below is therefore built so that *only the fitting of
thresholds* needs them: backends, checks, the harness and the packaging are
written and tested against synthetic and hand-made fixtures, behind a default
threshold file that the labels later replace. A milestone that cannot be
finished without labels stops at that seam, says so in its PR, and moves on.

- **M1 — corpus.** Builder, case schema, labelling tool, 300 labels, splits.
  Done when `python -m benchmarks.corpus --runs <path>` produces cases for every
  run without touching them, and the label file passes its schema test.
  *Code merged in PR #1; the 300 labels remain outstanding.*
- **M2 — presence.** Detector backends (torchvision, RT-DETR), the presence
  check against the declaration, calibration per framing.
- **M3 — identity.** YuNet + SFace, reference-vs-frame similarity, thresholds
  fitted per framing; abstain when no face is found in either side.
- **M4 — harness and baselines.** Competitors, leaderboard, timing protocol.
  Done when the table is reproducible from a clean clone.
- **M5 — anatomy and legibility research.** Only after M4 shows where the
  remaining errors are.
- **M6 — packaging.** `frame_jury` as a package, contract stable, licence test,
  `THIRD_PARTY_NOTICES.md`, and an example pipeline adapter.
- **M7 — mobile labelling, an offline-first PWA.** The labels are blocked on the
  operator having no desk time, so labelling has to fit a commute. Before
  leaving, the phone pulls a bundle — downscaled WebP frames, reference
  thumbnails and declarations, roughly 100–150 MB for the 782 cases against
  726 MB of originals — then labels with **no connectivity at all**, queueing
  decisions locally and syncing into `labels.jsonl` when it next reaches the
  machine. This keeps the corpus on the operator's machine, needs no hosting and
  no PC left running, and survives a subway. It reuses the M1 queue order,
  taxonomy v2.0 and append-only store rather than inventing a second labelling
  path: the same case ids, the same schema, the same file. Last in sequence
  because the detector work does not wait on it, but it is what finally unblocks
  every threshold in M2 and M3.

Quality bar for every milestone: tests that fail without the feature, no network
in tests, CPU-only path always available, and a README section a person can
follow from a clean machine.

## 10. Example integration: pipeline usage

An integrating pipeline (such as Content Factory) keeps its own port
(`VisualAssuranceGate` or similar) and its own contract. frame-jury is a provider
behind it, evaluating frames against shot declarations. A typical run-control loop
uses the verdict to trigger one regeneration with an adjusted prompt for a rejected
frame, then hands off unresolved issues for human review.

Nothing in frame-jury may import an upstream pipeline, and nothing upstream may
import a frame-jury internal: the JSON contract in §4 is the whole surface.

## 11. Working agreement for the delegation

- The agent works in the `frame-jury` repository only, never in external projects.
- External generator runs are **read-only input**: the corpus builder must open them
  read-only and copy nothing into the repository. No image is committed.
- Every claim of quality comes with the leaderboard row that supports it.
- When a donor's approach is reproduced, the paper or repository is cited in the
  code comment, with its licence, and the implementation is written from the
  description — not copied.
- Report at each milestone: what was built, the leaderboard diff, what surprised
  you, and what you would change in this spec.
