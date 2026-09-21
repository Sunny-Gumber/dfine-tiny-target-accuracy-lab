# CCTV AI Vision Lab — YOLOX-S + RelateAnything

A browser-based CCTV computer-vision lab using **YOLOX-S** for object detection and **RelateAnything** for image and live scene understanding.

**Live demo:** https://sunny-gumber.github.io/dfine-tiny-target-accuracy-lab/

## What works

### Object detection

- YOLOX-S at its native **640 × 640** model input
- All **80 COCO classes** enabled by default
- Optional Human + Vehicle display filter
- Back camera, front camera/webcam, and image upload
- WebGPU when available, with WASM fallback
- Detection timing and rolling inference-rate estimate

### Phase 1 — still-image scene understanding

For an uploaded image:

```text
Image
  ↓
YOLOX-S
  ↓
Detected bounding boxes
  ↓
RelateAnything ViT-S+
  ↓
Ranked visual relationships
```

The app feeds real YOLOX-S boxes into the released RelateAnything ONNX graph and shows relationships such as wearing, riding, holding, carrying, sitting on, using, attached to, beside, in front of, behind, above, and below.

### Phase 2 — live scene understanding + tracking

Live camera mode now runs:

```text
Camera
  ↓
YOLOX-S continuously
  ↓
Lightweight IoU tracker
  ↓
Stable short-lived Track IDs
  ↓
RelateAnything every 1–3 seconds
  ↓
Temporal relation smoothing
  ↓
Live relationship overlay
```

Phase 2 deliberately does **not** run RelateAnything on every detector frame. The detector can keep updating continuously while the heavier relation model runs at a selectable cadence:

- **Fast:** 1.0 second
- **Balanced:** 1.5 seconds
- **Light:** 3.0 seconds

Balanced is the default.

The tracker is intentionally lightweight and browser-friendly. It matches detections by label and bounding-box IoU, gives them short-lived IDs, and lets relationship results remain attached to the same moving objects between relation passes.

Example live output:

```text
#3 Human → riding → #5 Motorcycle
#3 Human → wearing → #8 Backpack
#7 Human → sitting on → #9 Chair
```

A small temporal smoother keeps a relation for one missed relation pass and applies an exponential moving average to the relationship score. This reduces flicker without pretending to be a full multi-object-tracking or activity-recognition system.

## Scene-understanding scoring

The relationship-confidence slider defaults to **56%**.

The relation runtime follows the released `relsgg-vits16plus` score contract:

```text
sigmoid(a × (predicate_logit + pair_logit) + b)
```

with the calibration values shipped by the upstream model. Detector confidence affects ranking after relation thresholding rather than redefining the relation confidence itself.

## Privacy and model loading

Camera frames and uploaded images are processed in the browser and are not sent to an inference API.

YOLOX-S and RelateAnything model assets still have to be downloaded by the browser. RelateAnything is lazy-loaded when scene understanding is first requested because its ONNX model is substantially larger than the detector.

The predicate bank is loaded from the upstream RelateAnything release, and the browser supplies a compact CCTV-oriented subset to the relation graph.

## Tech stack

- React
- TypeScript
- Vite
- LibreYOLO Web
- ONNX Runtime Web
- YOLOX-S
- RelateAnything ViT-S+
- Lightweight IoU object tracking
- Temporal relationship smoothing

## Run locally

Requirements: Node.js 22 or newer.

```bash
npm ci
npm run dev
```

Open the Vite development URL in your browser. Camera access requires localhost or HTTPS.

## Build

```bash
npm run build
```

The build runs the TypeScript check first and then creates the production bundle in `dist/`.

## Project structure

```text
.
├── .github/workflows/
│   ├── ci.yml
│   └── pages.yml
├── src/
│   ├── App.tsx          # UI, detector flow and Phase 1/2 orchestration
│   ├── relations.ts     # RelateAnything browser runtime + relation decode
│   ├── tracking.ts      # IoU object tracker + temporal relation smoother
│   ├── main.tsx
│   ├── runtime.ts       # ONNX Runtime configuration
│   └── styles.css
├── index.html
├── package.json
├── package-lock.json
├── tsconfig.json
└── vite.config.ts
```

## Implementation notes

RelateAnything accepts image pixels and bounding boxes; object class labels are not required as model inputs. Up to 32 highest-confidence detections are used, matching the released relation graph's box capacity.

The relation preprocessing follows the released deployment path:

- source frame resized directly to **448 × 448** RGB
- source boxes normalized and converted from XYXY to CXCYWH
- boxes padded to the graph's fixed capacity
- runtime-supplied predicate embeddings (`W`) and routing weights (`alpha`)
- calibrated relation scoring
- top relations ranked using relation score plus detector confidence

For live video, tracking is post-detection logic only. It does not change RelateAnything's visual inputs or inject object class labels into the relation model.

## Limits

This is an engineering/demo project, not a production surveillance system.

The Phase 2 tracker is a lightweight IoU matcher, not ByteTrack/BoT-SORT/DeepSORT. IDs can change during heavy occlusion, fast motion, major scale changes, or long detector misses. The live relation layer is also sampled periodically, so it is designed for persistent interactions such as riding, carrying, wearing, sitting, or beside—not millisecond-level event detection.

Detection and relation quality vary with target size, lighting, occlusion, camera angle, motion blur, device performance, and whether the detector found the required objects in the first place.

## Next phase

Phase 3 can add a CCTV event/rule engine on top of tracked objects and smoothed relationships, including temporal logic such as:

- abandoned-object candidates
- PPE compliance state
- person + vehicle interaction
- persistent loitering / dwell logic
- rule-based alert confidence and cooldowns
- event timeline / metadata output

## Upstream credits and licenses

- Browser object detection: [LibreYOLO Web](https://github.com/LibreYOLO/libreyolo-web)
- Relation prediction: [Maelic/RelateAnything](https://github.com/Maelic/RelateAnything)
- RelateAnything code is Apache-2.0; released weights are a DINOv3 derivative and follow the DINOv3 license described by the upstream project.

See each upstream project for its complete license and third-party notices.
