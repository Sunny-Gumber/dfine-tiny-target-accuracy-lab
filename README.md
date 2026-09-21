# CCTV AI Vision Lab — YOLOX-S + RelateAnything

A browser-based CCTV computer-vision lab using **YOLOX-S** for object detection and **RelateAnything** for Phase 1 scene understanding.

**Live demo:** https://sunny-gumber.github.io/dfine-tiny-target-accuracy-lab/

## What works

### Object detection

- YOLOX-S at its native **640 × 640** model input
- All **80 COCO classes** enabled by default
- Optional Human + Vehicle display filter
- Back camera, front camera/webcam, and image upload
- WebGPU when available, with WASM fallback
- Detection timing and rolling inference-rate estimate

### Phase 1 — scene understanding

For an uploaded image the app now runs this pipeline:

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

The first Phase 1 implementation is deliberately **image-only**. It feeds the actual YOLOX-S boxes into the released RelateAnything ONNX graph and shows relationships such as:

- wearing
- riding
- holding
- carrying
- sitting on
- using
- attached to
- beside
- in front of
- behind
- above / below

The relation score threshold is adjustable in the UI. The default is **56%**, matching the released calibrated operating-point guidance rather than applying a detector confidence to relation scores.

## Privacy and model loading

Camera frames and uploaded images are processed in the browser and are not sent to an inference API.

YOLOX-S and RelateAnything model assets still have to be downloaded by the browser. RelateAnything is loaded only when **Understand scene · Phase 1** is pressed, because the released ONNX model is substantially larger than the detector. The relation runtime uses WASM in Phase 1 for compatibility with the released ONNX graph.

The predicate bank is loaded from the upstream RelateAnything release and only a compact CCTV-oriented subset is supplied to the relation graph.

## Tech stack

- React
- TypeScript
- Vite
- LibreYOLO Web
- ONNX Runtime Web
- YOLOX-S
- RelateAnything ViT-S+

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
│   ├── App.tsx          # UI, detector flow and Phase 1 orchestration
│   ├── relations.ts     # RelateAnything browser runtime + relation decode
│   ├── main.tsx
│   ├── runtime.ts       # ONNX Runtime configuration
│   └── styles.css
├── index.html
├── package.json
├── package-lock.json
├── tsconfig.json
└── vite.config.ts
```

## Phase 1 implementation notes

RelateAnything accepts image pixels and bounding boxes; object class labels are not required as model inputs. The app keeps the YOLOX-S detector and passes its boxes to the relation head. Up to 32 highest-confidence detections are used, matching the released relation graph's box capacity.

The relation preprocessing follows the released deployment path:

- source image resized directly to **448 × 448** RGB
- source boxes normalized and converted from XYXY to CXCYWH
- boxes padded to the graph's fixed capacity
- runtime-supplied predicate embeddings (`W`) and routing weights (`alpha`)
- calibrated score: `sigmoid(a × (predicate_logit + pair_logit) + b)`

Phase 1 uses the released calibration values for `relsgg-vits16plus` and ranks surviving relationships with detector confidence after thresholding, matching the upstream deployment convention.

## Scope and next phases

This is an engineering/demo project, not a production surveillance system. Detection and relation quality vary with target size, lighting, occlusion, camera angle, motion blur, and device performance.

Phase 2 can add live-video relation inference at a reduced cadence plus tracking. Phase 3 can add a CCTV event/rule engine for temporal events such as abandoned objects, PPE logic, and other multi-frame behaviours.

## Upstream credits and licenses

- Browser object detection: [LibreYOLO Web](https://github.com/LibreYOLO/libreyolo-web)
- Relation prediction: [Maelic/RelateAnything](https://github.com/Maelic/RelateAnything)
- RelateAnything code is Apache-2.0; released weights are a DINOv3 derivative and follow the DINOv3 license described by the upstream project.

See each upstream project for its complete license and third-party notices.
