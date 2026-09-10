# YOLOX-S Browser Object Detection

A small browser-based object detection demo using **YOLOX-S** and the **COCO 80-class dataset**.

**Live demo:** https://sunny-gumber.github.io/dfine-tiny-target-accuracy-lab/

The project is intentionally simple: choose a camera or upload an image, run inference locally in the browser, and draw detections over the source.

## Features

- YOLOX-S at its native **640 × 640** model input
- All **80 COCO classes** enabled by default
- Default confidence threshold: **60%**
- Optional Human + Vehicle display filter for CCTV-focused tests
- Back camera, front camera/webcam, and image upload
- WebGPU when available, with WASM fallback
- Live inference timing and rolling inference-rate estimate
- Camera frames and uploaded images stay in the browser during inference

## Tech stack

- React
- TypeScript
- Vite
- LibreYOLO Web
- ONNX Runtime Web

## Run locally

Requirements: Node.js 22 or newer.

```bash
npm ci
npm run dev
```

Open the Vite development URL in your browser. Camera access requires localhost or HTTPS.

Use `npm install` when intentionally changing dependencies so the lock file is updated with `package.json`.

## Build

```bash
npm run build
```

The build command runs the TypeScript check first and then creates the production bundle in `dist/`.

## Project structure

```text
.
├── .github/workflows/
│   ├── ci.yml                    # pull-request build check
│   └── pages.yml                 # GitHub Pages deployment
├── src/
│   ├── App.tsx                   # camera, inference and UI logic
│   ├── coco.ts                   # COCO labels used by the UI
│   ├── main.tsx                  # React entry point
│   └── styles.css                # application styles
├── index.html
├── package.json
├── package-lock.json
├── tsconfig.json
└── vite.config.ts
```

## Detection modes

**All COCO objects** is the default mode. Labels use the standard COCO class names.

**Human + Vehicle** is only a display filter. It keeps `person` as Human and groups bicycle, car, motorcycle, bus, and truck as Vehicle. The underlying YOLOX-S inference pass is unchanged.

## Runtime notes

The application UI and the inference runtime are split into separate JavaScript chunks. This lets the page render before the heavier LibreYOLO/ONNX Runtime code is loaded.

YOLOX-S is heavier than YOLOX Nano, so browser speed depends strongly on the device and execution provider. The displayed FPS value is an inference-rate estimate calculated from measured model latency; it is not the camera capture frame rate.

For camera input the browser requests a 640 × 360 stream, while the model preprocessor converts the frame to the model's 640 × 640 input. Uploaded images are analysed as complete frames.

## Scope

This is an engineering/demo project, not a production surveillance system. Detection quality varies with target size, lighting, occlusion, camera angle, motion blur, browser support, and device performance.

The model and browser inference runtime are downloaded on first use, so the first visit can take longer than later visits when browser caching is available.

## Acknowledgements

Browser inference is provided by [LibreYOLO Web](https://github.com/LibreYOLO/libreyolo-web), which uses ONNX Runtime Web for WebGPU/WASM execution.
