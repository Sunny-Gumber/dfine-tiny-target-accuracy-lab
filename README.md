# Browser Human + Vehicle Detection

A small browser demo for detecting people and road vehicles from a webcam, mobile camera, or uploaded image.

**Live demo:** https://sunny-gumber.github.io/dfine-tiny-target-accuracy-lab/

## What it does

- Runs YOLOX Nano directly in the browser
- Uses WebGPU when available, with WASM as a fallback
- Supports rear camera, front camera/webcam, and image upload
- Keeps the public output simple: **Human** and **Vehicle**
- Shows live inference time and a rolling FPS estimate for camera input
- Does not upload camera frames or images to an application server

## Current setup

The model uses its native **416 × 416** input. For live use, the browser requests a **640 × 360** camera stream to keep capture and rendering overhead reasonable. Uploaded images are analysed as a complete frame and are resized by the model preprocessor.

The underlying COCO road-vehicle classes (bicycle, car, motorcycle, bus and truck) are grouped into one `Vehicle` label. This avoids presenting fine-grained vehicle classification as something the model was not tuned for in Indian traffic scenes.

## Run locally

```bash
npm install
npm run dev
```

Then open the local Vite URL in a browser. Camera access requires a secure context (`https://` or localhost).

## Build

```bash
npm run build
```

GitHub Actions publishes the `dist` folder to GitHub Pages on pushes to `main`.

## Notes

This is an engineering demo, not a production surveillance system. Detection quality depends on target size, lighting, occlusion, camera angle, browser runtime and device performance. The displayed FPS is an inference-rate estimate based on measured model latency; it is not the camera capture frame rate.

The model/runtime is downloaded on first use, so the first load can take longer than later runs.
