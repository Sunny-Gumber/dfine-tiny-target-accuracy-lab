# D-FINE Tiny-Target Accuracy Lab

A browser-based human and vehicle detector for testing distant or low-pixel
CCTV targets. It accepts images, uploaded videos, desktop webcams, and the front
or rear camera on a phone.

## What it includes

- D-FINE-S accuracy mode (onnx-community/dfine_s_coco-ONNX)
- D-FINE-N faster desktop mode (onnx-community/dfine_n_coco-ONNX)
- RT-DETRv2-R18 quantized mobile fallback (onnx-community/rtdetr_v2_r18vd-ONNX)
- WebGPU acceleration on compatible desktops
- Mobile-safe WebAssembly mode plus automatic model fallback for unsupported D-FINE operations
- Human and CCTV vehicle filtering
- Fast full-frame inference
- Five-pass precision scanning: one full frame plus four overlapping tiles
- Cross-tile duplicate merging
- Live boxes, confidence, pixel coordinates, class counts, and inference time
- Manual ground-truth counts for a quick per-frame count-agreement check
- Image, video, front camera/webcam, and mobile rear-camera input

The media and inference stay in the browser. The model files are downloaded
from Hugging Face the first time a model is used.

## Run locally

Requirements: Node.js 22.13 or newer.

    npm ci
    npm run dev

Open the local address shown in the terminal.

## Production build

    npm run build

## GitHub Pages

The repository includes an automatic GitHub Pages workflow. After GitHub Pages
is set to **GitHub Actions** in the repository settings, every push to `main`
builds and publishes the browser-only website.

    npm run build:github

Project site: https://sunny-gumber.github.io/dfine-tiny-target-accuracy-lab/

## Camera requirements

Browsers permit camera access on HTTPS pages or localhost. On a phone, choose
**Rear camera** for the outward-facing lens. On a laptop, choose
**Front/webcam**.

Android and iOS automatically use quantized RT-DETRv2-R18 on WebAssembly. Some
browser runtimes cannot execute the `ceil()` shape calculation in D-FINE's
MaxPool graph, so changing only the execution backend is insufficient. The app
now changes both the model and the backend, then retries the frame automatically.

## Accuracy notes

D-FINE-S is the recommended desktop accuracy test. D-FINE-N is its smaller,
faster alternative. RT-DETRv2-R18 is the compatible mobile path and is loaded
in quantized form to reduce download and memory pressure. Precision scanning
can expose small targets because each crop is enlarged for inference, but it
performs five model passes and is therefore slower.

The included weights are pretrained on COCO and the interface keeps only
person, bicycle, car, motorcycle, bus, and truck. Filtering the output classes
does not reduce the neural network's compute; a genuinely two-class optimized
model requires fine-tuning or distillation on a human/vehicle CCTV dataset and
exporting those weights.

The displayed count agreement is a convenient frame check, not formal model
accuracy. A proper benchmark needs annotated bounding boxes and evaluation with
precision, recall, and mAP.

## Project structure

- app/page.tsx — media, camera, analysis loop, overlay, and interface
- app/detection.ts — model loading, mobile fallback, tiled inference, class filtering, and merge logic
- app/globals.css — global theme

The browser runtime is loaded from the pinned
@huggingface/transformers 3.8.1 CDN module, so no extra inference dependency is
needed during the site build.
