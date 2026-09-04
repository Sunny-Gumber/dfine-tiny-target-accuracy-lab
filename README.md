# D-FINE Tiny-Target Accuracy Lab

A browser-based human and vehicle detector for testing distant or low-pixel
CCTV targets. It accepts images, uploaded videos, desktop webcams, and the front
or rear camera on a phone.

## What it includes

- D-FINE-S accuracy mode (onnx-community/dfine_s_coco-ONNX)
- D-FINE-N mobile-speed mode (onnx-community/dfine_n_coco-ONNX)
- WebGPU acceleration on compatible desktops
- Mobile-safe WebAssembly mode plus automatic fallback for unsupported WebGPU operations
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

## Camera requirements

Browsers permit camera access on HTTPS pages or localhost. On a phone, choose
**Rear camera** for the outward-facing lens. On a laptop, choose
**Front/webcam**.

Android and iOS use the WebAssembly backend because current mobile WebGPU
engines cannot execute every operation in the D-FINE ONNX graph. This changes
speed, not the selected model or its detection accuracy.

## Accuracy notes

D-FINE-S is the recommended accuracy test. D-FINE-N is smaller and usually a
better fit for phones. Precision scanning can expose small targets because each
crop is enlarged for inference, but it performs five model passes and is
therefore slower.

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
- app/detection.ts — D-FINE loading, tiled inference, class filtering, and merge logic
- app/globals.css — global theme

The browser runtime is loaded from the pinned
@huggingface/transformers 3.8.1 CDN module, so no extra inference dependency is
needed during the site build.
