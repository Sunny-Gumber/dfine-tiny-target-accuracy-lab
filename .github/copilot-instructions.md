# Copilot instructions

This repository contains a working browser CCTV AI lab and a new native Android implementation plan.

## Non-negotiable rules

- Do not break the existing web app.
- Native Android code belongs under `android/`.
- Read `ANDROID_APP_PLAN.md` before changing Android code.
- Prefer small focused PRs over broad rewrites.
- Build and test before declaring a task complete.
- Do not commit model weights unless redistribution licensing is verified.
- Keep third-party model URLs/version metadata explicit.
- Use clean camera/image pixels for RelateAnything; detector boxes are coordinates only.
- Keep inference and file IO off the UI thread.
- CameraX analysis must use latest-frame/backpressure behavior and must always release `ImageProxy`.
- Preserve provider fallbacks: NNAPI when viable, CPU fallback.
- Do not claim NNAPI/QNN acceleration unless the runtime actually selected/used that provider.
- Expose measurable inference timing and provider selection in the UI.
- The V1 detector remains 640x640; RelateAnything remains 448x448 unless a separate validated model is introduced.
- Relation inference is periodic and limited to at most 8 stable, deduplicated detections.
- Track IDs are diagnostic and may be short-lived.
- Do not upload camera frames or images to a server.
- Do not introduce telemetry in V1.
- New Android workflows must preserve the existing web CI/Pages workflows.

## Verification

For Android PRs, run the relevant Gradle build/test/lint tasks that exist at that stage. If the complete app cannot yet build because a prerequisite workstream has not landed, state that explicitly and keep the change isolated enough to merge safely.

For changes to inference math, add deterministic unit tests for transforms, tensor shapes, score calibration, deduplication, and selection logic where possible.
