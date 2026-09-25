# CCTV AI Vision Lab — Native Android v0.3

The Android application is a native implementation of the browser CCTV AI lab. It uses CameraX for live capture and ONNX Runtime Android for local inference.

## V1 architecture

```text
CameraX / selected image
        |
        v
YOLOX Nano 416 OR YOLOX Small 640
NNAPI first -> CPU fallback
        |
        v
duplicate suppression
        |
        v
IoU tracking
        |
        v
stable, person-prioritized top-5 objects
        |
        v
RelateAnything 448x448
NNAPI first -> CPU fallback
        |
        v
temporal relation smoothing
        |
        v
Compose overlay + metrics
```

The relation stage receives clean RGB pixels and box coordinates. Detector rectangles are not burned into the image given to RelateAnything.

## Model downloads

Model weights are **not bundled in the APK**.

The application downloads them into app-private storage when needed:

- YOLOX Small 640 detector: retained as the accuracy baseline
- YOLOX Nano 416 detector: downloaded only when selected for the speed benchmark
- RelateAnything: downloaded lazily when Scene AI is enabled
- RelateAnything predicate bank: downloaded with the relation model

After a successful download the local file and version metadata are reused on future launches, including offline launches. The app only downloads an asset again when it is missing, its version/URL changes, the user chooses **Clear and re-download models**, app data is cleared, or Android removes the application data.

The current model URLs mirror the already-tested web implementation. Third-party weights should not be redistributed inside public APK releases until their redistribution terms are confirmed.

## Runtime

- Kotlin
- Jetpack Compose
- CameraX
- ONNX Runtime Android 1.24.3
- minSdk 28
- targetSdk 36
- compileSdk 36
- Java 17

The runtime tries NNAPI session creation first. If the model/device cannot create an NNAPI session, it falls back to the default ONNX Runtime CPU provider. The actual provider and inference duration are shown in the app after inference.

Qualcomm QNN is intentionally outside V1.

## Features

- rear/front live camera
- CameraX KEEP_ONLY_LATEST analysis
- image import
- detector confidence control
- relationship confidence control
- 1.0 / 1.5 / 3.0 second requested relationship cadence with adaptive self-throttling
- all-object or human+vehicle display filter
- debug detector-box toggle
- duplicate suppression
- motion-aware low-FPS tracking with center-distance fallback
- semantic relation filtering (for example, impossible "Person wearing Chair" output is removed)
- maximum three validated relationship overlays
- person-prioritized top-5 relation candidates
- clean Scene AI overlay
- local model/download status
- detector/relation provider and timing metrics
- on-device detector benchmark selector: YOLOX Nano 416 vs YOLOX Small 640
- last inference time retained for each detector during the current app session
- effective adaptive relation cadence shown in the UI
- manual model clear/re-download

## Build

From the repository root:

```bash
gradle -p android :app:testDebugUnitTest
gradle -p android :app:lintDebug
gradle -p android :app:assembleDebug
```

The debug APK is generated at:

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

GitHub Actions also uploads the APK as the **cctv-ai-vision-lab-debug** artifact.

## Physical-device validation

A successful CI build proves the Android project compiles, but it cannot validate a phone GPU/NPU. On a physical Android device verify:

1. Grant camera permission.
2. Wait for the selected detector to show READY.
3. Start live camera and confirm detections.
4. Check Detector Runtime: NNAPI or CPU.
5. Keep the camera on the same scene, run YOLOX Small 640 for several frames, then switch to Fast Nano 416 and compare the "last inference" values.
6. Confirm a stable person/chair keeps the same track IDs for substantially longer than v0.2.
7. Enable Scene AI and wait for RelateAnything + predicate bank to become READY.
8. Check Relation Runtime, relation inference time, relation candidate count, and effective cadence.
9. Confirm implausible relations such as "Person wearing Chair" are not shown.
10. Close the app, disable network, reopen it, and confirm the local models remain READY.
11. Run live AI for at least 10 minutes and observe latency/thermal behavior.

See the repository-level `ANDROID_APP_PLAN.md` for complete acceptance criteria.
