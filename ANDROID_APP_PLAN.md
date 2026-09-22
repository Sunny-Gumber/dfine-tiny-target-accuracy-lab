# Native Android APK Plan — CCTV AI Vision Lab

## Goal

Build a native Android application that reproduces and improves the current browser CCTV AI lab by using Android-native camera, storage, and ONNX Runtime execution rather than WebGPU/WASM.

The app must support:

- Live rear/front camera
- Uploaded image analysis
- YOLOX-S object detection using the same model family as the web app
- RelateAnything scene-relation inference
- Lightweight object tracking and duplicate suppression
- Clean-frame relation inference (detector boxes are internal guidance only)
- Top-8 relation candidates
- Runtime/provider visibility
- Local model persistence after first successful download
- Offline reuse after the models are stored
- APK builds from GitHub Actions

## Target architecture

```text
CameraX / image
      |
      v
Frame scheduler
      |
      +-------------------------+
      |                         |
      v                         |
YOLOX detector                  |
      |                         |
      v                         |
dedupe + tracker                |
      |                         |
      v                         |
stable top-N boxes              |
      |                         |
      +--------> RelateAnything |
                     |           |
                     v           |
              semantic relations |
                     |           |
                     v           |
             temporal smoothing  |
                     |           |
                     v           |
               final overlay <---+
```

## Android stack

- Kotlin
- Jetpack Compose
- CameraX Preview + ImageAnalysis
- ONNX Runtime Android
- NNAPI execution provider when supported
- CPU fallback
- Coroutines / structured concurrency
- Android private app storage for models
- WorkManager or foreground-safe download component only if required
- GitHub Actions for debug APK build and release artifact upload

Initial minimum Android version: API 28 (Android 9).
Target/compile SDK: current stable API supported by the chosen Android Gradle Plugin.
Use stable dependency versions only.

## Model/runtime strategy

### Detector

The Android implementation must reproduce the current YOLOX-S preprocessing and postprocessing closely enough that the same test image produces comparable object labels and boxes.

Requirements:

- preserve 640 x 640 native detector input for the first APK
- confidence threshold configurable
- all COCO classes available
- optional Human + Vehicle display filter
- strong duplicate suppression before tracking
- maximum post-processing count kept conservative for mobile
- provider preference: NNAPI first when supported, CPU fallback

### RelateAnything

Requirements:

- input image 448 x 448
- accepts detector box coordinates, but never detector drawings burned into the relation input image
- at most 8 high-confidence, deduplicated, stable objects
- same released predicate bank/calibration behavior as current web implementation
- periodic relation inference rather than every camera frame
- selectable cadence: 1.0 s / 1.5 s / 3.0 s
- provider preference: NNAPI first when viable, CPU fallback
- runtime/provider and inference duration shown in UI

### QNN

Qualcomm QNN is NOT required for V1 because it requires a custom ONNX Runtime Android build and Qualcomm SDK integration.

Keep the runtime abstraction clean so a future QNN flavor can be added without changing camera/UI/business logic.

## Model storage

Do not redownload models at every launch.

Model manager requirements:

1. Check app-private model directory.
2. Verify model file exists and is non-zero.
3. Prefer SHA-256 verification when a trusted checksum is available.
4. Download only missing/outdated model assets.
5. Download to a temporary file.
6. Verify.
7. Atomically rename into final location.
8. Keep a small local manifest containing model version, URL, checksum if known, and downloaded timestamp.
9. Display download progress and retry state.
10. After first successful install/download, app must start offline using local models.

Do not package third-party model weights directly into a public APK until redistribution licensing is confirmed.

## Camera pipeline

Use CameraX with Preview and ImageAnalysis.

Requirements:

- front/rear camera switch
- KEEP_ONLY_LATEST backpressure
- avoid blocking the CameraX analyzer
- rotate/transform frames correctly
- detector scheduler prevents overlapping detector calls
- relation scheduler prevents overlapping relation calls
- use one captured/snapshotted frame for each relation request so relation boxes and pixels refer to the same moment
- release ImageProxy promptly
- pause/resume AI without closing the app
- camera permission handling
- lifecycle-safe start/stop

## Tracking

V1 uses a lightweight tracker, matching the browser behavior but improving duplicate stability.

Requirements:

- same-label matching
- IoU matching
- duplicate suppression using IoU and containment
- track lifetime/misses
- stable short-lived IDs
- expose track age
- relation model only receives tracks that have survived at least two detector passes

Future upgrades such as ByteTrack can be isolated behind a tracker interface.

## UI/UX

Main screen:

- camera preview
- start/stop AI
- front/back switch
- detector threshold
- relationship threshold
- relation cadence
- display filter
- image-import mode
- relation list
- final relation arrows/labels
- no detector boxes while Live Scene AI is enabled unless a Debug Overlay toggle is enabled

Performance panel:

- detector provider
- detector ms
- detector FPS
- relation provider
- relation ms
- relation update count
- detection count
- active track count
- relation count

Model panel:

- detector model state
- relation model state
- local/cached state
- model size
- download progress
- clear/re-download models button

## Threading/performance rules

- UI thread must never perform inference or model file IO.
- Camera frame acquisition, preprocessing, detector inference, relation inference, and downloads use background dispatchers/executors.
- Detector and relation inference should not run concurrently by default on low-power devices if concurrent execution causes thermal or latency regression.
- Add a simple scheduling policy so relation inference gets a captured frame and detector can be throttled during the relation pass if required.
- Prefer reusable buffers and tensors where practical.
- Avoid allocating a new full-resolution bitmap for every camera frame.
- Capture timing for preprocessing separately from inference where useful.

## Functional acceptance criteria

V1 is complete only when:

- project builds on GitHub Actions
- a debug APK is produced as a downloadable workflow artifact
- app installs on a physical Android phone
- first-run model download completes
- second launch uses local model files without downloading again
- rear and front cameras both work
- uploaded image mode works
- YOLOX detection works on live camera
- duplicate same-object boxes are materially reduced
- stable track IDs appear in debug mode
- RelateAnything receives clean pixels plus internal boxes
- relation inference can be started/stopped independently
- relation provider is visible
- inference times are visible
- application survives screen rotation/background/foreground without leaking the camera
- no inference occurs on the main/UI thread
- model download failures provide retry instead of crashing

## Performance acceptance criteria

Initial goals, not guaranteed hardware-independent promises:

- detector should be measurably faster than the current browser/WASM baseline on the same device when NNAPI can accelerate the graph
- relation model should be benchmarked on NNAPI and CPU independently
- no UI freeze longer than a normal frame drop during inference
- no unbounded queue of camera frames
- relation cadence should self-throttle if relation inference is slower than requested cadence
- record thermal/latency behavior for at least a 10-minute live run

## Security/privacy

- no camera frame upload to a server
- no analytics/telemetry in V1 unless explicitly added later
- models download only from documented HTTPS sources
- app-private storage for models
- network permission used only for model downloads
- camera permission requested only when camera mode is used

## Licensing

Before a public release:

- document LibreYOLO / YOLO model license and redistribution terms
- document RelateAnything code license
- document RelateAnything released-weight / DINOv3-derived weight terms
- keep third-party attribution in NOTICE/README
- do not silently redistribute model weights if their license does not allow it

## Repository layout

Keep the current web app intact.

```text
/
├── src/                         # existing web app
├── android/
│   ├── app/
│   │   ├── src/main/java/.../
│   │   │   ├── camera/
│   │   │   ├── inference/
│   │   │   ├── models/
│   │   │   ├── tracking/
│   │   │   ├── relations/
│   │   │   ├── ui/
│   │   │   └── util/
│   │   └── src/test/
│   ├── build.gradle.kts
│   ├── settings.gradle.kts
│   └── gradle.properties
├── .github/agents/
├── .github/workflows/
│   └── android-apk.yml
└── ANDROID_APP_PLAN.md
```

## Workstreams

### A — Android architecture/scaffold

Owner profile: Android Architect

Deliver:

- Gradle Android project in /android
- Kotlin + Compose
- package/application IDs
- navigation/state skeleton
- dependency/version management
- base DI/service ownership approach without overengineering
- unit-test setup
- build instructions
- no model inference yet beyond clean interfaces

### B — Native inference + model manager

Owner profile: Vision Inference Engineer

Deliver:

- ONNX Runtime Android integration
- provider abstraction
- NNAPI preferred, CPU fallback
- model manager/download-once implementation
- detector pre/post-processing
- RelateAnything preprocessing/postprocessing
- predicate bank loading
- provider/timing metrics
- tests for tensor shapes, box transforms, scoring, cache manifest

### C — Camera + UI + overlay

Owner profile: Camera/UI Engineer

Deliver:

- CameraX Preview
- ImageAnalysis pipeline
- front/rear switching
- permission handling
- frame scheduler
- image import
- Compose controls
- relation overlay
- debug overlay
- performance/model status panels

### D — Release/QA

Owner profile: Release & QA Engineer

Deliver:

- GitHub Actions Android build
- debug APK workflow artifact
- lint/unit tests
- release checklist
- device test matrix
- 10-minute stability benchmark procedure
- README install instructions
- licensing/NOTICE validation

### E — Integration

After A-D PRs exist:

- resolve conflicts
- run full build
- run unit/lint tests
- produce APK
- test first-run vs second-run model behavior
- benchmark NNAPI vs CPU
- document known limitations
- tag V1 only after acceptance criteria are met

## Merge order

1. Architecture/scaffold
2. Model manager + inference foundation
3. Camera/UI integration
4. Relation/tracking integration
5. CI/release/QA
6. Final integration/benchmark fixes

Agents should avoid making unrelated changes to the existing browser implementation.
