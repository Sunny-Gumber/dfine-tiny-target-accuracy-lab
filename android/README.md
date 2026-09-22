# CCTV AI Vision Lab — Native Android

This directory contains the native Android implementation. The browser application in the repository root remains independent.

## Current stage

Android V1 architecture scaffold:

- Kotlin
- Jetpack Compose
- minSdk 28
- targetSdk 36
- compileSdk 36
- Java 17
- contracts for detector, relation engine, model manager, tracker and frame scheduler

CameraX and ONNX Runtime are intentionally added in focused follow-up workstreams.

## Build

Use Android Studio Quail 4 or compatible tooling, or Gradle 9.6 with JDK 17.

```bash
cd android
gradle :app:assembleDebug
gradle :app:testDebugUnitTest
```

The installable debug APK is generated under:

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

## Package

```text
com.sunnygumber.cctvaivisionlab
```

## Architecture

```text
camera/image
    |
FrameData
    |
Detector
    |
Tracker
    |
stable detections
    |
RelationEngine
    |
SceneRelation
    |
Compose UI / overlay
```

See the repository-level `ANDROID_APP_PLAN.md` for the full V1 plan and acceptance criteria.
