import * as ort from "onnxruntime-web";
import { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";

type FacingMode = "environment" | "user";
type SourceMode = "camera" | "image";
type SourceElement = HTMLVideoElement | HTMLImageElement;
type Point = { x: number; y: number };
type Box = { x: number; y: number; width: number; height: number };
type Detection = { box: Box; score: number; keypoints: Point[] };
type Track = { id: number; box: Box; lastSeen: number; captureId: number; lastAgeSampleAt: number };
type AgeState = "waiting" | "sampling" | "ready" | "unavailable";

type CaptureEntry = {
  id: number;
  dataUrl: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  frameWidth: number;
  frameHeight: number;
  coverage: number;
  faceRatio: number;
  brightness: number;
  sharpness: number;
  pose: string;
  roll: number;
  frReadiness: "Good" | "Fair" | "Poor";
  capturedAt: string;
  source: string;
  detectionMs: number;
  ageRange?: string;
  ageGroup?: string;
  ageSamples: number;
  ageStability: string;
  ageMs?: number;
  ageState: AgeState;
};

const INPUT_SIZE = 640;
const AGE_INPUT_SIZE = 96;
const LIVE_INTERVAL_MS = 320;
const TRACK_TTL_MS = 1400;
const TRACK_IOU_THRESHOLD = 0.28;
const NMS_IOU_THRESHOLD = 0.42;
const PORTRAIT_WIDTH = 160;
const PORTRAIT_HEIGHT = 320;
const AGE_SAMPLE_INTERVAL_MS = 550;
const MAX_AGE_SAMPLES = 8;

const SCRFD_MODELS = [
  {
    label: "SCRFD-2.5G",
    url: "https://raw.githubusercontent.com/xlite-dev/scrfd-toolkit/main/examples/hub/onnx/cv/scrfd_2.5g_bnkps_shape640x640.onnx",
  },
  {
    label: "SCRFD-500M fallback",
    url: "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/det_500m.onnx",
  },
];

const AGE_MODELS = [
  "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/genderage.onnx",
  "https://huggingface.co/Arctic1998/insightface/resolve/main/genderage.onnx",
];

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
ort.env.wasm.numThreads = 1;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function formatMs(value?: number) {
  if (!value || !Number.isFinite(value)) return "—";
  return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;
}

function iou(a: Box, b: Box) {
  const ax2 = a.x + a.width;
  const ay2 = a.y + a.height;
  const bx2 = b.x + b.width;
  const by2 = b.y + b.height;
  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const intersection = iw * ih;
  if (!intersection) return 0;
  return intersection / (a.width * a.height + b.width * b.height - intersection);
}

function nonMaximumSuppression(input: Detection[]) {
  const sorted = [...input].sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  for (const candidate of sorted) {
    if (kept.every((existing) => iou(candidate.box, existing.box) < NMS_IOU_THRESHOLD)) {
      kept.push(candidate);
      if (kept.length >= 100) break;
    }
  }
  return kept;
}

function getSourceSize(source: SourceElement) {
  if (source instanceof HTMLVideoElement) return { width: source.videoWidth, height: source.videoHeight };
  return { width: source.naturalWidth, height: source.naturalHeight };
}

function drawRegionWithPadding(
  source: SourceElement,
  region: { x: number; y: number; width: number; height: number },
  targetWidth: number,
  targetHeight: number,
  background = "#071318",
) {
  const { width: frameWidth, height: frameHeight } = getSourceSize(source);
  const canvas = document.createElement("canvas");
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas is not available.");
  context.fillStyle = background;
  context.fillRect(0, 0, targetWidth, targetHeight);

  const sx = Math.max(0, region.x);
  const sy = Math.max(0, region.y);
  const ex = Math.min(frameWidth, region.x + region.width);
  const ey = Math.min(frameHeight, region.y + region.height);
  const sw = Math.max(0, ex - sx);
  const sh = Math.max(0, ey - sy);
  if (sw > 0 && sh > 0) {
    const dx = ((sx - region.x) / region.width) * targetWidth;
    const dy = ((sy - region.y) / region.height) * targetHeight;
    const dw = (sw / region.width) * targetWidth;
    const dh = (sh / region.height) * targetHeight;
    context.drawImage(source, sx, sy, sw, sh, dx, dy, dw, dh);
  }
  return canvas;
}

function createPortraitCrop(source: SourceElement, box: Box) {
  const cropWidth = Math.max(box.width * 1.35, box.height * 0.78);
  const cropHeight = cropWidth * 2;
  const centerX = box.x + box.width / 2;
  const centerY = box.y + box.height * 0.72;
  const canvas = drawRegionWithPadding(
    source,
    { x: centerX - cropWidth / 2, y: centerY - cropHeight / 2, width: cropWidth, height: cropHeight },
    PORTRAIT_WIDTH,
    PORTRAIT_HEIGHT,
  );
  return canvas.toDataURL("image/jpeg", 0.92);
}

function measureFaceQuality(source: SourceElement, box: Box) {
  const sample = drawRegionWithPadding(source, box, 96, 96, "#000");
  const context = sample.getContext("2d", { willReadFrequently: true });
  if (!context) return { brightness: 0, sharpness: 0 };
  const pixels = context.getImageData(0, 0, 96, 96).data;
  const gray = new Float32Array(96 * 96);
  let sum = 0;
  for (let i = 0; i < gray.length; i += 1) {
    const p = i * 4;
    const value = pixels[p] * 0.299 + pixels[p + 1] * 0.587 + pixels[p + 2] * 0.114;
    gray[i] = value;
    sum += value;
  }
  let laplacian = 0;
  let count = 0;
  for (let y = 1; y < 95; y += 1) {
    for (let x = 1; x < 95; x += 1) {
      const idx = y * 96 + x;
      const lap = -4 * gray[idx] + gray[idx - 1] + gray[idx + 1] + gray[idx - 96] + gray[idx + 96];
      laplacian += Math.abs(lap);
      count += 1;
    }
  }
  return { brightness: (sum / gray.length / 255) * 100, sharpness: count ? laplacian / count : 0 };
}

function estimatePose(detection: Detection) {
  const k = detection.keypoints;
  if (k.length < 5) return { pose: "Unknown", roll: 0 };
  const [leftEye, rightEye, nose, leftMouth, rightMouth] = k;
  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const mouthMidX = (leftMouth.x + rightMouth.x) / 2;
  const centerX = (eyeMidX + mouthMidX) / 2;
  const noseOffset = Math.abs(nose.x - centerX) / Math.max(1, detection.box.width);
  const roll = (Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180) / Math.PI;
  return { pose: noseOffset < 0.09 && Math.abs(roll) < 12 ? "Frontal-like" : "Angled", roll };
}

function readiness(box: Box, brightness: number, sharpness: number, pose: string) {
  const passed = [
    Math.min(box.width, box.height) >= 80,
    brightness >= 25 && brightness <= 88,
    sharpness >= 9,
    pose === "Frontal-like",
  ].filter(Boolean).length;
  if (passed === 4) return "Good" as const;
  if (passed >= 2) return "Fair" as const;
  return "Poor" as const;
}

function ageQualityEligible(detection: Detection, brightness: number, sharpness: number, pose: string) {
  return (
    detection.score >= 0.6 &&
    Math.min(detection.box.width, detection.box.height) >= 80 &&
    brightness >= 25 &&
    brightness <= 88 &&
    sharpness >= 7 &&
    pose === "Frontal-like"
  );
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : Math.round((sorted[middle - 1] + sorted[middle]) / 2);
}

function ageRange(age: number) {
  if (age < 10) return "0–9";
  const start = Math.min(90, Math.floor(age / 10) * 10);
  return `${start}–${start + 9}`;
}

function ageGroup(age: number) {
  if (age < 13) return "Child";
  if (age < 18) return "Teen";
  if (age < 25) return "Young adult";
  if (age < 40) return "Adult (25–39)";
  if (age < 60) return "Adult (40–59)";
  return "Senior (60+)";
}

function ageStability(samples: number[], singleFrame = false) {
  if (singleFrame) return "Single frame";
  if (samples.length < 3) return "Collecting";
  const spread = Math.max(...samples) - Math.min(...samples);
  if (spread <= 8) return "Good";
  if (spread <= 15) return "Fair";
  return "Unstable";
}

async function fetchFirst(urls: string[]) {
  let lastError: unknown = null;
  for (const url of urls) {
    try {
      const response = await fetch(url, { cache: "force-cache" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.arrayBuffer();
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Model download failed.");
}

function Metric({ label, value, note }: { label: string; value: string | number; note?: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {note ? <small>{note}</small> : null}
    </div>
  );
}

export default function FaceLabStableAge() {
  const [sourceMode, setSourceMode] = useState<SourceMode>("camera");
  const [threshold, setThreshold] = useState(0.45);
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [ageModelState, setAgeModelState] = useState<"loading" | "ready" | "error">("loading");
  const [detectorLabel, setDetectorLabel] = useState("SCRFD-2.5G");
  const [sourceName, setSourceName] = useState("Choose a source to begin");
  const [imageUrl, setImageUrl] = useState("");
  const [cameraActive, setCameraActive] = useState(false);
  const [running, setRunning] = useState(false);
  const [cameraAspect, setCameraAspect] = useState("4 / 3");
  const [faceCount, setFaceCount] = useState(0);
  const [analysisMs, setAnalysisMs] = useState(0);
  const [frameSize, setFrameSize] = useState("—");
  const [captures, setCaptures] = useState<CaptureEntry[]>([]);
  const [error, setError] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectorRef = useRef<ort.InferenceSession | null>(null);
  const ageRef = useRef<ort.InferenceSession | null>(null);
  const detectorPromiseRef = useRef<Promise<ort.InferenceSession> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef("");
  const loopRef = useRef<number | null>(null);
  const lastRunRef = useRef(0);
  const busyRef = useRef(false);
  const tracksRef = useRef<Track[]>([]);
  const trackIdRef = useRef(1);
  const captureIdRef = useRef(1);
  const ageSamplesRef = useRef<Map<number, number[]>>(new Map());
  const ageBusyRef = useRef<Set<number>>(new Set());

  const ensureModels = useCallback(async () => {
    if (detectorRef.current) return detectorRef.current;
    if (detectorPromiseRef.current) return detectorPromiseRef.current;
    setModelState("loading");
    setError("");

    const promise = (async () => {
      let detectorSession: ort.InferenceSession | null = null;
      let detectorError: unknown = null;
      for (const model of SCRFD_MODELS) {
        try {
          const bytes = await fetchFirst([model.url]);
          detectorSession = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
          setDetectorLabel(model.label);
          break;
        } catch (caught) {
          detectorError = caught;
        }
      }
      if (!detectorSession) throw detectorError instanceof Error ? detectorError : new Error("SCRFD could not be loaded.");
      detectorRef.current = detectorSession;
      setModelState("ready");

      if (!ageRef.current) {
        setAgeModelState("loading");
        void (async () => {
          try {
            const ageBytes = await fetchFirst(AGE_MODELS);
            ageRef.current = await ort.InferenceSession.create(ageBytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
            setAgeModelState("ready");
          } catch {
            setAgeModelState("error");
          }
        })();
      }
      return detectorSession;
    })();

    detectorPromiseRef.current = promise;
    try {
      return await promise;
    } catch (caught) {
      setModelState("error");
      setError(caught instanceof Error ? caught.message : "SCRFD could not be loaded.");
      throw caught;
    } finally {
      detectorPromiseRef.current = null;
    }
  }, []);

  useEffect(() => {
    void ensureModels().catch(() => undefined);
  }, [ensureModels]);

  useEffect(() => {
    return () => {
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      detectorRef.current?.release();
      ageRef.current?.release();
    };
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    tracksRef.current = [];
    setCameraActive(false);
    setRunning(false);
  }, []);

  const syncCameraAspect = useCallback(() => {
    const video = videoRef.current;
    if (video?.videoWidth && video.videoHeight) setCameraAspect(`${video.videoWidth} / ${video.videoHeight}`);
  }, []);

  const prepareScrfdInput = useCallback((source: SourceElement) => {
    const { width, height } = getSourceSize(source);
    const ratio = Math.min(INPUT_SIZE / width, INPUT_SIZE / height);
    const drawWidth = Math.floor(width * ratio);
    const drawHeight = Math.floor(height * ratio);
    const padX = Math.floor((INPUT_SIZE - drawWidth) / 2);
    const padY = Math.floor((INPUT_SIZE - drawHeight) / 2);
    const canvas = document.createElement("canvas");
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas is not available.");
    context.fillStyle = "#000";
    context.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    context.drawImage(source, 0, 0, width, height, padX, padY, drawWidth, drawHeight);
    const pixels = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const area = INPUT_SIZE * INPUT_SIZE;
    const tensorData = new Float32Array(area * 3);
    for (let i = 0; i < area; i += 1) {
      const p = i * 4;
      tensorData[i] = (pixels[p] - 127.5) / 128;
      tensorData[area + i] = (pixels[p + 1] - 127.5) / 128;
      tensorData[area * 2 + i] = (pixels[p + 2] - 127.5) / 128;
    }
    return { tensor: new ort.Tensor("float32", tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]), ratio, padX, padY, width, height };
  }, []);

  const decodeScrfd = useCallback((output: Record<string, ort.Tensor>, session: ort.InferenceSession, prep: { ratio: number; padX: number; padY: number; width: number; height: number }) => {
    const tensors = session.outputNames.map((name) => output[name]);
    if (tensors.length < 9) throw new Error(`Unexpected SCRFD output count: ${tensors.length}`);
    const detections: Detection[] = [];
    const strides = [8, 16, 32];
    for (let level = 0; level < 3; level += 1) {
      const stride = strides[level];
      const scores = tensors[level].data as Float32Array;
      const boxes = tensors[level + 3].data as Float32Array;
      const keypoints = tensors[level + 6].data as Float32Array;
      const numPoints = tensors[level].dims.length >= 2 ? Number(tensors[level].dims[tensors[level].dims.length - 2]) : scores.length;
      const gridWidth = INPUT_SIZE / stride;
      for (let i = 0; i < numPoints; i += 1) {
        const score = Number(scores[i]);
        if (score < threshold) continue;
        const cell = Math.floor(i / 2);
        const cx = cell % gridWidth;
        const cy = Math.floor(cell / gridWidth);
        const boxOffset = i * 4;
        const left = clamp(((cx - Number(boxes[boxOffset])) * stride - prep.padX) / prep.ratio, 0, prep.width - 1);
        const top = clamp(((cy - Number(boxes[boxOffset + 1])) * stride - prep.padY) / prep.ratio, 0, prep.height - 1);
        const right = clamp(((cx + Number(boxes[boxOffset + 2])) * stride - prep.padX) / prep.ratio, 0, prep.width - 1);
        const bottom = clamp(((cy + Number(boxes[boxOffset + 3])) * stride - prep.padY) / prep.ratio, 0, prep.height - 1);
        if (right - left < 2 || bottom - top < 2) continue;
        const points: Point[] = [];
        const kpOffset = i * 10;
        for (let k = 0; k < 10; k += 2) {
          points.push({
            x: clamp(((cx + Number(keypoints[kpOffset + k])) * stride - prep.padX) / prep.ratio, 0, prep.width - 1),
            y: clamp(((cy + Number(keypoints[kpOffset + k + 1])) * stride - prep.padY) / prep.ratio, 0, prep.height - 1),
          });
        }
        detections.push({ box: { x: left, y: top, width: right - left, height: bottom - top }, score, keypoints: points });
      }
    }
    return nonMaximumSuppression(detections);
  }, [threshold]);

  const drawDetections = useCallback((detections: Detection[], width: number, height: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    context.lineWidth = Math.max(2, width / 420);
    context.font = `700 ${Math.max(13, Math.round(width / 70))}px Arial`;
    context.textBaseline = "top";
    detections.forEach((detection, index) => {
      const { x, y, width: boxWidth, height: boxHeight } = detection.box;
      context.strokeStyle = "#58e2d3";
      context.fillStyle = "rgba(88,226,211,0.06)";
      context.strokeRect(x, y, boxWidth, boxHeight);
      context.fillRect(x, y, boxWidth, boxHeight);
      detection.keypoints.forEach((point) => {
        context.beginPath();
        context.arc(point.x, point.y, Math.max(2, width / 350), 0, Math.PI * 2);
        context.fillStyle = "#ffcf5a";
        context.fill();
      });
      const label = `Face ${index + 1} · ${Math.round(detection.score * 100)}%`;
      const labelWidth = context.measureText(label).width + 12;
      const labelHeight = Math.max(22, width / 48);
      context.fillStyle = "#58e2d3";
      context.fillRect(x, Math.max(0, y - labelHeight), labelWidth, labelHeight);
      context.fillStyle = "#04151b";
      context.fillText(label, x + 6, Math.max(1, y - labelHeight + 4));
    });
  }, []);

  const runAgeSample = useCallback(async (source: SourceElement, detection: Detection, captureId: number, singleFrame = false) => {
    const session = ageRef.current;
    if (!session || ageBusyRef.current.has(captureId)) return;
    const existing = ageSamplesRef.current.get(captureId) ?? [];
    if (existing.length >= MAX_AGE_SAMPLES) return;

    const quality = measureFaceQuality(source, detection.box);
    const poseInfo = estimatePose(detection);
    if (!singleFrame && !ageQualityEligible(detection, quality.brightness, quality.sharpness, poseInfo.pose)) {
      setCaptures((items) => items.map((item) => item.id === captureId && item.ageSamples === 0 ? { ...item, ageState: "waiting", ageStability: "Waiting for good face" } : item));
      return;
    }

    ageBusyRef.current.add(captureId);
    try {
      const box = detection.box;
      const side = Math.max(box.width, box.height) * 1.5;
      const centerX = box.x + box.width / 2;
      const centerY = box.y + box.height / 2;
      const ageCanvas = drawRegionWithPadding(source, { x: centerX - side / 2, y: centerY - side / 2, width: side, height: side }, AGE_INPUT_SIZE, AGE_INPUT_SIZE, "#000");
      const context = ageCanvas.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("Canvas unavailable");
      const pixels = context.getImageData(0, 0, AGE_INPUT_SIZE, AGE_INPUT_SIZE).data;
      const area = AGE_INPUT_SIZE * AGE_INPUT_SIZE;
      const input = new Float32Array(area * 3);
      for (let i = 0; i < area; i += 1) {
        const p = i * 4;
        input[i] = pixels[p];
        input[area + i] = pixels[p + 1];
        input[area * 2 + i] = pixels[p + 2];
      }
      const tensor = new ort.Tensor("float32", input, [1, 3, AGE_INPUT_SIZE, AGE_INPUT_SIZE]);
      const started = performance.now();
      const result = await session.run({ [session.inputNames[0]]: tensor });
      const elapsed = performance.now() - started;
      const values = result[session.outputNames[0]].data as Float32Array;
      const sampleAge = clamp(Math.round(Number(values[2]) * 100), 0, 100);
      const nextSamples = [...(ageSamplesRef.current.get(captureId) ?? []), sampleAge].slice(-MAX_AGE_SAMPLES);
      ageSamplesRef.current.set(captureId, nextSamples);
      const med = median(nextSamples);
      const stableEnough = singleFrame || nextSamples.length >= 3;
      setCaptures((items) => items.map((item) => item.id === captureId ? {
        ...item,
        ageRange: stableEnough ? ageRange(med) : undefined,
        ageGroup: stableEnough ? ageGroup(med) : undefined,
        ageSamples: nextSamples.length,
        ageStability: ageStability(nextSamples, singleFrame),
        ageMs: elapsed,
        ageState: singleFrame || nextSamples.length >= MAX_AGE_SAMPLES ? "ready" : "sampling",
      } : item));
    } catch {
      setCaptures((items) => items.map((item) => item.id === captureId ? { ...item, ageState: "unavailable", ageStability: "Unavailable" } : item));
    } finally {
      ageBusyRef.current.delete(captureId);
    }
  }, []);

  const captureNewFace = useCallback((source: SourceElement, detection: Detection, detectionMs: number, singleFrame = false) => {
    const { width: frameWidth, height: frameHeight } = getSourceSize(source);
    const quality = measureFaceQuality(source, detection.box);
    const poseInfo = estimatePose(detection);
    const captureId = captureIdRef.current++;
    const box = detection.box;
    const entry: CaptureEntry = {
      id: captureId,
      dataUrl: createPortraitCrop(source, box),
      confidence: detection.score,
      x: Math.round(box.x),
      y: Math.round(box.y),
      width: Math.round(box.width),
      height: Math.round(box.height),
      frameWidth,
      frameHeight,
      coverage: ((box.width * box.height) / (frameWidth * frameHeight)) * 100,
      faceRatio: box.width / Math.max(1, box.height),
      brightness: quality.brightness,
      sharpness: quality.sharpness,
      pose: poseInfo.pose,
      roll: poseInfo.roll,
      frReadiness: readiness(box, quality.brightness, quality.sharpness, poseInfo.pose),
      capturedAt: new Date().toLocaleTimeString(),
      source: sourceMode === "camera" ? sourceName : "Uploaded image",
      detectionMs,
      ageSamples: 0,
      ageStability: ageModelState === "error" ? "Unavailable" : singleFrame ? "Single frame" : "Waiting for good face",
      ageState: ageModelState === "error" ? "unavailable" : "waiting",
    };
    ageSamplesRef.current.set(captureId, []);
    setCaptures((items) => [...items, entry]);

    if (ageModelState === "ready") {
      void runAgeSample(source, detection, captureId, singleFrame);
    } else if (ageModelState === "loading") {
      window.setTimeout(() => {
        if (ageRef.current) void runAgeSample(source, detection, captureId, singleFrame);
        else setCaptures((items) => items.map((item) => item.id === captureId ? { ...item, ageState: "unavailable", ageStability: "Unavailable" } : item));
      }, 1800);
    }
    return captureId;
  }, [ageModelState, runAgeSample, sourceMode, sourceName]);

  const updateTracksAndCapture = useCallback((source: SourceElement, detections: Detection[], detectionMs: number) => {
    const now = performance.now();
    tracksRef.current = tracksRef.current.filter((track) => now - track.lastSeen <= TRACK_TTL_MS);
    const matchedTracks = new Set<number>();

    for (const detection of detections) {
      let bestIndex = -1;
      let bestOverlap = 0;
      tracksRef.current.forEach((track, index) => {
        if (matchedTracks.has(index)) return;
        const overlap = iou(track.box, detection.box);
        if (overlap > bestOverlap) {
          bestOverlap = overlap;
          bestIndex = index;
        }
      });

      if (bestIndex >= 0 && bestOverlap >= TRACK_IOU_THRESHOLD) {
        matchedTracks.add(bestIndex);
        const track = tracksRef.current[bestIndex];
        tracksRef.current[bestIndex] = { ...track, box: detection.box, lastSeen: now };
        const samples = ageSamplesRef.current.get(track.captureId) ?? [];
        if (ageModelState === "ready" && samples.length < MAX_AGE_SAMPLES && now - track.lastAgeSampleAt >= AGE_SAMPLE_INTERVAL_MS) {
          tracksRef.current[bestIndex].lastAgeSampleAt = now;
          void runAgeSample(source, detection, track.captureId);
        }
      } else {
        const captureId = captureNewFace(source, detection, detectionMs);
        tracksRef.current.push({ id: trackIdRef.current++, box: detection.box, lastSeen: now, captureId, lastAgeSampleAt: now });
      }
    }
  }, [ageModelState, captureNewFace, runAgeSample]);

  const processSource = useCallback(async (source: SourceElement, live: boolean) => {
    if (busyRef.current) return;
    const { width, height } = getSourceSize(source);
    if (!width || !height) return;
    busyRef.current = true;
    try {
      const session = await ensureModels();
      const prep = prepareScrfdInput(source);
      const started = performance.now();
      const output = await session.run({ [session.inputNames[0]]: prep.tensor });
      const elapsed = performance.now() - started;
      const detections = decodeScrfd(output, session, prep);
      setFaceCount(detections.length);
      setAnalysisMs(elapsed);
      setFrameSize(`${width} × ${height}`);
      drawDetections(detections, width, height);
      if (live) updateTracksAndCapture(source, detections, elapsed);
      else {
        tracksRef.current = [];
        detections.forEach((detection) => captureNewFace(source, detection, elapsed, true));
      }
    } catch (caught) {
      setRunning(false);
      setError(caught instanceof Error ? caught.message : "Face detection failed.");
    } finally {
      busyRef.current = false;
    }
  }, [captureNewFace, decodeScrfd, drawDetections, ensureModels, prepareScrfdInput, updateTracksAndCapture]);

  useEffect(() => {
    if (!running || sourceMode !== "camera") return;
    let cancelled = false;
    const tick = (now: number) => {
      if (cancelled) return;
      if (!busyRef.current && now - lastRunRef.current >= LIVE_INTERVAL_MS && videoRef.current) {
        lastRunRef.current = now;
        void processSource(videoRef.current, true);
      }
      loopRef.current = requestAnimationFrame(tick);
    };
    loopRef.current = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      loopRef.current = null;
    };
  }, [processSource, running, sourceMode]);

  const startCamera = useCallback(async (facingMode: FacingMode) => {
    stopCamera();
    tracksRef.current = [];
    setSourceMode("camera");
    setSourceName(facingMode === "environment" ? "Back camera" : "Front camera / Webcam");
    setError("");
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires HTTPS and a supported browser.");
      await ensureModels();
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: facingMode }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } },
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
      }
      streamRef.current = stream;
      if (!videoRef.current) throw new Error("Camera viewport is not ready.");
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      syncCameraAspect();
      setCameraActive(true);
      lastRunRef.current = 0;
      setRunning(true);
    } catch (caught) {
      stopCamera();
      setError(caught instanceof Error ? caught.message : "Camera permission was not granted.");
    }
  }, [ensureModels, stopCamera, syncCameraAspect]);

  const handleImage = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    stopCamera();
    tracksRef.current = [];
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setImageUrl(url);
    setSourceMode("image");
    setSourceName(file.name);
    setError("");
    setFaceCount(0);
    setAnalysisMs(0);
    event.target.value = "";
  }, [stopCamera]);

  const analyseImage = useCallback(async () => {
    if (!imageRef.current || !imageUrl) return;
    setCaptures([]);
    captureIdRef.current = 1;
    ageSamplesRef.current.clear();
    await processSource(imageRef.current, false);
  }, [imageUrl, processSource]);

  const clearList = () => {
    setCaptures([]);
    captureIdRef.current = 1;
    tracksRef.current = [];
    ageSamplesRef.current.clear();
    ageBusyRef.current.clear();
  };

  return (
    <main className="page-shell face-lab">
      <header className="hero">
        <p className="eyebrow">CCTV face analytics test lab</p>
        <h1>Face Detection + Stable Age Range</h1>
        <p className="intro">
          SCRFD detects faces and five landmarks. A new face is logged once. For live camera faces, the age stage samples several good-quality frames and reports a range instead of one exact age.
        </p>
        <div className="badges">
          <span>{detectorLabel} · 640px</span>
          <span>ONNX Runtime · WASM</span>
          <span>Age model: {ageModelState === "ready" ? "READY" : ageModelState.toUpperCase()}</span>
        </div>
        <a className="lab-link" href={window.location.pathname}>← Human + Vehicle demo</a>
      </header>

      <section className="panel viewport-panel">
        <div className="panel-head">
          <div><h2>Face detection viewport</h2><p>{sourceName}</p></div>
          <div className={`live-state ${running ? "active" : ""}`}><i />{modelState === "loading" ? "Loading SCRFD" : running ? "Detecting" : "Ready"}</div>
        </div>
        <div className="media-stage">
          {sourceMode === "camera" ? (
            <div className="media-layer face-camera-layer" style={{ aspectRatio: cameraAspect }}>
              <video ref={videoRef} playsInline muted className="media" onLoadedMetadata={syncCameraAspect} />
              {!cameraActive ? <div className="empty-state">Select a camera below to start.</div> : null}
              <canvas ref={canvasRef} className="overlay" />
            </div>
          ) : imageUrl ? (
            <div className="media-layer image-layer">
              <img ref={imageRef} src={imageUrl} alt="Selected for face detection" className="media" />
              <canvas ref={canvasRef} className="overlay" />
            </div>
          ) : <div className="empty-state image-empty">Choose an image or camera below.</div>}
        </div>
        <div className="metrics face-metrics">
          <Metric label="Faces now" value={faceCount} />
          <Metric label="Detection" value={formatMs(analysisMs)} />
          <Metric label="Source" value={frameSize} />
          <Metric label="Captured" value={captures.length} note="persistent log" />
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title"><span>1</span><div><h2>Choose source</h2><p>For live tests, one continuous face creates one row. Age sampling continues on that same row while the face remains in view.</p></div></div>
        <div className="source-grid">
          <button onClick={() => void startCamera("environment")}>Back camera</button>
          <button className="primary" onClick={() => void startCamera("user")}>Front camera / Webcam</button>
          <label className="button-like">Upload image<input type="file" accept="image/*" onChange={handleImage} /></label>
          {sourceMode === "image" ? (
            <button onClick={() => void analyseImage()} disabled={!imageUrl || modelState !== "ready"}>Analyse image</button>
          ) : cameraActive ? (
            <button onClick={() => setRunning((value) => !value)}>{running ? "Pause detection" : "Resume detection"}</button>
          ) : null}
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title"><span>2</span><div><h2>Face confidence</h2><p>SCRFD detection threshold. Age sampling additionally requires a sufficiently large, frontal, reasonably sharp and well-lit face.</p></div></div>
        <div className="slider-row">
          <input type="range" min="20" max="85" step="1" value={Math.round(threshold * 100)} onChange={(event) => setThreshold(Number(event.target.value) / 100)} />
          <strong>{Math.round(threshold * 100)}%</strong>
        </div>
      </section>

      <section className="panel face-results-panel">
        <div className="panel-head">
          <div><h2>Face capture log</h2><p>Age range is an AI estimate from up to {MAX_AGE_SAMPLES} accepted live samples. It is not verified demographic data.</p></div>
          <div className="capture-actions"><strong className="crop-count">{captures.length} captured</strong><button onClick={clearList}>Clear list</button></div>
        </div>

        {captures.length ? (
          <div className="face-log-list">
            {captures.map((capture) => (
              <article className="face-log-row" key={capture.id}>
                <div className="face-log-number">#{capture.id}</div>
                <img className="face-log-crop" src={capture.dataUrl} alt={`Captured face ${capture.id}`} />
                <div className="face-log-main">
                  <div className="face-log-heading"><strong>Captured face {capture.id}</strong><span>{Math.round(capture.confidence * 100)}% confidence</span></div>
                  <div className="face-chip-row">
                    <span><b>TIME</b> {capture.capturedAt}</span>
                    <span><b>AGE RANGE</b> {capture.ageRange ?? (capture.ageState === "unavailable" ? "unavailable" : capture.ageSamples ? "collecting…" : "waiting…")}</span>
                    <span><b>AGE GROUP</b> {capture.ageGroup ?? "—"}</span>
                    <span><b>AGE SAMPLES</b> {capture.ageSamples}/{sourceMode === "image" ? 1 : MAX_AGE_SAMPLES}</span>
                    <span><b>AGE STABILITY</b> {capture.ageStability}</span>
                    <span><b>POSE</b> {capture.pose}</span>
                    <span><b>FR READINESS</b> {capture.frReadiness}</span>
                    <span><b>FACE</b> {capture.width} × {capture.height}px</span>
                    <span><b>BOX</b> X {capture.x}, Y {capture.y}</span>
                    <span><b>FRAME</b> {capture.frameWidth} × {capture.frameHeight}</span>
                    <span><b>COVERAGE</b> {capture.coverage.toFixed(2)}%</span>
                    <span><b>FACE RATIO</b> {capture.faceRatio.toFixed(2)}</span>
                    <span><b>BRIGHTNESS</b> {Math.round(capture.brightness)}%</span>
                    <span><b>SHARPNESS</b> {capture.sharpness.toFixed(1)}</span>
                    <span><b>ROLL</b> {capture.roll.toFixed(1)}°</span>
                    <span><b>DETECTION</b> {formatMs(capture.detectionMs)}</span>
                    <span><b>AGE MODEL</b> {formatMs(capture.ageMs)}</span>
                    <span><b>CROP</b> 160 × 320px · 1:2</span>
                  </div>
                </div>
                <a className="crop-download face-log-save" href={capture.dataUrl} download={`face-${capture.id}.jpg`}>Save</a>
              </article>
            ))}
          </div>
        ) : <div className="face-empty">New face arrivals will be added here line by line.</div>}
      </section>

      {error ? <div className="error-box">{error}</div> : null}

      <footer className="footer-card">
        <div><strong>SCRFD detection</strong><p>Face box, confidence and five landmarks are produced by SCRFD. Tracking prevents duplicate rows while the same face stays in view.</p></div>
        <div><strong>Multi-frame age range</strong><p>Good-quality live frames are sampled repeatedly. The median estimate is converted to a range, and sample spread is shown as age stability. Exact age is intentionally not displayed.</p></div>
        <div><strong>Quality metadata</strong><p>Brightness, sharpness, pose and FR readiness are local diagnostics/heuristics for CCTV testing. Gender output is not displayed or stored.</p></div>
      </footer>
      <p className="model-license-note">Research test note: InsightFace/SCRFD pretrained weights have non-commercial research licensing. Review model licensing before any commercial deployment.</p>
    </main>
  );
}
