"use client";

import {
  Camera,
  CheckCircle2,
  Cpu,
  Gauge,
  Layers3,
  LoaderCircle,
  Pause,
  Play,
  RefreshCw,
  ScanSearch,
  Smartphone,
  Target,
  Upload,
  XCircle,
  Zap,
} from "lucide-react";
import {
  ChangeEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type YoloBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  cls: number;
  name: string;
};

type Detection = YoloBox & {
  trackId?: number;
  origin: "full" | "roi" | "track";
};

type YoloResults = {
  width: number;
  height: number;
  boxes: YoloBox[];
  speed?: {
    preprocess?: number;
    inference?: number;
    postprocess?: number;
  };
};

type YoloModel = {
  predict: (
    input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    options?: { conf?: number; iou?: number; classes?: number[] },
  ) => Promise<YoloResults>;
  free: () => void;
  device: string;
};

type YoloRuntime = {
  YOLO: {
    load: (
      source: Uint8Array | string,
      options?: { device?: "auto" | "webgpu" | "cpu" },
    ) => Promise<YoloModel>;
  };
};

type ModelChoice = "yolo26n" | "yolo26s";
type SourceMode = "camera" | "image";
type FacingMode = "environment" | "user";
type HybridMode = "fast" | "smart";

type Region = {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
};

type Track = Detection & {
  trackId: number;
  missed: number;
};

const RUNTIME_URL =
  "https://cdn.jsdelivr.net/npm/@ultralytics/yolo@0.0.41/dist/index.js";

const MODEL_CHOICES: Record<
  ModelChoice,
  { name: string; detail: string; url: string; badge: string }
> = {
  yolo26n: {
    name: "YOLO26-N",
    detail: "2.4M parameters · speed baseline",
    url: "https://huggingface.co/prithivMLmods/YOLO26-ONNX/resolve/main/yolo26n/yolo26n.onnx?download=true",
    badge: "Fastest",
  },
  yolo26s: {
    name: "YOLO26-S",
    detail: "9.5M parameters · stronger small-target baseline",
    url: "https://huggingface.co/prithivMLmods/YOLO26-ONNX/resolve/main/yolo26s/yolo26s.onnx?download=true",
    badge: "Accuracy",
  },
};

// COCO ids: person, bicycle, car, motorcycle, bus, truck.
const HUMAN_VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7];
const VEHICLE_NAMES = new Set(["bicycle", "car", "motorcycle", "bus", "truck"]);
const TARGET_INTERVAL_MS = 50;
const TRACK_MISSES = 3;

function isHuman(box: YoloBox) {
  return box.name.toLowerCase() === "person";
}

function isVehicle(box: YoloBox) {
  return VEHICLE_NAMES.has(box.name.toLowerCase());
}

function kind(box: YoloBox) {
  return isHuman(box) ? "human" : "vehicle";
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function area(box: YoloBox) {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

function intersection(a: YoloBox, b: YoloBox) {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  return w * h;
}

function iou(a: YoloBox, b: YoloBox) {
  const inter = intersection(a, b);
  if (!inter) return 0;
  const union = area(a) + area(b) - inter;
  return union ? inter / union : 0;
}

function duplicate(a: YoloBox, b: YoloBox) {
  if (kind(a) !== kind(b)) return false;
  const inter = intersection(a, b);
  if (!inter) return false;
  const overlap = iou(a, b);
  const containment = inter / Math.max(1, Math.min(area(a), area(b)));
  return overlap >= 0.45 || containment >= 0.82;
}

function mergeDetections(input: Detection[]) {
  const merged: Detection[] = [];
  const sorted = [...input].sort((a, b) => b.conf - a.conf);
  for (const candidate of sorted) {
    if (!merged.some((kept) => duplicate(kept, candidate))) {
      merged.push(candidate);
    }
  }
  return merged;
}

function sourceSize(source: HTMLVideoElement | HTMLImageElement) {
  if (source instanceof HTMLVideoElement) {
    return { width: source.videoWidth, height: source.videoHeight };
  }
  return { width: source.naturalWidth, height: source.naturalHeight };
}

function normalizeBoxes(
  result: YoloResults,
  targetWidth: number,
  targetHeight: number,
  offsetX = 0,
  offsetY = 0,
  origin: Detection["origin"] = "full",
): Detection[] {
  const sx = targetWidth / Math.max(1, result.width);
  const sy = targetHeight / Math.max(1, result.height);
  return result.boxes.map((box) => ({
    ...box,
    x1: box.x1 * sx + offsetX,
    y1: box.y1 * sy + offsetY,
    x2: box.x2 * sx + offsetX,
    y2: box.y2 * sy + offsetY,
    origin,
  }));
}

function cropRegion(
  source: HTMLVideoElement | HTMLImageElement,
  region: Region,
) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(region.width));
  canvas.height = Math.max(1, Math.round(region.height));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Could not create the smart ROI canvas.");
  context.drawImage(
    source,
    region.x,
    region.y,
    region.width,
    region.height,
    0,
    0,
    canvas.width,
    canvas.height,
  );
  return canvas;
}

function chooseSmartRegion(
  detections: Detection[],
  width: number,
  height: number,
  threshold: number,
): Region {
  const candidates = detections.filter((box) => {
    const bw = box.x2 - box.x1;
    const bh = box.y2 - box.y1;
    return bh < height * 0.12 || bw < width * 0.10 || box.conf < Math.max(0.38, threshold + 0.12);
  });

  if (!candidates.length) {
    return {
      x: width * 0.14,
      y: height * 0.18,
      width: width * 0.72,
      height: height * 0.52,
      label: "Perspective far-zone fallback",
    };
  }

  const cols = 4;
  const rows = 3;
  const scores = Array.from({ length: cols * rows }, () => 0);
  candidates.forEach((box) => {
    const cx = (box.x1 + box.x2) / 2;
    const cy = (box.y1 + box.y2) / 2;
    const col = clamp(Math.floor((cx / width) * cols), 0, cols - 1);
    const row = clamp(Math.floor((cy / height) * rows), 0, rows - 1);
    const bh = Math.max(1, box.y2 - box.y1);
    const tinyWeight = clamp((height * 0.12) / bh, 1, 3);
    const weakWeight = box.conf < 0.4 ? 1.35 : 1;
    scores[row * cols + col] += tinyWeight * weakWeight;
  });

  let bestIndex = 0;
  for (let i = 1; i < scores.length; i += 1) {
    if (scores[i] > scores[bestIndex]) bestIndex = i;
  }
  const bestCol = bestIndex % cols;
  const bestRow = Math.floor(bestIndex / cols);

  const cluster = candidates.filter((box) => {
    const cx = (box.x1 + box.x2) / 2;
    const cy = (box.y1 + box.y2) / 2;
    const col = clamp(Math.floor((cx / width) * cols), 0, cols - 1);
    const row = clamp(Math.floor((cy / height) * rows), 0, rows - 1);
    return Math.abs(col - bestCol) <= 1 && Math.abs(row - bestRow) <= 1;
  });

  let x1 = Math.min(...cluster.map((box) => box.x1));
  let y1 = Math.min(...cluster.map((box) => box.y1));
  let x2 = Math.max(...cluster.map((box) => box.x2));
  let y2 = Math.max(...cluster.map((box) => box.y2));
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;
  const minWidth = width * 0.38;
  const minHeight = height * 0.38;
  const paddedWidth = Math.min(width * 0.74, Math.max(minWidth, (x2 - x1) * 2.4));
  const paddedHeight = Math.min(height * 0.68, Math.max(minHeight, (y2 - y1) * 2.6));
  x1 = clamp(centerX - paddedWidth / 2, 0, width - paddedWidth);
  y1 = clamp(centerY - paddedHeight / 2, 0, height - paddedHeight);
  x2 = x1 + paddedWidth;
  y2 = y1 + paddedHeight;

  return {
    x: x1,
    y: y1,
    width: x2 - x1,
    height: y2 - y1,
    label: "AutoFocus-style small-object cluster",
  };
}

function formatMs(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "—";
  return value < 10 ? value.toFixed(1) + " ms" : Math.round(value) + " ms";
}

function bandLabel(height: number) {
  if (height < 16) return "<16";
  if (height < 32) return "16–32";
  if (height < 64) return "32–64";
  if (height < 128) return "64–128";
  return ">128";
}

export default function SmartPage() {
  const [modelChoice, setModelChoice] = useState<ModelChoice>("yolo26s");
  const [mode, setMode] = useState<HybridMode>("smart");
  const [sourceMode, setSourceMode] = useState<SourceMode>("image");
  const [threshold, setThreshold] = useState(0.2);
  const [refineEvery, setRefineEvery] = useState(3);
  const [running, setRunning] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [modelState, setModelState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [modelProgress, setModelProgress] = useState(0);
  const [modelFile, setModelFile] = useState("");
  const [device, setDevice] = useState("waiting");
  const [boxes, setBoxes] = useState<Detection[]>([]);
  const [totalMs, setTotalMs] = useState(0);
  const [fullInferenceMs, setFullInferenceMs] = useState(0);
  const [roiInferenceMs, setRoiInferenceMs] = useState(0);
  const [roiGain, setRoiGain] = useState(0);
  const [passes, setPasses] = useState(0);
  const [roi, setRoi] = useState<Region | null>(null);
  const [error, setError] = useState("");
  const [imageUrl, setImageUrl] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<YoloModel | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef("");
  const busyRef = useRef(false);
  const loopRef = useRef<number | null>(null);
  const lastStartRef = useRef(0);
  const loadIdRef = useRef(0);
  const scanCounterRef = useRef(0);
  const nextTrackIdRef = useRef(1);
  const tracksRef = useRef<Track[]>([]);

  const activeChoice = MODEL_CHOICES[modelChoice];
  const humans = useMemo(() => boxes.filter(isHuman).length, [boxes]);
  const vehicles = useMemo(() => boxes.filter(isVehicle).length, [boxes]);
  const effectiveFps = totalMs ? Math.min(20, 1000 / totalMs) : 0;
  const bands = useMemo(() => {
    const result: Record<string, number> = { "<16": 0, "16–32": 0, "32–64": 0, "64–128": 0, ">128": 0 };
    boxes.forEach((box) => {
      result[bandLabel(box.y2 - box.y1)] += 1;
    });
    return result;
  }, [boxes]);

  const resetTracking = useCallback(() => {
    tracksRef.current = [];
    nextTrackIdRef.current = 1;
    scanCounterRef.current = 0;
  }, []);

  const stopCamera = useCallback(() => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setRunning(false);
    resetTracking();
  }, [resetTracking]);

  useEffect(() => {
    return () => {
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      modelRef.current?.free();
    };
  }, []);

  const loadModel = useCallback(async () => {
    const loadId = ++loadIdRef.current;
    setModelState("loading");
    setModelProgress(0);
    setModelFile("Downloading " + activeChoice.name);
    setError("");
    setBoxes([]);
    resetTracking();

    try {
      modelRef.current?.free();
      modelRef.current = null;
      const response = await fetch(activeChoice.url);
      if (!response.ok) throw new Error("Model download failed with HTTP " + response.status + ".");

      const total = Number(response.headers.get("content-length") || 0);
      let bytes: Uint8Array;
      if (response.body && total > 0) {
        const reader = response.body.getReader();
        const chunks: Uint8Array[] = [];
        let received = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            chunks.push(value);
            received += value.length;
            if (loadId === loadIdRef.current) {
              setModelProgress(Math.min(85, Math.round((received / total) * 85)));
              setModelFile(activeChoice.name + " · " + (received / 1024 / 1024).toFixed(1) + " MB");
            }
          }
        }
        bytes = new Uint8Array(received);
        let offset = 0;
        chunks.forEach((chunk) => {
          bytes.set(chunk, offset);
          offset += chunk.length;
        });
      } else {
        bytes = new Uint8Array(await response.arrayBuffer());
        setModelProgress(85);
      }

      setModelFile("Initializing WebGPU / WASM runtime");
      const moduleUrl: string = RUNTIME_URL;
      const runtime = (await import(/* @vite-ignore */ moduleUrl)) as unknown as YoloRuntime;
      const prefersWebGpu = typeof navigator !== "undefined" && "gpu" in navigator;
      const model = await runtime.YOLO.load(bytes, {
        device: prefersWebGpu ? "webgpu" : "cpu",
      });
      if (loadId !== loadIdRef.current) {
        model.free();
        return;
      }
      modelRef.current = model;
      setDevice(model.device);
      setModelState("ready");
      setModelProgress(100);
      setModelFile(activeChoice.name + " ready");
    } catch (caught) {
      if (loadId !== loadIdRef.current) return;
      setModelState("error");
      setError(caught instanceof Error ? caught.message : "The model could not load.");
    }
  }, [activeChoice.name, activeChoice.url, resetTracking]);

  useEffect(() => {
    void loadModel();
  }, [loadModel]);

  const updateTracks = useCallback((detections: Detection[]) => {
    const oldTracks = tracksRef.current;
    const used = new Set<number>();
    const next: Track[] = [];

    detections.forEach((detection) => {
      let bestIndex = -1;
      let bestScore = 0;
      oldTracks.forEach((track, index) => {
        if (used.has(index) || kind(track) !== kind(detection)) return;
        const score = iou(track, detection);
        if (score > bestScore) {
          bestScore = score;
          bestIndex = index;
        }
      });

      if (bestIndex >= 0 && bestScore >= 0.22) {
        used.add(bestIndex);
        const existing = oldTracks[bestIndex];
        next.push({ ...detection, trackId: existing.trackId, missed: 0 });
      } else {
        next.push({ ...detection, trackId: nextTrackIdRef.current++, missed: 0 });
      }
    });

    oldTracks.forEach((track, index) => {
      if (used.has(index)) return;
      const missed = track.missed + 1;
      if (missed <= TRACK_MISSES) {
        next.push({
          ...track,
          conf: track.conf * 0.96,
          missed,
          origin: "track",
        });
      }
    });

    tracksRef.current = next;
    return next as Detection[];
  }, []);

  const draw = useCallback((nextBoxes: Detection[], width: number, height: number, activeRoi: Region | null) => {
    const canvas = overlayRef.current;
    if (!canvas || !width || !height) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);

    if (activeRoi) {
      context.save();
      context.strokeStyle = "#f472b6";
      context.lineWidth = Math.max(2, width / 650);
      context.setLineDash([12, 8]);
      context.strokeRect(activeRoi.x, activeRoi.y, activeRoi.width, activeRoi.height);
      context.setLineDash([]);
      context.fillStyle = "#f472b6";
      context.font = "700 " + Math.max(12, Math.round(width / 95)) + "px Arial";
      context.fillText("SMART ROI · " + activeRoi.label, activeRoi.x + 8, Math.max(18, activeRoi.y + 18));
      context.restore();
    }

    context.lineWidth = Math.max(2, width / 500);
    context.font = "700 " + Math.max(12, Math.round(width / 90)) + "px Arial";
    context.textBaseline = "top";
    nextBoxes.forEach((box) => {
      const human = isHuman(box);
      const color = box.origin === "roi" ? "#f472b6" : human ? "#5eead4" : "#fcd34d";
      const id = box.trackId ? " #" + box.trackId : "";
      const label = (human ? "Human" : box.name) + id + " " + Math.round(box.conf * 100) + "%";
      const x = clamp(box.x1, 0, width);
      const y = clamp(box.y1, 0, height);
      const w = Math.max(1, clamp(box.x2, 0, width) - x);
      const h = Math.max(1, clamp(box.y2, 0, height) - y);
      context.strokeStyle = color;
      context.fillStyle = color + "16";
      context.strokeRect(x, y, w, h);
      context.fillRect(x, y, w, h);
      const metrics = context.measureText(label);
      const labelH = Math.max(20, width / 58);
      context.fillStyle = color;
      context.fillRect(x, Math.max(0, y - labelH), metrics.width + 12, labelH);
      context.fillStyle = "#021018";
      context.fillText(label, x + 6, Math.max(1, y - labelH + 3));
    });
  }, []);

  const analyse = useCallback(async () => {
    if (busyRef.current || !modelRef.current) return;
    const source = sourceMode === "camera" ? videoRef.current : imageRef.current;
    if (!source) return;
    const { width, height } = sourceSize(source);
    if (!width || !height) return;

    busyRef.current = true;
    const started = performance.now();
    try {
      const fullResult = await modelRef.current.predict(source, {
        conf: Math.max(0.05, threshold),
        iou: 0.7,
        classes: HUMAN_VEHICLE_CLASSES,
      });
      let full = normalizeBoxes(fullResult, width, height, 0, 0, "full").filter(
        (box) => (isHuman(box) || isVehicle(box)) && box.conf >= threshold,
      );
      full = mergeDetections(full);

      scanCounterRef.current += 1;
      const shouldRefine =
        mode === "smart" &&
        (sourceMode === "image" || scanCounterRef.current % Math.max(1, refineEvery) === 0);

      let activeRoi: Region | null = null;
      let roiBoxes: Detection[] = [];
      let roiMs = 0;
      let passCount = 1;

      if (shouldRefine) {
        activeRoi = chooseSmartRegion(full, width, height, threshold);
        const crop = cropRegion(source, activeRoi);
        const roiResult = await modelRef.current.predict(crop, {
          conf: Math.max(0.05, threshold * 0.82),
          iou: 0.7,
          classes: HUMAN_VEHICLE_CLASSES,
        });
        roiMs = roiResult.speed?.inference || 0;
        roiBoxes = normalizeBoxes(
          roiResult,
          activeRoi.width,
          activeRoi.height,
          activeRoi.x,
          activeRoi.y,
          "roi",
        ).filter(
          (box) => (isHuman(box) || isVehicle(box)) && box.conf >= threshold * 0.82,
        );
        roiBoxes = mergeDetections(roiBoxes);
        passCount = 2;
      }

      const merged = mergeDetections([...full, ...roiBoxes]);
      const gain = Math.max(0, merged.length - full.length);
      const displayed = sourceMode === "camera" ? updateTracks(merged) : merged;
      const elapsed = performance.now() - started;

      setBoxes(displayed);
      setTotalMs(elapsed);
      setFullInferenceMs(fullResult.speed?.inference || 0);
      setRoiInferenceMs(roiMs);
      setRoiGain(gain);
      setPasses(passCount);
      setRoi(activeRoi);
      setDevice(modelRef.current.device);
      draw(displayed, width, height, activeRoi);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Smart inference failed.");
      setRunning(false);
    } finally {
      busyRef.current = false;
    }
  }, [draw, mode, refineEvery, sourceMode, threshold, updateTracks]);

  useEffect(() => {
    if (!running || sourceMode !== "camera") return;
    let cancelled = false;
    const tick = (now: number) => {
      if (cancelled) return;
      if (!busyRef.current && now - lastStartRef.current >= TARGET_INTERVAL_MS) {
        lastStartRef.current = now;
        void analyse();
      }
      loopRef.current = requestAnimationFrame(tick);
    };
    loopRef.current = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      loopRef.current = null;
    };
  }, [analyse, running, sourceMode]);

  const startCamera = useCallback(async (facing: FacingMode) => {
    stopCamera();
    setSourceMode("camera");
    setError("");
    setBoxes([]);
    setRoi(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera requires HTTPS and a supported browser.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 },
        },
      });
      cameraStreamRef.current = stream;
      setCameraActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      lastStartRef.current = 0;
      setRunning(true);
    } catch (caught) {
      stopCamera();
      setError(caught instanceof Error ? caught.message : "Camera permission was not granted.");
    }
  }, [stopCamera]);

  const handleImage = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    stopCamera();
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setImageUrl(url);
    setSourceMode("image");
    setBoxes([]);
    setRoi(null);
    setTotalMs(0);
    setFullInferenceMs(0);
    setRoiInferenceMs(0);
    setRoiGain(0);
    resetTracking();
    event.target.value = "";
  }, [resetTracking, stopCamera]);

  const switchModel = useCallback((next: ModelChoice) => {
    if (next === modelChoice) return;
    loadIdRef.current += 1;
    modelRef.current?.free();
    modelRef.current = null;
    setModelChoice(next);
    setModelState("idle");
    setBoxes([]);
    setTotalMs(0);
    resetTracking();
  }, [modelChoice, resetTracking]);

  return (
    <main className="min-h-screen bg-[#021018] text-slate-100">
      <header className="border-b border-white/10 bg-[#03151d]/95">
        <div className="mx-auto flex max-w-[1540px] flex-col gap-4 px-5 py-5 sm:px-8 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="mb-1 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.32em] text-pink-300">
              <Zap size={14} /> Smart tiny-target experiment
            </div>
            <h1 className="text-xl font-bold sm:text-2xl">Adaptive Human + Vehicle CCTV Detector</h1>
            <p className="mt-1 max-w-3xl text-xs leading-5 text-slate-400">
              Fast full-frame scan + AutoFocus-style cluster ROI + enlarged second pass + temporal track persistence.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5">Device: <strong className="text-cyan-200">{device.toUpperCase()}</strong></span>
            <a href="./live.html" className="rounded-full border border-cyan-300/25 bg-cyan-300/10 px-3 py-1.5 font-semibold text-cyan-100">50 ms Live</a>
            <a href="./" className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-slate-300">D-FINE Lab</a>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1540px] gap-5 px-4 py-6 sm:px-8 xl:grid-cols-[minmax(0,1fr)_410px]">
        <section className="min-w-0 overflow-hidden rounded-[28px] border border-white/10 bg-[#061d26]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
            <div className="flex items-center gap-3">
              <ScanSearch className="text-pink-300" size={22} />
              <div>
                <h2 className="text-sm font-semibold">Smart analysis viewport</h2>
                <p className="text-xs text-slate-400">Pink ROI = area selected for tiny-target refinement</p>
              </div>
            </div>
            <span className="text-xs text-slate-400">{mode === "smart" ? "Adaptive smart mode" : "Fast 1-pass baseline"}</span>
          </div>

          <div className="relative flex min-h-[390px] items-center justify-center overflow-hidden bg-black lg:min-h-[610px]">
            {sourceMode === "camera" ? (
              <div className="relative w-full">
                <video ref={videoRef} muted playsInline autoPlay className={"block h-auto min-h-[390px] w-full object-contain lg:min-h-[610px] " + (cameraActive ? "" : "opacity-0")} />
                <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
                {!cameraActive ? <div className="absolute inset-0 grid place-items-center p-8 text-center text-slate-400">Start rear camera or webcam.</div> : null}
              </div>
            ) : imageUrl ? (
              <div className="relative w-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img ref={imageRef} src={imageUrl} alt="Smart tiny-target test" className="block h-auto w-full" onLoad={() => void analyse()} />
                <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
              </div>
            ) : (
              <div className="p-10 text-center text-slate-400">Upload the same traffic/CCTV image to compare against 1-pass YOLO.</div>
            )}

            {modelState === "loading" ? (
              <div className="absolute left-4 top-4 max-w-[320px] rounded-2xl border border-cyan-200/20 bg-[#03151ded] px-4 py-3 shadow-2xl backdrop-blur">
                <div className="flex items-center gap-3"><LoaderCircle className="animate-spin text-cyan-300" size={20} /><div><div className="text-xs font-semibold">Loading {activeChoice.name}</div><div className="mt-0.5 text-[10px] text-slate-400">{modelFile}</div></div></div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full bg-cyan-300" style={{ width: modelProgress + "%" }} /></div>
              </div>
            ) : null}
          </div>

          {error ? <div className="flex items-start gap-2 border-t border-red-300/20 bg-red-400/10 px-5 py-3 text-xs text-red-200"><XCircle size={16} /> {error}</div> : null}

          <div className="grid grid-cols-2 border-t border-white/10 sm:grid-cols-4 lg:grid-cols-8">
            {[
              ["Detections", boxes.length],
              ["Humans", humans],
              ["Vehicles", vehicles],
              ["ROI gain", "+" + roiGain],
              ["Passes", passes || "—"],
              ["Full AI", formatMs(fullInferenceMs)],
              ["ROI AI", formatMs(roiInferenceMs)],
              ["Total", formatMs(totalMs)],
            ].map(([label, value]) => (
              <div key={String(label)} className="border-r border-white/10 px-3 py-4 last:border-r-0">
                <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</div>
                <div className="mt-1 text-lg font-bold">{value}</div>
              </div>
            ))}
          </div>

          <div className="border-t border-white/10 px-5 py-4">
            <div className="mb-2 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Detected target height bands</div>
            <div className="grid grid-cols-5 gap-2 text-center text-xs">
              {Object.entries(bands).map(([label, value]) => <div key={label} className="rounded-xl border border-white/8 bg-white/5 p-2"><div className="font-bold text-cyan-200">{value}</div><div className="mt-0.5 text-[9px] text-slate-500">{label} px</div></div>)}
            </div>
          </div>

          {totalMs ? (
            <div className="flex items-center gap-2 border-t border-pink-300/20 bg-pink-400/8 px-5 py-3 text-xs text-pink-100">
              <Gauge size={16} /> Effective analysis rate: {effectiveFps.toFixed(1)} FPS. Smart mode is an accuracy experiment; compare ROI gain against added latency.
            </div>
          ) : null}
        </section>

        <aside className="flex flex-col gap-5">
          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold"><Layers3 size={18} className="text-pink-300" /> 1 · Processing mode</h2>
            <div className="grid grid-cols-2 gap-2">
              <button type="button" onClick={() => { setMode("fast"); setRoi(null); }} className={"rounded-2xl border p-4 text-left " + (mode === "fast" ? "border-cyan-300/60 bg-cyan-300/10" : "border-white/10 bg-[#03161e]")}><strong className="text-sm">Fast · 1 pass</strong><span className="mt-1 block text-[11px] leading-5 text-slate-400">Baseline only. No ROI refinement.</span></button>
              <button type="button" onClick={() => setMode("smart")} className={"rounded-2xl border p-4 text-left " + (mode === "smart" ? "border-pink-300/60 bg-pink-300/10" : "border-white/10 bg-[#03161e]")}><strong className="text-sm">Smart Hybrid</strong><span className="mt-1 block text-[11px] leading-5 text-slate-400">Adaptive ROI + tracking.</span></button>
            </div>
            <label className="mt-4 block rounded-2xl border border-white/10 bg-[#03161e] p-4">
              <span className="flex justify-between text-xs text-slate-300">ROI refinement cadence <strong className="text-pink-200">every {refineEvery} scan{refineEvery > 1 ? "s" : ""}</strong></span>
              <input type="range" min="1" max="5" step="1" value={refineEvery} onChange={(event) => setRefineEvery(Number(event.target.value))} className="mt-4 w-full accent-pink-300" />
              <span className="mt-2 block text-[10px] leading-4 text-slate-500">1 = maximum tiny-target recall. 3–5 = lower average cost for live CCTV.</span>
            </label>
          </section>

          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold"><Cpu size={18} className="text-cyan-300" /> 2 · Model</h2>
            <div className="grid gap-2">
              {(Object.keys(MODEL_CHOICES) as ModelChoice[]).map((key) => {
                const item = MODEL_CHOICES[key];
                const active = modelChoice === key;
                return <button key={key} type="button" onClick={() => switchModel(key)} className={"relative rounded-2xl border p-4 text-left " + (active ? "border-cyan-300/60 bg-cyan-300/10" : "border-white/10 bg-[#03161e]")}><strong className="text-sm">{item.name}</strong><span className="mt-1 block text-[11px] text-slate-400">{item.detail}</span><span className="absolute right-3 top-3 rounded-full bg-white/10 px-2 py-1 text-[9px] uppercase">{item.badge}</span></button>;
              })}
            </div>
            <button type="button" onClick={() => void loadModel()} className="mt-3 flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm"><RefreshCw size={16} /> Reload model</button>
          </section>

          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold"><Camera size={18} className="text-cyan-300" /> 3 · Source & confidence</h2>
            <div className="grid grid-cols-2 gap-2">
              <button type="button" onClick={() => void startCamera("environment")} className="flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm"><Smartphone size={16} /> Rear camera</button>
              <button type="button" onClick={() => void startCamera("user")} className="flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm"><Camera size={16} /> Webcam</button>
            </div>
            <label className="mt-2 flex min-h-11 cursor-pointer items-center justify-center gap-2 rounded-xl bg-cyan-300 font-semibold text-[#021118]"><Upload size={16} /> Test image<input type="file" accept="image/*" className="hidden" onChange={handleImage} /></label>
            {sourceMode === "image" && imageUrl ? <button type="button" disabled={modelState !== "ready"} onClick={() => void analyse()} className="mt-2 flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm disabled:opacity-50"><Target size={16} /> Analyse image</button> : null}
            {cameraActive ? <button type="button" onClick={() => setRunning((value) => !value)} className="mt-2 flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm">{running ? <Pause size={16} /> : <Play size={16} />}{running ? "Pause" : "Run"}</button> : null}
            <label className="mt-4 block rounded-2xl border border-white/10 bg-[#03161e] p-4"><span className="flex justify-between text-xs text-slate-300">Confidence <strong className="text-cyan-200">{Math.round(threshold * 100)}%</strong></span><input type="range" min="5" max="60" step="1" value={Math.round(threshold * 100)} onChange={(event) => setThreshold(Number(event.target.value) / 100)} className="mt-4 w-full accent-cyan-300" /></label>
          </section>

          <section className="rounded-[26px] border border-pink-300/15 bg-pink-300/5 p-5 text-[11px] leading-5 text-slate-300">
            <div className="mb-2 flex items-center gap-2 font-semibold text-pink-100"><CheckCircle2 size={16} /> What this combines</div>
            Full-frame detector = fast baseline. Small/weak detections vote for a focus cluster. That cluster is cropped from the original pixels and analysed again at the model input size. Duplicates are merged, then IoU-based track persistence keeps recent objects alive for a few live scans. This is a browser approximation of the AutoFocus/ClusDet/ByteTrack strategy, not a reproduction of those research implementations.
          </section>
        </aside>
      </div>
    </main>
  );
}
