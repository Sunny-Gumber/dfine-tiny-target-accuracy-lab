"use client";

import {
  CheckCircle2,
  Cpu,
  ExternalLink,
  Gauge,
  Layers3,
  MonitorUp,
  Pause,
  Play,
  RefreshCw,
  Target,
  Upload,
  Video,
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

type SourceInput = HTMLVideoElement | HTMLCanvasElement;
type Mode = "fast" | "smart";
type ModelChoice = "yolox-n" | "yolo9-t" | "rfdetr-n";
type ModelFamily = "yolox" | "yolo9" | "rfdetr";

type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  cls: number;
  name: string;
  origin: "full" | "roi";
};

type Region = {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
};

type LibreDetection = {
  classId: number;
  confidence: number;
  bbox: [number, number, number, number];
  label?: string;
};

type LibreResult = {
  detections: LibreDetection[];
  numDetections: number;
};

type LibreModel = {
  predict: (
    input: SourceInput,
    options?: { confThres?: number; iouThres?: number; maxDet?: number },
  ) => Promise<LibreResult>;
  release: () => Promise<void>;
  provider: "webgpu" | "wasm" | null;
  inputSize: number;
  modelFamily: ModelFamily | null;
};

type LibreRuntime = {
  loadModel: (
    source: string,
    options?: {
      confThres?: number;
      iouThres?: number;
      maxDet?: number;
      device?: "auto" | "webgpu" | "wasm" | ("webgpu" | "wasm")[];
      inputSize?: number;
      modelFamily?: ModelFamily | "auto";
      onProgress?: (progress: number) => void;
    },
  ) => Promise<LibreModel>;
};

type ModelSpec = {
  name: string;
  source: string;
  family: ModelFamily;
  inputSize: number;
  detail: string;
  badge: string;
};

const LIBRE_RUNTIME_URL =
  "https://esm.sh/libreyolo-web@0.0.6?bundle&deps=onnxruntime-web@1.24.3";

const MODEL_CHOICES: Record<ModelChoice, ModelSpec> = {
  "yolox-n": {
    name: "YOLOX Nano",
    source: "LibreYOLOXn",
    family: "yolox",
    inputSize: 416,
    detail: "3.6 MB · current mobile speed winner",
    badge: "Recommended",
  },
  "yolo9-t": {
    name: "YOLO9 Tiny",
    source: "LibreYOLO9t",
    family: "yolo9",
    inputSize: 640,
    detail: "8 MB · stronger human recall in our traffic test",
    badge: "Human recall",
  },
  "rfdetr-n": {
    name: "RF-DETR Nano",
    source: "LibreRFDETRn",
    family: "rfdetr",
    inputSize: 384,
    detail: "103 MB · accuracy reference, not expected to be real-time on phone",
    badge: "Accuracy ref",
  },
};

const TARGET_IDS = new Set([0, 1, 2, 3, 5, 7]);
const VEHICLE_IDS = new Set([1, 2, 3, 5, 7]);
const COCO_NAMES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
  "toothbrush",
];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function isHuman(box: Detection) {
  return box.cls === 0;
}

function isVehicle(box: Detection) {
  return VEHICLE_IDS.has(box.cls);
}

function kind(box: Detection) {
  return isHuman(box) ? "human" : "vehicle";
}

function area(box: Detection) {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

function intersection(a: Detection, b: Detection) {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  return w * h;
}

function iou(a: Detection, b: Detection) {
  const inter = intersection(a, b);
  if (!inter) return 0;
  const union = area(a) + area(b) - inter;
  return union ? inter / union : 0;
}

function duplicate(a: Detection, b: Detection) {
  if (kind(a) !== kind(b)) return false;
  const inter = intersection(a, b);
  if (!inter) return false;
  const containment = inter / Math.max(1, Math.min(area(a), area(b)));
  return iou(a, b) >= 0.45 || containment >= 0.82;
}

function mergeDetections(input: Detection[]) {
  const output: Detection[] = [];
  for (const candidate of [...input].sort((a, b) => b.conf - a.conf)) {
    if (!output.some((kept) => duplicate(kept, candidate))) output.push(candidate);
  }
  return output;
}

function convertDetections(
  raw: LibreDetection[],
  threshold: number,
  origin: Detection["origin"],
  offsetX = 0,
  offsetY = 0,
) {
  return raw
    .filter((box) => TARGET_IDS.has(box.classId) && box.confidence >= threshold)
    .map((box): Detection => ({
      x1: box.bbox[0] + offsetX,
      y1: box.bbox[1] + offsetY,
      x2: box.bbox[2] + offsetX,
      y2: box.bbox[3] + offsetY,
      conf: box.confidence,
      cls: box.classId,
      name: box.label || COCO_NAMES[box.classId] || `class ${box.classId}`,
      origin,
    }));
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
    return bh < height * 0.14 || bw < width * 0.11 || box.conf < Math.max(0.38, threshold + 0.12);
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
  for (const box of candidates) {
    const cx = (box.x1 + box.x2) / 2;
    const cy = (box.y1 + box.y2) / 2;
    const col = clamp(Math.floor((cx / width) * cols), 0, cols - 1);
    const row = clamp(Math.floor((cy / height) * rows), 0, rows - 1);
    const bh = Math.max(1, box.y2 - box.y1);
    const tinyWeight = clamp((height * 0.14) / bh, 1, 3);
    const farWeight = 1 + Math.max(0, 0.68 - cy / height) * 0.9;
    const weakWeight = box.conf < 0.4 ? 1.35 : 1;
    scores[row * cols + col] += tinyWeight * farWeight * weakWeight;
  }

  let best = 0;
  for (let i = 1; i < scores.length; i += 1) if (scores[i] > scores[best]) best = i;
  const bestCol = best % cols;
  const bestRow = Math.floor(best / cols);
  const cluster = candidates.filter((box) => {
    const cx = (box.x1 + box.x2) / 2;
    const cy = (box.y1 + box.y2) / 2;
    const col = clamp(Math.floor((cx / width) * cols), 0, cols - 1);
    const row = clamp(Math.floor((cy / height) * rows), 0, rows - 1);
    return Math.abs(col - bestCol) <= 1 && Math.abs(row - bestRow) <= 1;
  });

  const x1 = Math.min(...cluster.map((box) => box.x1));
  const y1 = Math.min(...cluster.map((box) => box.y1));
  const x2 = Math.max(...cluster.map((box) => box.x2));
  const y2 = Math.max(...cluster.map((box) => box.y2));
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;
  const roiWidth = Math.min(width * 0.74, Math.max(width * 0.38, (x2 - x1) * 2.4));
  const roiHeight = Math.min(height * 0.68, Math.max(height * 0.38, (y2 - y1) * 2.6));
  const x = clamp(centerX - roiWidth / 2, 0, width - roiWidth);
  const y = clamp(centerY - roiHeight / 2, 0, height - roiHeight);
  return { x, y, width: roiWidth, height: roiHeight, label: "Adaptive tiny-object cluster" };
}

function cropVideo(source: HTMLVideoElement, region: Region) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(region.width));
  canvas.height = Math.max(1, Math.round(region.height));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Could not create ROI canvas.");
  ctx.drawImage(
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

function average(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatMs(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "—";
  return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;
}

function validYouTubeUrl(value: string) {
  try {
    const url = new URL(value);
    return ["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"].includes(url.hostname);
  } catch {
    return false;
  }
}

export default function VideoPage() {
  const [modelChoice, setModelChoice] = useState<ModelChoice>("yolox-n");
  const [mode, setMode] = useState<Mode>("fast");
  const [threshold, setThreshold] = useState(0.2);
  const [refineEvery, setRefineEvery] = useState(4);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [modelState, setModelState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [provider, setProvider] = useState("waiting");
  const [sourceLabel, setSourceLabel] = useState("No video source");
  const [sourceReady, setSourceReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [warmupDone, setWarmupDone] = useState(false);
  const [boxes, setBoxes] = useState<Detection[]>([]);
  const [roi, setRoi] = useState<Region | null>(null);
  const [fullMs, setFullMs] = useState(0);
  const [roiMs, setRoiMs] = useState(0);
  const [totalMs, setTotalMs] = useState(0);
  const [roiGain, setRoiGain] = useState(0);
  const [frames, setFrames] = useState(0);
  const [avgTotalMs, setAvgTotalMs] = useState(0);
  const [avgFullMs, setAvgFullMs] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [actualHumans, setActualHumans] = useState("");
  const [actualVehicles, setActualVehicles] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<LibreModel | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef("");
  const loadIdRef = useRef(0);
  const busyRef = useRef(false);
  const loopRef = useRef<number | null>(null);
  const scanCounterRef = useRef(0);
  const totalHistoryRef = useRef<number[]>([]);
  const fullHistoryRef = useRef<number[]>([]);

  const spec = MODEL_CHOICES[modelChoice];
  const humans = useMemo(() => boxes.filter(isHuman).length, [boxes]);
  const vehicles = useMemo(() => boxes.filter(isVehicle).length, [boxes]);
  const effectiveFps = avgTotalMs > 0 ? 1000 / avgTotalMs : 0;
  const captureSupported =
    typeof navigator !== "undefined" && !!navigator.mediaDevices?.getDisplayMedia;

  const countAgreement = useMemo(() => {
    if (actualHumans === "" || actualVehicles === "") return null;
    const actualTotal = Number(actualHumans) + Number(actualVehicles);
    const predictedTotal = humans + vehicles;
    if (actualTotal === 0) return predictedTotal === 0 ? 100 : 0;
    return clamp((1 - Math.abs(predictedTotal - actualTotal) / actualTotal) * 100, 0, 100);
  }, [actualHumans, actualVehicles, humans, vehicles]);

  const stopCurrentSource = useCallback(() => {
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = "";
    }
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
      video.removeAttribute("src");
      video.load();
    }
  }, []);

  const resetMetrics = useCallback(() => {
    scanCounterRef.current = 0;
    totalHistoryRef.current = [];
    fullHistoryRef.current = [];
    setFrames(0);
    setFullMs(0);
    setRoiMs(0);
    setTotalMs(0);
    setAvgFullMs(0);
    setAvgTotalMs(0);
    setRoiGain(0);
    setBoxes([]);
    setRoi(null);
  }, []);

  useEffect(() => {
    return () => {
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      void modelRef.current?.release();
    };
  }, []);

  useEffect(() => {
    const id = ++loadIdRef.current;
    setRunning(false);
    setWarmupDone(false);
    setError("");
    setStatus(`Loading ${spec.name}...`);
    setModelState("loading");
    setProgress(0);
    resetMetrics();

    void (async () => {
      try {
        await modelRef.current?.release();
        modelRef.current = null;
        const runtime = (await import(/* @vite-ignore */ LIBRE_RUNTIME_URL)) as unknown as LibreRuntime;
        const model = await runtime.loadModel(spec.source, {
          confThres: 0.05,
          iouThres: 0.45,
          maxDet: 300,
          device: "auto",
          inputSize: spec.inputSize,
          modelFamily: spec.family,
          onProgress: (value) => {
            if (loadIdRef.current === id) setProgress(value);
          },
        });
        if (loadIdRef.current !== id) {
          await model.release();
          return;
        }
        modelRef.current = model;
        setProvider(model.provider || "ready");
        setModelState("ready");
        setStatus(`${spec.name} ready · ${model.provider || "backend ready"}`);
      } catch (err) {
        if (loadIdRef.current !== id) return;
        setModelState("error");
        setError(err instanceof Error ? err.message : String(err));
        setStatus("Model failed to load");
      }
    })();
  }, [modelChoice, resetMetrics, spec.family, spec.inputSize, spec.name, spec.source]);

  const openYouTube = useCallback(() => {
    setError("");
    if (!validYouTubeUrl(youtubeUrl)) {
      setError("Paste a valid youtube.com or youtu.be URL first.");
      return;
    }
    window.open(youtubeUrl, "_blank", "noopener,noreferrer");
    setStatus("YouTube opened. Return here and choose Capture YouTube / screen.");
  }, [youtubeUrl]);

  const startDisplayCapture = useCallback(async () => {
    setError("");
    setRunning(false);
    resetMetrics();
    if (!navigator.mediaDevices?.getDisplayMedia) {
      setError("Screen/tab capture is not supported by this browser. Use Upload video instead.");
      return;
    }
    try {
      stopCurrentSource();
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      mediaStreamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Video viewport is not ready.");
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play();
      const track = stream.getVideoTracks()[0];
      track.addEventListener("ended", () => {
        setRunning(false);
        setSourceReady(false);
        setSourceLabel("Capture ended");
        setStatus("Screen/tab sharing stopped.");
      });
      setSourceReady(true);
      setSourceLabel("Captured tab / screen");
      setStatus("Capture ready. If using YouTube, select the YouTube tab in the browser share picker.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("Capture cancelled or unavailable.");
    }
  }, [resetMetrics, stopCurrentSource]);

  const uploadVideo = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError("");
    setRunning(false);
    resetMetrics();
    stopCurrentSource();
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    const video = videoRef.current;
    if (!video) return;
    video.src = url;
    video.srcObject = null;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.onloadedmetadata = () => {
      setSourceReady(true);
      setSourceLabel(file.name);
      setStatus("Video file ready. Press Start analysis.");
      void video.play();
    };
    video.load();
    event.target.value = "";
  }, [resetMetrics, stopCurrentSource]);

  const analyseFrame = useCallback(async () => {
    if (busyRef.current || !running) return;
    const model = modelRef.current;
    const video = videoRef.current;
    if (!model || !video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;

    busyRef.current = true;
    try {
      if (!warmupDone) {
        setStatus(`Warming up ${spec.name}; this run is excluded from timing...`);
        await model.predict(video, { confThres: threshold, iouThres: 0.45, maxDet: 300 });
        setWarmupDone(true);
        setStatus(`${spec.name} live analysis running · warm-up excluded`);
        return;
      }

      const totalStart = performance.now();
      const fullStart = performance.now();
      const fullRaw = await model.predict(video, {
        confThres: threshold,
        iouThres: 0.45,
        maxDet: 300,
      });
      const currentFullMs = performance.now() - fullStart;
      const fullBoxes = convertDetections(fullRaw.detections, threshold, "full");

      scanCounterRef.current += 1;
      let currentRoiMs = 0;
      let activeRoi: Region | null = null;
      let finalBoxes = fullBoxes;
      let gain = 0;

      const shouldRefine = mode === "smart" && scanCounterRef.current % refineEvery === 0;
      if (shouldRefine) {
        activeRoi = chooseSmartRegion(fullBoxes, video.videoWidth, video.videoHeight, threshold);
        const crop = cropVideo(video, activeRoi);
        const roiStart = performance.now();
        const roiRaw = await model.predict(crop, {
          confThres: threshold,
          iouThres: 0.45,
          maxDet: 300,
        });
        currentRoiMs = performance.now() - roiStart;
        const roiBoxes = convertDetections(
          roiRaw.detections,
          threshold,
          "roi",
          activeRoi.x,
          activeRoi.y,
        );
        finalBoxes = mergeDetections([...fullBoxes, ...roiBoxes]);
        gain = Math.max(0, finalBoxes.length - fullBoxes.length);
      }

      const currentTotalMs = performance.now() - totalStart;
      totalHistoryRef.current.push(currentTotalMs);
      fullHistoryRef.current.push(currentFullMs);
      if (totalHistoryRef.current.length > 30) totalHistoryRef.current.shift();
      if (fullHistoryRef.current.length > 30) fullHistoryRef.current.shift();

      setBoxes(finalBoxes);
      setRoi(activeRoi);
      setFullMs(currentFullMs);
      setRoiMs(currentRoiMs);
      setTotalMs(currentTotalMs);
      setRoiGain(gain);
      setFrames((value) => value + 1);
      setAvgTotalMs(average(totalHistoryRef.current));
      setAvgFullMs(average(fullHistoryRef.current));
    } catch (err) {
      setRunning(false);
      setError(err instanceof Error ? err.message : String(err));
      setStatus("Live inference stopped because of an error.");
    } finally {
      busyRef.current = false;
    }
  }, [mode, refineEvery, running, spec.name, threshold, warmupDone]);

  useEffect(() => {
    if (!running) {
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      loopRef.current = null;
      return;
    }
    const loop = () => {
      void analyseFrame();
      loopRef.current = requestAnimationFrame(loop);
    };
    loopRef.current = requestAnimationFrame(loop);
    return () => {
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      loopRef.current = null;
    };
  }, [analyseFrame, running]);

  useEffect(() => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !video.videoWidth || !video.videoHeight) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (roi) {
      ctx.save();
      ctx.setLineDash([12, 8]);
      ctx.lineWidth = Math.max(2, canvas.width / 700);
      ctx.strokeStyle = "#f472b6";
      ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
      ctx.restore();
    }

    const fontSize = Math.max(12, Math.round(canvas.width / 90));
    ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
    for (const box of boxes) {
      const human = isHuman(box);
      const color = box.origin === "roi" ? "#f472b6" : human ? "#67e8f9" : "#facc15";
      ctx.lineWidth = Math.max(2, canvas.width / 800);
      ctx.strokeStyle = color;
      ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      const label = `${human ? "Human" : box.name} ${Math.round(box.conf * 100)}%`;
      const width = ctx.measureText(label).width + 8;
      const y = Math.max(fontSize + 4, box.y1);
      ctx.fillStyle = color;
      ctx.fillRect(box.x1, y - fontSize - 4, width, fontSize + 6);
      ctx.fillStyle = "#021018";
      ctx.fillText(label, box.x1 + 4, y);
    }
  }, [boxes, roi]);

  const startAnalysis = useCallback(() => {
    setError("");
    if (modelState !== "ready" || !modelRef.current) {
      setError("Wait for the model to finish loading.");
      return;
    }
    if (!sourceReady || !videoRef.current) {
      setError("Choose a captured tab/screen or upload a video first.");
      return;
    }
    resetMetrics();
    setWarmupDone(false);
    setRunning(true);
    void videoRef.current.play();
  }, [modelState, resetMetrics, sourceReady]);

  const pauseVideoForScore = useCallback(() => {
    setRunning(false);
    videoRef.current?.pause();
    setStatus("Frame paused. Enter visible human/vehicle counts below for a quick count-agreement check.");
  }, []);

  return (
    <main className="min-h-screen bg-[#021018] text-[#e8f7fa]">
      <div className="mx-auto max-w-5xl px-4 py-7 sm:px-6">
        <div className="mb-7">
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.32em] text-cyan-300">
            <Video className="h-4 w-4" /> Real-Time CCTV Video Lab
          </div>
          <h1 className="text-3xl font-bold sm:text-4xl">YouTube / Video Human + Vehicle Test</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
            Analyse a user-selected YouTube tab, screen, or local video with the same small-target models. YouTube itself is not read from an iframe; the browser capture stream supplies the frames.
          </p>
        </div>

        <section className="overflow-hidden rounded-[28px] border border-cyan-900/50 bg-[#071e27]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-cyan-900/50 px-5 py-4">
            <div>
              <div className="font-semibold">Live analysis viewport</div>
              <div className="text-xs text-slate-400">{sourceLabel}</div>
            </div>
            <div className="rounded-full bg-slate-900/70 px-3 py-1.5 text-xs text-cyan-200">
              {spec.name} · {mode === "fast" ? "Fast" : `Smart / ${refineEvery}`}
            </div>
          </div>

          <div className="relative bg-black">
            <video ref={videoRef} className="block max-h-[72vh] w-full object-contain" playsInline muted controls />
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
          </div>

          <div className="grid grid-cols-2 border-t border-cyan-900/50 sm:grid-cols-4">
            {[
              ["Detections", boxes.length],
              ["Humans", humans],
              ["Vehicles", vehicles],
              ["ROI gain", `+${roiGain}`],
              ["Full pass", formatMs(fullMs)],
              ["ROI pass", roiMs ? formatMs(roiMs) : "—"],
              ["Frame total", formatMs(totalMs)],
              ["Rolling FPS", effectiveFps ? effectiveFps.toFixed(1) : "—"],
            ].map(([label, value]) => (
              <div key={String(label)} className="border-b border-r border-cyan-900/40 p-4 sm:p-5">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">{label}</div>
                <div className="mt-2 text-2xl font-bold">{value}</div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-x-6 gap-y-2 bg-slate-950/30 px-5 py-3 text-xs text-slate-400">
            <span>Frames analysed: <b className="text-slate-200">{frames}</b></span>
            <span>Avg full: <b className="text-slate-200">{formatMs(avgFullMs)}</b></span>
            <span>Backend: <b className="text-cyan-200">{provider}</b></span>
            <span>Warm-up: <b className="text-slate-200">{warmupDone ? "excluded" : "pending"}</b></span>
          </div>
        </section>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <section className="rounded-[28px] border border-cyan-900/50 bg-[#071e27] p-5">
            <div className="mb-4 flex items-center gap-2 font-semibold"><MonitorUp className="h-5 w-5 text-cyan-300" /> 1 · YouTube / source</div>
            <div className="space-y-3">
              <div className="flex gap-2">
                <input
                  value={youtubeUrl}
                  onChange={(event) => setYoutubeUrl(event.target.value)}
                  placeholder="Paste YouTube URL"
                  className="min-w-0 flex-1 rounded-xl border border-cyan-900/60 bg-slate-950/50 px-3 py-3 text-sm outline-none focus:border-cyan-500"
                />
                <button onClick={openYouTube} className="rounded-xl border border-cyan-700/60 px-3 py-2 text-sm hover:bg-cyan-950/40">
                  <ExternalLink className="h-4 w-4" />
                </button>
              </div>
              <button
                onClick={startDisplayCapture}
                disabled={!captureSupported}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <MonitorUp className="h-5 w-5" /> Capture YouTube / screen
              </button>
              <label className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl border border-cyan-800/70 px-4 py-3 font-semibold hover:bg-cyan-950/30">
                <Upload className="h-5 w-5" /> Upload video file
                <input type="file" accept="video/*" className="hidden" onChange={uploadVideo} />
              </label>
              <p className="text-xs leading-5 text-slate-500">
                On desktop, open the YouTube video in another tab, return here, press Capture, then select that YouTube tab. Screen capture support varies by browser/mobile OS; upload video is the fallback.
              </p>
            </div>
          </section>

          <section className="rounded-[28px] border border-cyan-900/50 bg-[#071e27] p-5">
            <div className="mb-4 flex items-center gap-2 font-semibold"><Cpu className="h-5 w-5 text-cyan-300" /> 2 · Model</div>
            <div className="space-y-3">
              {(Object.entries(MODEL_CHOICES) as [ModelChoice, ModelSpec][]).map(([key, item]) => (
                <button
                  key={key}
                  onClick={() => setModelChoice(key)}
                  className={`w-full rounded-2xl border p-4 text-left ${modelChoice === key ? "border-cyan-400 bg-cyan-950/50" : "border-cyan-950 bg-slate-950/20"}`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-semibold">{item.name}</span>
                    <span className="rounded-full bg-slate-800 px-2 py-1 text-[10px] uppercase text-slate-300">{item.badge}</span>
                  </div>
                  <div className="mt-1 text-xs text-slate-400">{item.detail}</div>
                </button>
              ))}
              <div className="rounded-xl border border-cyan-950 bg-slate-950/30 px-3 py-3 text-xs">
                {modelState === "loading" ? `Loading ${Math.round(progress * 100)}%` : status || "Choose a model"}
              </div>
            </div>
          </section>

          <section className="rounded-[28px] border border-cyan-900/50 bg-[#071e27] p-5">
            <div className="mb-4 flex items-center gap-2 font-semibold"><Layers3 className="h-5 w-5 text-cyan-300" /> 3 · Processing</div>
            <div className="grid grid-cols-2 gap-3">
              <button onClick={() => setMode("fast")} className={`rounded-2xl border p-4 text-left ${mode === "fast" ? "border-cyan-400 bg-cyan-950/50" : "border-cyan-950"}`}>
                <div className="font-semibold">Fast</div><div className="mt-1 text-xs text-slate-400">1 pass every analysed frame</div>
              </button>
              <button onClick={() => setMode("smart")} className={`rounded-2xl border p-4 text-left ${mode === "smart" ? "border-pink-400 bg-pink-950/20" : "border-cyan-950"}`}>
                <div className="font-semibold">Smart ROI</div><div className="mt-1 text-xs text-slate-400">Periodic second pass</div>
              </button>
            </div>
            <div className="mt-4">
              <div className="mb-2 flex justify-between text-sm"><span>ROI cadence</span><b>every {refineEvery} scans</b></div>
              <input type="range" min={1} max={8} value={refineEvery} onChange={(e) => setRefineEvery(Number(e.target.value))} className="w-full" />
            </div>
            <div className="mt-4">
              <div className="mb-2 flex justify-between text-sm"><span>Confidence</span><b>{Math.round(threshold * 100)}%</b></div>
              <input type="range" min={5} max={70} value={Math.round(threshold * 100)} onChange={(e) => setThreshold(Number(e.target.value) / 100)} className="w-full" />
            </div>
          </section>

          <section className="rounded-[28px] border border-cyan-900/50 bg-[#071e27] p-5">
            <div className="mb-4 flex items-center gap-2 font-semibold"><Gauge className="h-5 w-5 text-cyan-300" /> 4 · Run / score</div>
            <div className="grid grid-cols-2 gap-3">
              <button onClick={startAnalysis} className="flex items-center justify-center gap-2 rounded-xl bg-cyan-400 px-4 py-3 font-semibold text-slate-950">
                <Play className="h-4 w-4" /> Start analysis
              </button>
              <button onClick={() => setRunning(false)} className="flex items-center justify-center gap-2 rounded-xl border border-cyan-800 px-4 py-3 font-semibold">
                <Pause className="h-4 w-4" /> Stop AI
              </button>
              <button onClick={pauseVideoForScore} className="col-span-2 flex items-center justify-center gap-2 rounded-xl border border-pink-800/70 px-4 py-3 font-semibold text-pink-200">
                <Target className="h-4 w-4" /> Pause frame for count check
              </button>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <label className="text-xs text-slate-400">Actual humans
                <input type="number" min={0} value={actualHumans} onChange={(e) => setActualHumans(e.target.value)} className="mt-1 w-full rounded-xl border border-cyan-950 bg-slate-950/50 px-3 py-2 text-base text-slate-100" />
              </label>
              <label className="text-xs text-slate-400">Actual vehicles
                <input type="number" min={0} value={actualVehicles} onChange={(e) => setActualVehicles(e.target.value)} className="mt-1 w-full rounded-xl border border-cyan-950 bg-slate-950/50 px-3 py-2 text-base text-slate-100" />
              </label>
            </div>
            {countAgreement !== null && (
              <div className="mt-3 rounded-xl bg-slate-950/40 p-3 text-sm">
                Quick count agreement: <b className="text-cyan-300">{countAgreement.toFixed(1)}%</b>
                <div className="mt-1 text-[11px] text-slate-500">Count agreement is not mAP/precision/recall; formal accuracy needs bounding-box ground truth.</div>
              </div>
            )}
          </section>
        </div>

        {error && (
          <div className="mt-6 flex gap-3 rounded-2xl border border-red-900/60 bg-red-950/30 p-4 text-sm text-red-200">
            <XCircle className="mt-0.5 h-5 w-5 shrink-0" /> {error}
          </div>
        )}

        <section className="mt-6 rounded-[28px] border border-cyan-900/50 bg-[#071e27] p-5 text-sm leading-6 text-slate-400">
          <div className="mb-2 flex items-center gap-2 font-semibold text-slate-100"><Zap className="h-5 w-5 text-cyan-300" /> How to use with YouTube</div>
          <ol className="list-decimal space-y-1 pl-5">
            <li>Paste/open a YouTube traffic, crowd, or CCTV video in another tab.</li>
            <li>Return here and press <b className="text-slate-200">Capture YouTube / screen</b>.</li>
            <li>Select the YouTube tab/window in the browser share picker.</li>
            <li>Start with <b className="text-slate-200">YOLOX Nano + Fast + 20%</b>, then compare Smart ROI every 4 scans.</li>
            <li>For a measurable checkpoint, pause one frame and enter the visible human/vehicle counts.</li>
          </ol>
        </section>

        <div className="mt-6 flex flex-wrap gap-3 text-xs text-slate-500">
          <span className="inline-flex items-center gap-1"><CheckCircle2 className="h-4 w-4 text-emerald-400" /> Warm-up excluded from rolling timing</span>
          <span className="inline-flex items-center gap-1"><RefreshCw className="h-4 w-4 text-cyan-400" /> Rolling average = last 30 analysed frames</span>
        </div>
      </div>
    </main>
  );
}
