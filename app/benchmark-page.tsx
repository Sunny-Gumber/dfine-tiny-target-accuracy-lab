"use client";

import {
  BarChart3,
  CheckCircle2,
  Download,
  LoaderCircle,
  Play,
  Target,
  Trash2,
  Undo2,
  Upload,
} from "lucide-react";
import {
  ChangeEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type GTClass = "human" | "vehicle";
type Mode = "fast" | "smart";

type GTBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cls: GTClass;
};

type Prediction = GTBox & {
  conf: number;
  origin: "full" | "roi";
};

type SampleFrame = {
  id: number;
  time: number;
  url: string;
  width: number;
  height: number;
  gt: GTBox[];
};

type Region = {
  x: number;
  y: number;
  width: number;
  height: number;
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
    input: HTMLImageElement | HTMLCanvasElement,
    options?: { confThres?: number; iouThres?: number; maxDet?: number },
  ) => Promise<LibreResult>;
  release: () => Promise<void>;
  provider: "webgpu" | "wasm" | null;
  inputSize: number;
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
      modelFamily?: "yolox" | "auto";
      onProgress?: (progress: number) => void;
    },
  ) => Promise<LibreModel>;
};

type FrameMetric = {
  frameId: number;
  time: number;
  tp: number;
  fp: number;
  fn: number;
  humanTp: number;
  humanFp: number;
  humanFn: number;
  vehicleTp: number;
  vehicleFp: number;
  vehicleFn: number;
  fullMs: number;
  roiMs: number;
  totalMs: number;
  roiGain: number;
  matchedGt: Set<number>;
};

type BenchmarkResult = {
  mode: Mode;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
  humanPrecision: number;
  humanRecall: number;
  humanF1: number;
  vehiclePrecision: number;
  vehicleRecall: number;
  vehicleF1: number;
  avgFullMs: number;
  avgRoiMs: number;
  avgTotalMs: number;
  roiGain: number;
  frames: FrameMetric[];
  bands: Record<string, { total: number; hit: number }>;
};

const LIBRE_RUNTIME_URL =
  "https://esm.sh/libreyolo-web@0.0.6?bundle&deps=onnxruntime-web@1.24.3";
const MODEL_SOURCE = "LibreYOLOXn";
const MODEL_INPUT = 416;
const TARGET_IDS = new Set([0, 1, 2, 3, 5, 7]);
const VEHICLE_IDS = new Set([1, 2, 3, 5, 7]);
const STORAGE_KEY = "cctv-ground-truth-benchmark-v1";

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function area(box: GTBox) {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

function intersection(a: GTBox, b: GTBox) {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  return w * h;
}

function iou(a: GTBox, b: GTBox) {
  const inter = intersection(a, b);
  if (!inter) return 0;
  const union = area(a) + area(b) - inter;
  return union ? inter / union : 0;
}

function duplicate(a: Prediction, b: Prediction) {
  if (a.cls !== b.cls) return false;
  const inter = intersection(a, b);
  if (!inter) return false;
  const containment = inter / Math.max(1, Math.min(area(a), area(b)));
  return iou(a, b) >= 0.45 || containment >= 0.82;
}

function mergePredictions(input: Prediction[]) {
  const out: Prediction[] = [];
  for (const candidate of [...input].sort((a, b) => b.conf - a.conf)) {
    if (!out.some((kept) => duplicate(kept, candidate))) out.push(candidate);
  }
  return out;
}

function convertPredictions(
  raw: LibreDetection[],
  threshold: number,
  origin: Prediction["origin"],
  offsetX = 0,
  offsetY = 0,
) {
  return raw
    .filter((box) => TARGET_IDS.has(box.classId) && box.confidence >= threshold)
    .map((box): Prediction => ({
      x1: box.bbox[0] + offsetX,
      y1: box.bbox[1] + offsetY,
      x2: box.bbox[2] + offsetX,
      y2: box.bbox[3] + offsetY,
      conf: box.confidence,
      cls: box.classId === 0 ? "human" : "vehicle",
      origin,
    }));
}

function chooseSmartRegion(
  detections: Prediction[],
  width: number,
  height: number,
  threshold: number,
): Region {
  const candidates = detections.filter((box) => {
    const bw = box.x2 - box.x1;
    const bh = box.y2 - box.y1;
    return bh < height * 0.12 || bw < width * 0.1 || box.conf < Math.max(0.38, threshold + 0.12);
  });

  if (!candidates.length) {
    return { x: width * 0.14, y: height * 0.18, width: width * 0.72, height: height * 0.52 };
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
    const tinyWeight = clamp((height * 0.12) / bh, 1, 3);
    const farWeight = 1 + Math.max(0, 0.7 - cy / height) * 0.8;
    const weakWeight = box.conf < 0.4 ? 1.35 : 1;
    scores[row * cols + col] += tinyWeight * farWeight * weakWeight;
  }

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

  const x1 = Math.min(...cluster.map((box) => box.x1));
  const y1 = Math.min(...cluster.map((box) => box.y1));
  const x2 = Math.max(...cluster.map((box) => box.x2));
  const y2 = Math.max(...cluster.map((box) => box.y2));
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;
  const roiWidth = Math.min(width * 0.74, Math.max(width * 0.38, (x2 - x1) * 2.4));
  const roiHeight = Math.min(height * 0.68, Math.max(height * 0.38, (y2 - y1) * 2.6));
  return {
    x: clamp(centerX - roiWidth / 2, 0, width - roiWidth),
    y: clamp(centerY - roiHeight / 2, 0, height - roiHeight),
    width: roiWidth,
    height: roiHeight,
  };
}

function cropRegion(source: HTMLImageElement, region: Region) {
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

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function safeDiv(a: number, b: number) {
  return b ? a / b : 0;
}

function f1(precision: number, recall: number) {
  return precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
}

function bandFor(px: number) {
  if (px < 16) return "<16 px";
  if (px < 32) return "16–32 px";
  if (px < 64) return "32–64 px";
  return ">=64 px";
}

function evaluateFrame(gt: GTBox[], predictions: Prediction[], threshold: number) {
  const matchedGt = new Set<number>();
  let tp = 0;
  let fp = 0;
  let humanTp = 0;
  let humanFp = 0;
  let vehicleTp = 0;
  let vehicleFp = 0;

  const ordered = [...predictions].sort((a, b) => b.conf - a.conf);
  for (const pred of ordered) {
    let bestIndex = -1;
    let bestIou = 0;
    gt.forEach((truth, index) => {
      if (matchedGt.has(index) || truth.cls !== pred.cls) return;
      const score = iou(truth, pred);
      if (score > bestIou) {
        bestIou = score;
        bestIndex = index;
      }
    });
    if (bestIndex >= 0 && bestIou >= threshold) {
      matchedGt.add(bestIndex);
      tp += 1;
      if (pred.cls === "human") humanTp += 1;
      else vehicleTp += 1;
    } else {
      fp += 1;
      if (pred.cls === "human") humanFp += 1;
      else vehicleFp += 1;
    }
  }

  const humanGt = gt.filter((box) => box.cls === "human").length;
  const vehicleGt = gt.filter((box) => box.cls === "vehicle").length;
  return {
    tp,
    fp,
    fn: gt.length - matchedGt.size,
    humanTp,
    humanFp,
    humanFn: humanGt - humanTp,
    vehicleTp,
    vehicleFp,
    vehicleFn: vehicleGt - vehicleTp,
    matchedGt,
  };
}

function loadImage(url: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load sampled frame."));
    image.src = url;
  });
}

function waitFor(video: HTMLVideoElement, event: "loadedmetadata" | "seeked") {
  return new Promise<void>((resolve, reject) => {
    const done = () => {
      cleanup();
      resolve();
    };
    const fail = () => {
      cleanup();
      reject(new Error("Video could not be read."));
    };
    const cleanup = () => {
      video.removeEventListener(event, done);
      video.removeEventListener("error", fail);
    };
    video.addEventListener(event, done, { once: true });
    video.addEventListener("error", fail, { once: true });
  });
}

export default function BenchmarkPage() {
  const [frames, setFrames] = useState<SampleFrame[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [activeClass, setActiveClass] = useState<GTClass>("human");
  const [draft, setDraft] = useState<GTBox | null>(null);
  const [videoName, setVideoName] = useState("");
  const [videoDuration, setVideoDuration] = useState(0);
  const [sampleCount, setSampleCount] = useState(3);
  const [confidence, setConfidence] = useState(0.2);
  const [matchIou, setMatchIou] = useState(0.5);
  const [modelState, setModelState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [provider, setProvider] = useState("waiting");
  const [running, setRunning] = useState(false);
  const [runProgress, setRunProgress] = useState(0);
  const [fastResult, setFastResult] = useState<BenchmarkResult | null>(null);
  const [smartResult, setSmartResult] = useState<BenchmarkResult | null>(null);
  const [error, setError] = useState("");

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const objectUrlRef = useRef("");
  const modelRef = useRef<LibreModel | null>(null);
  const pointerIdRef = useRef<number | null>(null);

  const current = frames[currentIndex] || null;
  const annotatedFrames = useMemo(() => frames.filter((frame) => frame.gt.length > 0).length, [frames]);
  const totalGt = useMemo(() => frames.reduce((sum, frame) => sum + frame.gt.length, 0), [frames]);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      modelRef.current?.release();
    };
  }, []);

  useEffect(() => {
    if (!videoName || !frames.length) return;
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        videoName,
        duration: videoDuration,
        frames: frames.map((frame) => ({ time: frame.time, gt: frame.gt })),
      }),
    );
  }, [frames, videoDuration, videoName]);

  const drawCanvas = useCallback(async () => {
    if (!current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const image = await loadImage(current.url);
    canvas.width = current.width;
    canvas.height = current.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(image, 0, 0, current.width, current.height);

    const drawBox = (box: GTBox, dashed = false) => {
      const human = box.cls === "human";
      ctx.strokeStyle = human ? "#5ee7f7" : "#ffd166";
      ctx.fillStyle = human ? "rgba(94,231,247,.14)" : "rgba(255,209,102,.14)";
      ctx.lineWidth = Math.max(2, current.width / 500);
      ctx.setLineDash(dashed ? [12, 8] : []);
      ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      ctx.fillRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      ctx.setLineDash([]);
      const label = box.cls === "human" ? "GT Human" : "GT Vehicle";
      const fontSize = Math.max(15, current.width / 55);
      ctx.font = `600 ${fontSize}px sans-serif`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = human ? "#5ee7f7" : "#ffd166";
      ctx.fillRect(box.x1, Math.max(0, box.y1 - fontSize - 8), textWidth + 10, fontSize + 8);
      ctx.fillStyle = "#021018";
      ctx.fillText(label, box.x1 + 5, Math.max(fontSize, box.y1 - 6));
    };

    current.gt.forEach((box) => drawBox(box));
    if (draft) drawBox(draft, true);
  }, [current, draft]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const pointFromPointer = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!current) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    pointerIdRef.current = event.pointerId;
    const point = pointFromPointer(event);
    setDraft({ x1: point.x, y1: point.y, x2: point.x, y2: point.y, cls: activeClass });
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!draft || pointerIdRef.current !== event.pointerId) return;
    const point = pointFromPointer(event);
    setDraft((box) => (box ? { ...box, x2: point.x, y2: point.y } : box));
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!draft || pointerIdRef.current !== event.pointerId || !current) return;
    const x1 = Math.min(draft.x1, draft.x2);
    const y1 = Math.min(draft.y1, draft.y2);
    const x2 = Math.max(draft.x1, draft.x2);
    const y2 = Math.max(draft.y1, draft.y2);
    pointerIdRef.current = null;
    setDraft(null);
    if (x2 - x1 < 8 || y2 - y1 < 8) return;
    setFrames((items) =>
      items.map((frame, index) =>
        index === currentIndex ? { ...frame, gt: [...frame.gt, { x1, y1, x2, y2, cls: activeClass }] } : frame,
      ),
    );
  };

  const extractFrames = async (file: File) => {
    setError("");
    setFastResult(null);
    setSmartResult(null);
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    videoRef.current = video;
    const metaPromise = waitFor(video, "loadedmetadata");
    video.load();
    await metaPromise;
    const duration = video.duration;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!duration || !width || !height) throw new Error("Video metadata is incomplete.");

    const positions = Array.from({ length: sampleCount }, (_, index) => (index + 1) / (sampleCount + 1));
    const savedRaw = localStorage.getItem(STORAGE_KEY);
    let saved: { videoName?: string; duration?: number; frames?: { time: number; gt: GTBox[] }[] } | null = null;
    try {
      saved = savedRaw ? JSON.parse(savedRaw) : null;
    } catch {
      saved = null;
    }
    const canRestore = saved?.videoName === file.name && Math.abs((saved?.duration || 0) - duration) < 0.2;

    const samples: SampleFrame[] = [];
    for (let index = 0; index < positions.length; index += 1) {
      const time = duration * positions[index];
      const seek = waitFor(video, "seeked");
      video.currentTime = time;
      await seek;
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not extract video frame.");
      ctx.drawImage(video, 0, 0, width, height);
      const frameUrl = canvas.toDataURL("image/jpeg", 0.9);
      const restoredGt = canRestore && saved?.frames?.[index] ? saved.frames[index].gt || [] : [];
      samples.push({ id: index, time, url: frameUrl, width, height, gt: restoredGt });
    }
    setVideoName(file.name);
    setVideoDuration(duration);
    setFrames(samples);
    setCurrentIndex(0);
  };

  const onVideoChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      await extractFrames(file);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not sample the video.");
    } finally {
      event.target.value = "";
    }
  };

  const loadModel = async () => {
    if (modelRef.current) return modelRef.current;
    setModelState("loading");
    setProgress(0);
    try {
      const runtime = (await import(/* @vite-ignore */ LIBRE_RUNTIME_URL)) as unknown as LibreRuntime;
      const model = await runtime.loadModel(MODEL_SOURCE, {
        device: ["webgpu", "wasm"],
        inputSize: MODEL_INPUT,
        modelFamily: "yolox",
        confThres: 0.05,
        iouThres: 0.45,
        maxDet: 300,
        onProgress: setProgress,
      });
      modelRef.current = model;
      setProvider(model.provider || "unknown");
      setModelState("ready");
      return model;
    } catch (err) {
      setModelState("error");
      throw err;
    }
  };

  const runMode = async (mode: Mode, model: LibreModel): Promise<BenchmarkResult> => {
    const usableFrames = frames.filter((frame) => frame.gt.length > 0);
    const metrics: FrameMetric[] = [];
    const bands: Record<string, { total: number; hit: number }> = {
      "<16 px": { total: 0, hit: 0 },
      "16–32 px": { total: 0, hit: 0 },
      "32–64 px": { total: 0, hit: 0 },
      ">=64 px": { total: 0, hit: 0 },
    };

    for (let index = 0; index < usableFrames.length; index += 1) {
      const frame = usableFrames[index];
      setRunProgress((index + (mode === "smart" ? usableFrames.length : 0)) / (usableFrames.length * 2));
      const image = await loadImage(frame.url);
      const wallStart = performance.now();
      const fullStart = performance.now();
      const fullRaw = await model.predict(image, { confThres: Math.min(0.05, confidence), iouThres: 0.45, maxDet: 300 });
      const fullMs = performance.now() - fullStart;
      const full = convertPredictions(fullRaw.detections, confidence, "full");
      let final = full;
      let roiMs = 0;
      let roiGain = 0;

      if (mode === "smart") {
        const region = chooseSmartRegion(full, frame.width, frame.height, confidence);
        const crop = cropRegion(image, region);
        const roiStart = performance.now();
        const roiRaw = await model.predict(crop, { confThres: Math.min(0.05, confidence), iouThres: 0.45, maxDet: 300 });
        roiMs = performance.now() - roiStart;
        const roiPred = convertPredictions(roiRaw.detections, confidence, "roi", region.x, region.y);
        final = mergePredictions([...full, ...roiPred]);
        roiGain = Math.max(0, final.length - full.length);
      }

      const evaluation = evaluateFrame(frame.gt, final, matchIou);
      const totalMs = performance.now() - wallStart;
      metrics.push({ frameId: frame.id, time: frame.time, fullMs, roiMs, totalMs, roiGain, ...evaluation });

      const scale = Math.min(MODEL_INPUT / frame.width, MODEL_INPUT / frame.height);
      frame.gt.forEach((truth, gtIndex) => {
        const effectiveHeight = (truth.y2 - truth.y1) * scale;
        const band = bandFor(effectiveHeight);
        bands[band].total += 1;
        if (evaluation.matchedGt.has(gtIndex)) bands[band].hit += 1;
      });
    }

    const sum = (key: keyof FrameMetric) => metrics.reduce((acc, item) => acc + Number(item[key]), 0);
    const tp = sum("tp");
    const fp = sum("fp");
    const fn = sum("fn");
    const humanTp = sum("humanTp");
    const humanFp = sum("humanFp");
    const humanFn = sum("humanFn");
    const vehicleTp = sum("vehicleTp");
    const vehicleFp = sum("vehicleFp");
    const vehicleFn = sum("vehicleFn");
    const precision = safeDiv(tp, tp + fp);
    const recall = safeDiv(tp, tp + fn);
    const humanPrecision = safeDiv(humanTp, humanTp + humanFp);
    const humanRecall = safeDiv(humanTp, humanTp + humanFn);
    const vehiclePrecision = safeDiv(vehicleTp, vehicleTp + vehicleFp);
    const vehicleRecall = safeDiv(vehicleTp, vehicleTp + vehicleFn);
    return {
      mode,
      tp,
      fp,
      fn,
      precision,
      recall,
      f1: f1(precision, recall),
      humanPrecision,
      humanRecall,
      humanF1: f1(humanPrecision, humanRecall),
      vehiclePrecision,
      vehicleRecall,
      vehicleF1: f1(vehiclePrecision, vehicleRecall),
      avgFullMs: safeDiv(sum("fullMs"), metrics.length),
      avgRoiMs: safeDiv(sum("roiMs"), metrics.length),
      avgTotalMs: safeDiv(sum("totalMs"), metrics.length),
      roiGain: sum("roiGain"),
      frames: metrics,
      bands,
    };
  };

  const runBenchmark = async () => {
    if (!frames.length || !annotatedFrames) {
      setError("Add ground-truth boxes to at least one sampled frame first.");
      return;
    }
    setRunning(true);
    setError("");
    setRunProgress(0);
    try {
      const model = await loadModel();
      const warmFrame = frames.find((frame) => frame.gt.length > 0)!;
      const warmImage = await loadImage(warmFrame.url);
      await model.predict(warmImage, { confThres: 0.05, iouThres: 0.45, maxDet: 300 });
      const fast = await runMode("fast", model);
      setFastResult(fast);
      const smart = await runMode("smart", model);
      setSmartResult(smart);
      setRunProgress(1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed.");
    } finally {
      setRunning(false);
    }
  };

  const exportAnnotations = () => {
    if (!frames.length) return;
    const payload = {
      schema: "cctv-ground-truth-v1",
      video: { name: videoName, duration: videoDuration },
      modelInputReference: MODEL_INPUT,
      frames: frames.map((frame) => ({ time: frame.time, width: frame.width, height: frame.height, boxes: frame.gt })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${videoName || "video"}-ground-truth.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const undo = () => {
    if (!current) return;
    setFrames((items) =>
      items.map((frame, index) =>
        index === currentIndex ? { ...frame, gt: frame.gt.slice(0, -1) } : frame,
      ),
    );
  };

  const clear = () => {
    if (!current) return;
    setFrames((items) => items.map((frame, index) => (index === currentIndex ? { ...frame, gt: [] } : frame)));
  };

  const resultCard = (result: BenchmarkResult, title: string) => (
    <div className="rounded-[28px] border border-cyan-300/20 bg-[#071e27] p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm uppercase tracking-[.22em] text-cyan-300/70">{title}</div>
          <div className="mt-1 text-2xl font-semibold">YOLOX Nano · {result.mode === "fast" ? "1 pass" : "Smart 2 pass"}</div>
        </div>
        <CheckCircle2 className="h-7 w-7 text-cyan-300" />
      </div>
      <div className="mt-5 grid grid-cols-3 gap-3 text-center">
        <Metric label="Precision" value={pct(result.precision)} />
        <Metric label="Recall" value={pct(result.recall)} />
        <Metric label="F1" value={pct(result.f1)} />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3 text-center">
        <Metric label="TP" value={String(result.tp)} />
        <Metric label="FP" value={String(result.fp)} />
        <Metric label="FN" value={String(result.fn)} />
      </div>
      <div className="mt-5 grid gap-3 md:grid-cols-2">
        <div className="rounded-2xl border border-white/10 bg-black/15 p-4">
          <div className="text-sm text-slate-400">Human</div>
          <div className="mt-2 text-sm">P {pct(result.humanPrecision)} · R {pct(result.humanRecall)} · F1 {pct(result.humanF1)}</div>
        </div>
        <div className="rounded-2xl border border-white/10 bg-black/15 p-4">
          <div className="text-sm text-slate-400">Vehicle</div>
          <div className="mt-2 text-sm">P {pct(result.vehiclePrecision)} · R {pct(result.vehicleRecall)} · F1 {pct(result.vehicleF1)}</div>
        </div>
      </div>
      <div className="mt-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric label="Avg full" value={`${result.avgFullMs.toFixed(0)} ms`} />
        <Metric label="Avg ROI" value={result.mode === "smart" ? `${result.avgRoiMs.toFixed(0)} ms` : "—"} />
        <Metric label="Avg total" value={`${result.avgTotalMs.toFixed(0)} ms`} />
        <Metric label="ROI gain" value={result.mode === "smart" ? `+${result.roiGain}` : "—"} />
      </div>
      <div className="mt-5">
        <div className="mb-2 text-sm font-medium text-slate-300">Recall by effective model-input target height</div>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {Object.entries(result.bands).map(([label, value]) => (
            <div key={label} className="rounded-2xl border border-white/10 bg-black/15 p-3 text-center">
              <div className="text-xs text-slate-400">{label}</div>
              <div className="mt-1 text-lg font-semibold">{value.total ? pct(value.hit / value.total) : "—"}</div>
              <div className="text-xs text-slate-500">{value.hit}/{value.total}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <main className="min-h-screen bg-[#021018] px-4 py-8 text-slate-100 md:px-8">
      <div className="mx-auto max-w-6xl">
        <div className="text-xs font-semibold uppercase tracking-[.35em] text-cyan-300">Ground-truth CCTV benchmark</div>
        <h1 className="mt-3 max-w-4xl text-4xl font-semibold leading-tight md:text-6xl">Measure real Precision, Recall and F1 on your own video</h1>
        <p className="mt-4 max-w-3xl text-base leading-7 text-slate-400">
          Sample fixed video frames, draw Human/Vehicle ground-truth boxes once, then compare YOLOX Nano Fast and Smart ROI using IoU matching. This is a real object-detection benchmark, not count agreement.
        </p>

        <section className="mt-8 rounded-[30px] border border-cyan-300/20 bg-[#071e27] p-5 md:p-7">
          <div className="flex items-center gap-3 text-xl font-semibold"><Upload className="h-6 w-6 text-cyan-300" />1 · Load video and sample frames</div>
          <div className="mt-5 grid gap-4 md:grid-cols-[1fr_auto]">
            <label className="flex cursor-pointer items-center justify-center gap-3 rounded-2xl border border-cyan-300/30 bg-cyan-300/10 px-5 py-4 font-medium">
              <Upload className="h-5 w-5" /> Upload video
              <input className="hidden" type="file" accept="video/*" onChange={onVideoChange} />
            </label>
            <label className="rounded-2xl border border-white/10 bg-black/15 px-4 py-3">
              <span className="text-xs text-slate-400">Frames to sample</span>
              <select className="mt-1 block w-full bg-transparent text-lg" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))}>
                <option className="bg-[#071e27]" value={3}>3 frames</option>
                <option className="bg-[#071e27]" value={5}>5 frames</option>
                <option className="bg-[#071e27]" value={7}>7 frames</option>
              </select>
            </label>
          </div>
          {videoName && <div className="mt-4 text-sm text-slate-400">{videoName} · {videoDuration.toFixed(1)} s · {frames.length} sampled frames</div>}
        </section>

        {current && (
          <section className="mt-6 rounded-[30px] border border-cyan-300/20 bg-[#071e27] p-5 md:p-7">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3 text-xl font-semibold"><Target className="h-6 w-6 text-cyan-300" />2 · Draw ground truth</div>
              <div className="text-sm text-slate-400">Frame {currentIndex + 1}/{frames.length} · {current.time.toFixed(2)} s · GT boxes: {current.gt.length}</div>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <button onClick={() => setActiveClass("human")} className={`rounded-xl px-4 py-2 text-sm font-medium ${activeClass === "human" ? "bg-cyan-300 text-[#021018]" : "border border-white/10 bg-black/15"}`}>Human</button>
              <button onClick={() => setActiveClass("vehicle")} className={`rounded-xl px-4 py-2 text-sm font-medium ${activeClass === "vehicle" ? "bg-amber-300 text-[#021018]" : "border border-white/10 bg-black/15"}`}>Vehicle</button>
              <button onClick={undo} className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-black/15 px-4 py-2 text-sm"><Undo2 className="h-4 w-4" />Undo</button>
              <button onClick={clear} className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-black/15 px-4 py-2 text-sm"><Trash2 className="h-4 w-4" />Clear frame</button>
            </div>
            <p className="mt-3 text-sm text-slate-400">Drag a box tightly around every visible target. Include partially visible people/vehicles if you want the detector to be judged on them. Use the same rule on every frame.</p>
            <canvas
              ref={canvasRef}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={() => { pointerIdRef.current = null; setDraft(null); }}
              className="mt-4 block h-auto w-full touch-none rounded-2xl border border-white/10"
            />
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <button disabled={currentIndex === 0} onClick={() => setCurrentIndex((value) => Math.max(0, value - 1))} className="rounded-xl border border-white/10 px-4 py-2 disabled:opacity-30">Previous</button>
              <div className="flex gap-2">
                {frames.map((frame, index) => (
                  <button key={frame.id} onClick={() => setCurrentIndex(index)} className={`h-9 min-w-9 rounded-full px-2 text-sm ${index === currentIndex ? "bg-cyan-300 text-[#021018]" : frame.gt.length ? "bg-emerald-400/20 text-emerald-200" : "bg-white/10 text-slate-400"}`}>{index + 1}</button>
                ))}
              </div>
              <button disabled={currentIndex === frames.length - 1} onClick={() => setCurrentIndex((value) => Math.min(frames.length - 1, value + 1))} className="rounded-xl border border-white/10 px-4 py-2 disabled:opacity-30">Next</button>
            </div>
          </section>
        )}

        {frames.length > 0 && (
          <section className="mt-6 rounded-[30px] border border-cyan-300/20 bg-[#071e27] p-5 md:p-7">
            <div className="flex items-center gap-3 text-xl font-semibold"><BarChart3 className="h-6 w-6 text-cyan-300" />3 · Run Fast vs Smart benchmark</div>
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <label className="rounded-2xl border border-white/10 bg-black/15 p-4">
                <div className="flex justify-between text-sm"><span>Detection confidence</span><span>{Math.round(confidence * 100)}%</span></div>
                <input className="mt-3 w-full" type="range" min="0.1" max="0.5" step="0.01" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} />
              </label>
              <label className="rounded-2xl border border-white/10 bg-black/15 p-4">
                <div className="flex justify-between text-sm"><span>Match IoU</span><span>{matchIou.toFixed(2)}</span></div>
                <input className="mt-3 w-full" type="range" min="0.3" max="0.7" step="0.05" value={matchIou} onChange={(e) => setMatchIou(Number(e.target.value))} />
              </label>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-4">
              <Metric label="Annotated frames" value={`${annotatedFrames}/${frames.length}`} />
              <Metric label="Ground-truth boxes" value={String(totalGt)} />
              <Metric label="Backend" value={provider} />
              <Metric label="Model input" value="416 px" />
            </div>
            <div className="mt-5 flex flex-wrap gap-3">
              <button onClick={runBenchmark} disabled={running || annotatedFrames === 0} className="inline-flex flex-1 items-center justify-center gap-3 rounded-2xl bg-cyan-300 px-5 py-4 font-semibold text-[#021018] disabled:opacity-40">
                {running ? <LoaderCircle className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
                {running ? `Benchmarking ${Math.round(runProgress * 100)}%` : "Run Fast + Smart benchmark"}
              </button>
              <button onClick={exportAnnotations} className="inline-flex items-center justify-center gap-2 rounded-2xl border border-white/10 px-5 py-4"><Download className="h-5 w-5" />Export GT JSON</button>
            </div>
            {modelState === "loading" && <div className="mt-3 text-sm text-slate-400">Loading YOLOX Nano… {Math.round(progress * 100)}%</div>}
          </section>
        )}

        {(fastResult || smartResult) && (
          <section className="mt-6 grid gap-5 lg:grid-cols-2">
            {fastResult && resultCard(fastResult, "Baseline")}
            {smartResult && resultCard(smartResult, "Adaptive ROI")}
          </section>
        )}

        {error && <div className="mt-6 rounded-2xl border border-red-400/30 bg-red-400/10 p-4 text-red-200">{error}</div>}

        <section className="mt-6 rounded-[28px] border border-white/10 bg-[#08141f] p-5 text-sm leading-6 text-slate-400">
          <div className="font-semibold text-slate-200">Benchmark rules</div>
          <p className="mt-2">TP requires the correct Human/Vehicle class and IoU at or above the selected threshold. Unmatched predictions are FP; unmatched ground-truth boxes are FN. Precision = TP/(TP+FP), Recall = TP/(TP+FN), and F1 is their harmonic mean. The pixel-band table measures each ground-truth target after scaling to the 416 px full-frame YOLOX input, which is the right way to see where tiny-target recall collapses.</p>
        </section>
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/15 p-3 text-center">
      <div className="text-xs uppercase tracking-[.16em] text-slate-500">{label}</div>
      <div className="mt-1 text-xl font-semibold text-slate-100">{value}</div>
    </div>
  );
}
