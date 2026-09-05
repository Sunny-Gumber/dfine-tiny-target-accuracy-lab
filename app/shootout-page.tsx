"use client";

import {
  CheckCircle2,
  Cpu,
  FlaskConical,
  Gauge,
  Layers3,
  LoaderCircle,
  RefreshCw,
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

type SourceInput = HTMLImageElement | HTMLCanvasElement;
type Mode = "fast" | "smart";
type ModelChoice = "yolo26n" | "rfdetr-n" | "yolox-n" | "yolo9-t";
type ModelFamily = "yolo" | "yolox" | "yolo9" | "rfdetr";

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
  badge: string;
  detail: string;
  source: string;
  family: ModelFamily;
  inputSize: number;
  size: string;
  role: string;
};

const LIBRE_RUNTIME_URL =
  "https://esm.sh/libreyolo-web@0.0.6?bundle&deps=onnxruntime-web@1.24.3";

const YOLO26_N_URL =
  "https://huggingface.co/prithivMLmods/YOLO26-ONNX/resolve/main/yolo26n/yolo26n.onnx?download=true";

const MODEL_CHOICES: Record<ModelChoice, ModelSpec> = {
  yolo26n: {
    name: "YOLO26-N",
    badge: "Current winner",
    detail: "Our existing speed baseline, now run through the same ONNX Runtime wrapper as the challengers.",
    source: YOLO26_N_URL,
    family: "yolo",
    inputSize: 640,
    size: "~2.4M params",
    role: "Baseline",
  },
  "rfdetr-n": {
    name: "RF-DETR Nano",
    badge: "Transformer",
    detail: "384 px transformer challenger. Strong accuracy/latency design, but much larger model download.",
    source: "LibreRFDETRn",
    family: "rfdetr",
    inputSize: 384,
    size: "~103 MB ONNX",
    role: "Accuracy challenger",
  },
  "yolox-n": {
    name: "YOLOX Nano",
    badge: "Tiny runtime",
    detail: "Very small 416 px model. Included to see the fastest practical floor on this phone.",
    source: "LibreYOLOXn",
    family: "yolox",
    inputSize: 416,
    size: "~3.6 MB ONNX",
    role: "Speed challenger",
  },
  "yolo9-t": {
    name: "YOLO9 Tiny",
    badge: "Compact 640",
    detail: "Small 640 px model. Useful to test whether higher input size beats the tiny 416 px option.",
    source: "LibreYOLO9t",
    family: "yolo9",
    inputSize: 640,
    size: "~8 MB ONNX",
    role: "Balanced challenger",
  },
};

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

const TARGET_IDS = new Set([0, 1, 2, 3, 5, 7]);
const VEHICLE_IDS = new Set([1, 2, 3, 5, 7]);

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

  const rawX1 = Math.min(...cluster.map((box) => box.x1));
  const rawY1 = Math.min(...cluster.map((box) => box.y1));
  const rawX2 = Math.max(...cluster.map((box) => box.x2));
  const rawY2 = Math.max(...cluster.map((box) => box.y2));
  const centerX = (rawX1 + rawX2) / 2;
  const centerY = (rawY1 + rawY2) / 2;
  const roiWidth = Math.min(width * 0.74, Math.max(width * 0.38, (rawX2 - rawX1) * 2.4));
  const roiHeight = Math.min(height * 0.68, Math.max(height * 0.38, (rawY2 - rawY1) * 2.6));
  const x = clamp(centerX - roiWidth / 2, 0, width - roiWidth);
  const y = clamp(centerY - roiHeight / 2, 0, height - roiHeight);

  return {
    x,
    y,
    width: roiWidth,
    height: roiHeight,
    label: "AutoFocus-style tiny/weak cluster",
  };
}

function formatMs(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "—";
  return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;
}

function approximateModelHeight(box: Detection, imageWidth: number, imageHeight: number, spec: ModelSpec) {
  const sourceHeight = Math.max(1, box.y2 - box.y1);
  if (spec.family === "yolox" || spec.family === "yolo") {
    const scale = Math.min(spec.inputSize / imageWidth, spec.inputSize / imageHeight);
    return sourceHeight * scale;
  }
  return sourceHeight * (spec.inputSize / imageHeight);
}

function bandLabel(px: number) {
  if (px < 16) return "<16";
  if (px < 32) return "16–32";
  if (px < 64) return "32–64";
  if (px < 128) return "64–128";
  return ">128";
}

export default function ShootoutPage() {
  const [modelChoice, setModelChoice] = useState<ModelChoice>("yolo26n");
  const [mode, setMode] = useState<Mode>("smart");
  const [threshold, setThreshold] = useState(0.2);
  const [modelState, setModelState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [provider, setProvider] = useState("waiting");
  const [imageUrl, setImageUrl] = useState("");
  const [boxes, setBoxes] = useState<Detection[]>([]);
  const [roi, setRoi] = useState<Region | null>(null);
  const [fullMs, setFullMs] = useState(0);
  const [roiMs, setRoiMs] = useState(0);
  const [totalMs, setTotalMs] = useState(0);
  const [roiGain, setRoiGain] = useState(0);
  const [passes, setPasses] = useState(0);
  const [warmupDone, setWarmupDone] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  const imageRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<LibreModel | null>(null);
  const objectUrlRef = useRef("");
  const loadIdRef = useRef(0);
  const busyRef = useRef(false);

  const spec = MODEL_CHOICES[modelChoice];
  const humans = useMemo(() => boxes.filter(isHuman).length, [boxes]);
  const vehicles = useMemo(() => boxes.filter(isVehicle).length, [boxes]);

  const modelBands = useMemo(() => {
    const result: Record<string, number> = { "<16": 0, "16–32": 0, "32–64": 0, "64–128": 0, ">128": 0 };
    const img = imageRef.current;
    if (!img?.naturalWidth) return result;
    boxes.forEach((box) => {
      const px = approximateModelHeight(box, img.naturalWidth, img.naturalHeight, spec);
      result[bandLabel(px)] += 1;
    });
    return result;
  }, [boxes, spec]);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      void modelRef.current?.release();
    };
  }, []);

  const clearResults = useCallback(() => {
    setBoxes([]);
    setRoi(null);
    setFullMs(0);
    setRoiMs(0);
    setTotalMs(0);
    setRoiGain(0);
    setPasses(0);
    const canvas = overlayRef.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const loadModel = useCallback(async () => {
    const loadId = ++loadIdRef.current;
    setModelState("loading");
    setProgress(0);
    setError("");
    setStatus(`Loading ${spec.name}…`);
    setWarmupDone(false);
    clearResults();

    try {
      if (modelRef.current) {
        await modelRef.current.release();
        modelRef.current = null;
      }

      const moduleUrl: string = LIBRE_RUNTIME_URL;
      const runtime = (await import(/* @vite-ignore */ moduleUrl)) as unknown as LibreRuntime;
      const options = {
        device: ["webgpu", "wasm"] as ("webgpu" | "wasm")[],
        confThres: 0.05,
        iouThres: 0.65,
        maxDet: 500,
        modelFamily: spec.family,
        inputSize: spec.inputSize,
        onProgress: (value: number) => {
          if (loadId === loadIdRef.current) setProgress(Math.round(value * 100));
        },
      };
      const model = await runtime.loadModel(spec.source, options);
      if (loadId !== loadIdRef.current) {
        await model.release();
        return;
      }
      modelRef.current = model;
      setProvider((model.provider || "unknown").toUpperCase());
      setModelState("ready");
      setProgress(100);
      setStatus(`${spec.name} ready · ${model.provider || "backend ready"}`);
    } catch (caught) {
      if (loadId !== loadIdRef.current) return;
      setModelState("error");
      setError(caught instanceof Error ? caught.message : "Model failed to load.");
      setStatus("");
    }
  }, [clearResults, spec]);

  useEffect(() => {
    void loadModel();
  }, [loadModel]);

  const draw = useCallback((nextBoxes: Detection[], nextRoi: Region | null) => {
    const img = imageRef.current;
    const canvas = overlayRef.current;
    if (!img || !canvas || !img.naturalWidth || !img.naturalHeight) return;
    if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (nextRoi) {
      ctx.save();
      ctx.strokeStyle = "#f472b6";
      ctx.lineWidth = Math.max(3, canvas.width / 500);
      ctx.setLineDash([14, 9]);
      ctx.strokeRect(nextRoi.x, nextRoi.y, nextRoi.width, nextRoi.height);
      ctx.restore();
    }

    const fontSize = Math.max(14, Math.round(canvas.width / 85));
    ctx.font = `700 ${fontSize}px Arial`;
    ctx.textBaseline = "top";
    nextBoxes.forEach((box) => {
      const color = box.origin === "roi" ? "#f472b6" : isHuman(box) ? "#5eead4" : "#fcd34d";
      const x = Math.max(0, box.x1);
      const y = Math.max(0, box.y1);
      const w = Math.max(1, box.x2 - box.x1);
      const h = Math.max(1, box.y2 - box.y1);
      const label = `${isHuman(box) ? "Human" : box.name} ${Math.round(box.conf * 100)}%`;
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, canvas.width / 550);
      ctx.strokeRect(x, y, w, h);
      const labelH = fontSize + 8;
      const labelW = ctx.measureText(label).width + 12;
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - labelH), labelW, labelH);
      ctx.fillStyle = "#021018";
      ctx.fillText(label, x + 6, Math.max(1, y - labelH + 3));
    });
  }, []);

  const analyse = useCallback(async () => {
    if (busyRef.current || !modelRef.current || !imageRef.current?.naturalWidth) return;
    const model = modelRef.current;
    const source = imageRef.current;
    busyRef.current = true;
    setError("");

    try {
      if (!warmupDone) {
        setStatus(`Warming up ${spec.name} — this run is not timed…`);
        await model.predict(source, { confThres: 0.4, iouThres: 0.65, maxDet: 50 });
        setWarmupDone(true);
      }

      setStatus(`Analysing with ${spec.name}…`);
      const totalStarted = performance.now();
      const fullStarted = performance.now();
      const fullResult = await model.predict(source, {
        confThres: 0.05,
        iouThres: 0.65,
        maxDet: 500,
      });
      const measuredFull = performance.now() - fullStarted;
      const fullAccepted = mergeDetections(
        convertDetections(fullResult.detections, threshold, "full"),
      );

      let finalBoxes = fullAccepted;
      let nextRoi: Region | null = null;
      let measuredRoi = 0;
      let gain = 0;
      let passCount = 1;

      if (mode === "smart") {
        nextRoi = chooseSmartRegion(fullAccepted, source.naturalWidth, source.naturalHeight, threshold);
        const crop = cropRegion(source, nextRoi);
        const roiStarted = performance.now();
        const roiResult = await model.predict(crop, {
          confThres: 0.04,
          iouThres: 0.65,
          maxDet: 500,
        });
        measuredRoi = performance.now() - roiStarted;
        const roiGate = Math.max(0.08, threshold * 0.72);
        const roiAccepted = convertDetections(
          roiResult.detections,
          roiGate,
          "roi",
          nextRoi.x,
          nextRoi.y,
        );
        finalBoxes = mergeDetections([...fullAccepted, ...roiAccepted]);
        gain = Math.max(0, finalBoxes.length - fullAccepted.length);
        passCount = 2;
      }

      const measuredTotal = performance.now() - totalStarted;
      setBoxes(finalBoxes);
      setRoi(nextRoi);
      setFullMs(measuredFull);
      setRoiMs(measuredRoi);
      setTotalMs(measuredTotal);
      setRoiGain(gain);
      setPasses(passCount);
      setProvider((model.provider || "unknown").toUpperCase());
      setStatus(`${spec.name} complete · timed after warm-up`);
      draw(finalBoxes, nextRoi);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Inference failed.");
      setStatus("");
    } finally {
      busyRef.current = false;
    }
  }, [draw, mode, spec.name, threshold, warmupDone]);

  const handleImage = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setImageUrl(url);
    clearResults();
    event.target.value = "";
  }, [clearResults]);

  const switchModel = useCallback((next: ModelChoice) => {
    if (next === modelChoice) return;
    loadIdRef.current += 1;
    void modelRef.current?.release();
    modelRef.current = null;
    setModelChoice(next);
    setModelState("idle");
    setProvider("waiting");
    setWarmupDone(false);
    clearResults();
  }, [clearResults, modelChoice]);

  const effectiveFps = totalMs ? 1000 / totalMs : 0;

  return (
    <main className="min-h-screen bg-[#021018] text-slate-100">
      <header className="border-b border-white/10 bg-[#03151d]/95">
        <div className="mx-auto flex max-w-[1500px] flex-col gap-4 px-5 py-5 sm:px-8 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="mb-1 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.32em] text-violet-300">
              <FlaskConical size={14} /> browser model shootout
            </div>
            <h1 className="text-xl font-bold sm:text-2xl">Tiny-Target Model Shootout</h1>
            <p className="mt-1 max-w-3xl text-xs leading-5 text-slate-400">
              Same image · same Human/Vehicle filter · same WebGPU/WASM runtime · optional identical Smart ROI pass.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5">Backend: <strong className="text-cyan-200">{provider}</strong></span>
            <a href="./smart.html" className="rounded-full border border-pink-300/25 bg-pink-300/10 px-3 py-1.5 font-semibold text-pink-100">Smart Lab</a>
            <a href="./live.html" className="rounded-full border border-cyan-300/25 bg-cyan-300/10 px-3 py-1.5 font-semibold text-cyan-100">50 ms Live</a>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1500px] gap-5 px-4 py-6 sm:px-8 xl:grid-cols-[minmax(0,1fr)_410px]">
        <section className="min-w-0 overflow-hidden rounded-[28px] border border-white/10 bg-[#061d26]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
            <div className="flex items-center gap-3">
              <Target className="text-violet-300" size={22} />
              <div>
                <h2 className="text-sm font-semibold">A/B benchmark viewport</h2>
                <p className="text-xs text-slate-400">Pink boxes = recovered by Smart ROI only</p>
              </div>
            </div>
            <span className="rounded-full bg-white/5 px-3 py-1 text-[11px] text-slate-300">{spec.name} · {mode === "smart" ? "2-pass Smart" : "1-pass Fast"}</span>
          </div>

          <div className="relative flex min-h-[380px] items-center justify-center overflow-hidden bg-black lg:min-h-[600px]">
            {imageUrl ? (
              <div className="relative w-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img ref={imageRef} src={imageUrl} alt="Model benchmark" className="block h-auto w-full" onLoad={() => clearResults()} />
                <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
              </div>
            ) : (
              <div className="p-10 text-center">
                <Upload size={54} strokeWidth={1.3} className="mx-auto mb-4 text-violet-300/60" />
                <p className="font-semibold text-white">Upload the same traffic image</p>
                <p className="mt-2 text-sm text-slate-400">Then switch models without changing confidence or processing mode.</p>
              </div>
            )}

            {modelState === "loading" ? (
              <div className="absolute left-4 top-4 min-w-[260px] max-w-[330px] rounded-2xl border border-violet-200/20 bg-[#03151dee] px-4 py-3 shadow-2xl backdrop-blur">
                <div className="flex items-center gap-3">
                  <LoaderCircle className="animate-spin text-violet-300" size={20} />
                  <div>
                    <div className="text-xs font-semibold">Loading {spec.name}</div>
                    <div className="mt-1 text-[10px] text-slate-400">Model download / ONNX session · {progress}%</div>
                  </div>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full bg-violet-300 transition-all" style={{ width: `${progress}%` }} /></div>
              </div>
            ) : null}
          </div>

          {error ? <div className="flex items-start gap-2 border-t border-red-300/20 bg-red-400/10 px-5 py-3 text-xs text-red-200"><XCircle size={16} /> {error}</div> : null}
          {status ? <div className="border-t border-white/10 px-5 py-2 text-[11px] text-slate-400">{status}</div> : null}

          <div className="grid grid-cols-2 border-t border-white/10 sm:grid-cols-4 lg:grid-cols-8">
            {[
              ["Detections", boxes.length],
              ["Humans", humans],
              ["Vehicles", vehicles],
              ["ROI gain", mode === "smart" ? `+${roiGain}` : "+0"],
              ["Passes", passes || "—"],
              ["Full pass", formatMs(fullMs)],
              ["ROI pass", roiMs ? formatMs(roiMs) : "—"],
              ["Total", formatMs(totalMs)],
            ].map(([label, value]) => (
              <div key={String(label)} className="border-r border-t border-white/10 px-3 py-4 first:border-t-0 sm:border-t-0">
                <div className="text-[8px] font-semibold uppercase tracking-[0.16em] text-slate-500">{label}</div>
                <div className="mt-1 text-lg font-bold text-white">{value}</div>
              </div>
            ))}
          </div>

          {totalMs ? (
            <div className="border-t border-violet-300/15 bg-violet-400/8 px-5 py-3 text-xs text-violet-100">
              Effective timed rate: <strong>{effectiveFps.toFixed(1)} FPS</strong>. Timing starts after one automatic warm-up run; model load/download is excluded.
            </div>
          ) : null}

          <div className="border-t border-white/10 px-5 py-4">
            <div className="mb-3 text-[9px] font-semibold uppercase tracking-[0.22em] text-slate-500">Approx target height seen by full-frame model</div>
            <div className="grid grid-cols-5 gap-2">
              {["<16", "16–32", "32–64", "64–128", ">128"].map((label) => (
                <div key={label} className="rounded-xl border border-white/10 bg-white/5 px-2 py-3 text-center">
                  <div className="text-base font-bold text-cyan-100">{modelBands[label]}</div>
                  <div className="mt-1 text-[9px] text-slate-500">{label} px</div>
                </div>
              ))}
            </div>
            <p className="mt-3 text-[10px] leading-4 text-slate-500">Approximation uses each model's input size and resize style; it is more useful than the old source-coordinate-only pixel bands, but it is not a formal DORI measurement.</p>
          </div>
        </section>

        <aside className="flex flex-col gap-5">
          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold"><Cpu size={18} className="text-violet-300" /> 1 · Choose model</h2>
            <div className="grid gap-2">
              {(Object.keys(MODEL_CHOICES) as ModelChoice[]).map((key) => {
                const item = MODEL_CHOICES[key];
                const active = key === modelChoice;
                return (
                  <button key={key} type="button" onClick={() => switchModel(key)} className={`relative rounded-2xl border p-4 text-left transition ${active ? "border-violet-300/60 bg-violet-300/10" : "border-white/10 bg-[#03161e]"}`}>
                    <span className="block pr-24 text-sm font-semibold text-white">{item.name}</span>
                    <span className="mt-1 block text-[11px] text-cyan-100">{item.role} · {item.size} · {item.inputSize}px</span>
                    <span className="mt-2 block text-xs leading-5 text-slate-400">{item.detail}</span>
                    <span className="absolute right-3 top-3 rounded-full bg-white/10 px-2 py-1 text-[8px] font-bold uppercase text-violet-100">{item.badge}</span>
                  </button>
                );
              })}
            </div>
            <button type="button" onClick={() => void loadModel()} className="mt-3 flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm font-semibold"><RefreshCw size={16} /> Reload selected model</button>
          </section>

          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold"><Layers3 size={18} className="text-pink-300" /> 2 · Processing</h2>
            <div className="grid grid-cols-2 gap-2">
              <button type="button" onClick={() => { setMode("fast"); clearResults(); }} className={`rounded-xl border p-3 text-left ${mode === "fast" ? "border-cyan-300/60 bg-cyan-300/10" : "border-white/10 bg-white/5"}`}>
                <div className="text-sm font-semibold">Fast · 1 pass</div><div className="mt-1 text-[10px] text-slate-400">Pure model comparison</div>
              </button>
              <button type="button" onClick={() => { setMode("smart"); clearResults(); }} className={`rounded-xl border p-3 text-left ${mode === "smart" ? "border-pink-300/60 bg-pink-300/10" : "border-white/10 bg-white/5"}`}>
                <div className="text-sm font-semibold">Smart · 2 pass</div><div className="mt-1 text-[10px] text-slate-400">Same adaptive ROI logic</div>
              </button>
            </div>
          </section>

          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold"><Upload size={18} className="text-cyan-300" /> 3 · Image & threshold</h2>
            <label className="flex min-h-12 cursor-pointer items-center justify-center gap-2 rounded-xl bg-cyan-300 font-semibold text-[#021118] hover:bg-cyan-200"><Upload size={17} /> Test image<input type="file" accept="image/*" className="hidden" onChange={handleImage} /></label>
            <button type="button" disabled={!imageUrl || modelState !== "ready" || busyRef.current} onClick={() => void analyse()} className="mt-2 flex min-h-12 w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm font-semibold disabled:opacity-40"><Target size={17} /> Analyse image</button>
            <div className="mt-4 rounded-xl border border-white/10 bg-[#03161e] p-4">
              <div className="flex justify-between text-xs"><span>Confidence</span><strong className="text-cyan-200">{Math.round(threshold * 100)}%</strong></div>
              <input className="mt-3 w-full accent-cyan-300" type="range" min="0.1" max="0.6" step="0.01" value={threshold} onChange={(event) => { setThreshold(Number(event.target.value)); clearResults(); }} />
            </div>
          </section>

          <section className="rounded-[26px] border border-emerald-300/15 bg-emerald-400/5 p-5 text-xs leading-5 text-slate-300">
            <div className="mb-2 flex items-center gap-2 font-semibold text-emerald-100"><CheckCircle2 size={17} /> Fair-test rules</div>
            Keep the same image, confidence and mode. Run each model at least twice after load; the page automatically discards one warm-up inference before starting the timer.
          </section>

          <section className="rounded-[26px] border border-amber-300/15 bg-amber-400/5 p-5 text-xs leading-5 text-slate-300">
            <div className="mb-2 flex items-center gap-2 font-semibold text-amber-100"><Gauge size={17} /> Research models not faked into the UI</div>
            <p><strong>LW-DETR-T:</strong> official checkpoints and ONNX export exist, but we have not verified a ready browser checkpoint/pre-postprocessor for this page.</p>
            <p className="mt-2"><strong>TinyFormer:</strong> the official project supports ONNX export and is especially relevant to tiny objects, but the current browser runtime does not yet expose a verified TinyFormer model family. We will add it only after a real browser export is proven.</p>
          </section>

          <div className="flex items-center gap-2 rounded-[22px] border border-white/10 bg-[#03161e] px-4 py-3 text-[10px] text-slate-500"><Zap size={14} /> Goal: beat YOLO26-N Smart on recall without making mobile latency worse.</div>
        </aside>
      </div>
    </main>
  );
}
