"use client";

import {
  Camera,
  CarFront,
  CheckCircle2,
  Gauge,
  LoaderCircle,
  Pause,
  Play,
  RefreshCw,
  ShieldCheck,
  Smartphone,
  Target,
  Upload,
  UserRound,
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
type FacingMode = "environment" | "user";
type SourceMode = "camera" | "image";

const RUNTIME_URL =
  "https://cdn.jsdelivr.net/npm/@ultralytics/yolo@0.0.41/dist/index.js";

const MODEL_CHOICES: Record<
  ModelChoice,
  { name: string; detail: string; url: string; badge: string }
> = {
  yolo26n: {
    name: "YOLO26-N",
    detail: "2.4M parameters · maximum browser speed baseline",
    url: "https://huggingface.co/prithivMLmods/YOLO26-ONNX/resolve/main/yolo26n/yolo26n.onnx?download=true",
    badge: "Fastest",
  },
  yolo26s: {
    name: "YOLO26-S",
    detail: "9.5M parameters · stronger accuracy while targeting live speed",
    url: "https://huggingface.co/prithivMLmods/YOLO26-ONNX/resolve/main/yolo26s/yolo26s.onnx?download=true",
    badge: "Recommended",
  },
};

// COCO class ids: person, bicycle, car, motorcycle, bus, truck.
const HUMAN_VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7];
const VEHICLE_NAMES = new Set(["bicycle", "car", "motorcycle", "bus", "truck"]);
const TARGET_INTERVAL_MS = 50;

function isHuman(box: YoloBox) {
  return box.name.toLowerCase() === "person";
}

function isVehicle(box: YoloBox) {
  return VEHICLE_NAMES.has(box.name.toLowerCase());
}

function formatMs(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "—";
  return value < 10 ? value.toFixed(1) + " ms" : Math.round(value) + " ms";
}

export default function LivePage() {
  const [modelChoice, setModelChoice] = useState<ModelChoice>("yolo26n");
  const [sourceMode, setSourceMode] = useState<SourceMode>("camera");
  const [threshold, setThreshold] = useState(0.2);
  const [running, setRunning] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [modelState, setModelState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [modelProgress, setModelProgress] = useState(0);
  const [modelFile, setModelFile] = useState("");
  const [device, setDevice] = useState("waiting");
  const [boxes, setBoxes] = useState<YoloBox[]>([]);
  const [totalMs, setTotalMs] = useState(0);
  const [inferenceMs, setInferenceMs] = useState(0);
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

  const humans = useMemo(() => boxes.filter(isHuman).length, [boxes]);
  const vehicles = useMemo(() => boxes.filter(isVehicle).length, [boxes]);
  const effectiveFps = totalMs ? Math.min(20, 1000 / totalMs) : 0;
  const insideTarget = totalMs > 0 && totalMs <= TARGET_INTERVAL_MS;
  const activeChoice = MODEL_CHOICES[modelChoice];

  const stopCamera = useCallback(() => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setRunning(false);
  }, []);

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
    setModelFile("Downloading " + activeChoice.name + " model");
    setError("");
    setBoxes([]);

    try {
      modelRef.current?.free();
      modelRef.current = null;

      const response = await fetch(activeChoice.url);
      if (!response.ok) {
        throw new Error("Model download failed with HTTP " + response.status + ".");
      }

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
              setModelFile(
                "Downloading " +
                  activeChoice.name +
                  " · " +
                  (received / 1024 / 1024).toFixed(1) +
                  " MB",
              );
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
      const runtime = (await import(
        /* @vite-ignore */ moduleUrl
      )) as unknown as YoloRuntime;

      const prefersWebGpu =
        typeof navigator !== "undefined" && "gpu" in navigator;
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
      setError(caught instanceof Error ? caught.message : "The YOLO model could not load.");
    }
  }, [activeChoice.name, activeChoice.url]);

  useEffect(() => {
    void loadModel();
  }, [loadModel]);

  const drawBoxes = useCallback((nextBoxes: YoloBox[], width: number, height: number) => {
    const canvas = overlayRef.current;
    if (!canvas || !width || !height) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    context.lineWidth = Math.max(2, width / 500);
    context.font = "700 " + Math.max(13, Math.round(width / 85)) + "px Arial";
    context.textBaseline = "top";

    nextBoxes.forEach((box) => {
      const human = isHuman(box);
      const color = human ? "#5eead4" : "#fcd34d";
      const label =
        (human ? "Human" : box.name) + " " + Math.round(box.conf * 100) + "%";
      const x = Math.max(0, box.x1);
      const y = Math.max(0, box.y1);
      const w = Math.max(1, box.x2 - box.x1);
      const h = Math.max(1, box.y2 - box.y1);
      context.strokeStyle = color;
      context.fillStyle = color + "18";
      context.strokeRect(x, y, w, h);
      context.fillRect(x, y, w, h);
      const metrics = context.measureText(label);
      const labelH = Math.max(21, width / 55);
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
    if (source instanceof HTMLVideoElement && !source.videoWidth) return;
    if (source instanceof HTMLImageElement && !source.naturalWidth) return;

    busyRef.current = true;
    const started = performance.now();
    try {
      const result = await modelRef.current.predict(source, {
        conf: Math.max(0.05, threshold),
        iou: 0.7,
        classes: HUMAN_VEHICLE_CLASSES,
      });
      const accepted = result.boxes.filter(
        (box) => (isHuman(box) || isVehicle(box)) && box.conf >= threshold,
      );
      const elapsed = performance.now() - started;
      setBoxes(accepted);
      setTotalMs(elapsed);
      setInferenceMs(result.speed?.inference || 0);
      setDevice(modelRef.current.device);
      drawBoxes(accepted, result.width, result.height);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Live inference failed.");
      setRunning(false);
    } finally {
      busyRef.current = false;
    }
  }, [drawBoxes, sourceMode, threshold]);

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

  const startCamera = useCallback(
    async (nextFacing: FacingMode) => {
      stopCamera();
      setSourceMode("camera");
      setError("");
      setBoxes([]);
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("Camera requires HTTPS and a supported browser.");
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: nextFacing },
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
    },
    [stopCamera],
  );

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
    setTotalMs(0);
    setInferenceMs(0);
    event.target.value = "";
  }, [stopCamera]);

  const switchModel = useCallback((next: ModelChoice) => {
    if (next === modelChoice) return;
    loadIdRef.current += 1;
    modelRef.current?.free();
    modelRef.current = null;
    setModelChoice(next);
    setModelState("idle");
    setBoxes([]);
    setTotalMs(0);
    setInferenceMs(0);
  }, [modelChoice]);

  return (
    <main className="min-h-screen bg-[#021018] text-slate-100">
      <header className="border-b border-white/10 bg-[#03151d]/95">
        <div className="mx-auto flex max-w-[1500px] flex-col gap-4 px-5 py-5 sm:px-8 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="mb-1 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.32em] text-cyan-300">
              <Zap size={14} /> 50 ms real-time experiment
            </div>
            <h1 className="text-xl font-bold sm:text-2xl">Human + Vehicle Live Detector</h1>
            <p className="mt-1 text-xs text-slate-400">YOLO26 · 1-pass · WebGPU when available · 20 analysis FPS target</p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-slate-300">
              Device: <strong className="text-cyan-200">{device.toUpperCase()}</strong>
            </span>
            <a
              href="./"
              className="rounded-full border border-cyan-300/25 bg-cyan-300/10 px-3 py-1.5 font-semibold text-cyan-100 hover:bg-cyan-300/20"
            >
              D-FINE Lab
            </a>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1500px] gap-5 px-4 py-6 sm:px-8 xl:grid-cols-[minmax(0,1fr)_390px]">
        <section className="min-w-0 overflow-hidden rounded-[28px] border border-white/10 bg-[#061d26]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
            <div className="flex items-center gap-3">
              <Target className="text-cyan-300" size={22} />
              <div>
                <h2 className="text-sm font-semibold">Live analysis viewport</h2>
                <p className="text-xs text-slate-400">Only person + CCTV vehicle classes are returned</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className={"h-2 w-2 rounded-full " + (running ? "animate-pulse bg-emerald-400" : "bg-slate-600")} />
              {running ? "Running · 50 ms cadence" : "Paused"}
            </div>
          </div>

          <div className="relative flex min-h-[380px] items-center justify-center overflow-hidden bg-black lg:min-h-[590px]">
            {sourceMode === "camera" ? (
              <div className="relative w-full">
                <video
                  ref={videoRef}
                  muted
                  playsInline
                  autoPlay
                  className={"block h-auto min-h-[380px] w-full object-contain lg:min-h-[590px] " + (cameraActive ? "" : "opacity-0")}
                />
                <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
                {!cameraActive ? (
                  <div className="absolute inset-0 grid place-items-center p-8 text-center">
                    <div>
                      <Camera size={54} strokeWidth={1.3} className="mx-auto mb-4 text-cyan-300/60" />
                      <p className="font-semibold text-white">Start a camera to test live speed</p>
                      <p className="mt-2 text-sm text-slate-400">Use rear camera for CCTV-like distant targets or webcam on desktop.</p>
                    </div>
                  </div>
                ) : null}
              </div>
            ) : imageUrl ? (
              <div className="relative w-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  ref={imageRef}
                  src={imageUrl}
                  alt="YOLO26 test"
                  className="block h-auto w-full"
                  onLoad={() => void analyse()}
                />
                <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
              </div>
            ) : (
              <div className="p-8 text-center text-slate-400">Choose a test image.</div>
            )}

            {modelState === "loading" ? (
              <div className="absolute left-4 top-4 max-w-[310px] rounded-2xl border border-cyan-200/20 bg-[#03151ded] px-4 py-3 shadow-2xl backdrop-blur">
                <div className="flex items-center gap-3">
                  <LoaderCircle className="animate-spin text-cyan-300" size={20} />
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-white">Loading {activeChoice.name}</div>
                    <div className="mt-0.5 truncate text-[10px] text-slate-400">{modelFile}</div>
                  </div>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10">
                  <div className="h-full bg-cyan-300 transition-all" style={{ width: modelProgress + "%" }} />
                </div>
              </div>
            ) : null}
          </div>

          {error ? (
            <div className="flex items-start gap-2 border-t border-red-300/20 bg-red-400/10 px-5 py-3 text-xs leading-5 text-red-200">
              <XCircle size={16} className="mt-0.5 shrink-0" /> {error}
            </div>
          ) : null}

          <div className="grid grid-cols-2 border-t border-white/10 sm:grid-cols-6">
            {[
              ["Detections", boxes.length],
              ["Humans", humans],
              ["Vehicles", vehicles],
              ["Total", formatMs(totalMs)],
              ["AI only", formatMs(inferenceMs)],
              ["Effective", effectiveFps ? effectiveFps.toFixed(1) + " FPS" : "—"],
            ].map(([label, value]) => (
              <div key={String(label)} className="border-r border-white/10 px-4 py-4 last:border-r-0">
                <div className="text-[9px] font-semibold uppercase tracking-[0.18em] text-slate-500">{label}</div>
                <div className="mt-1 text-lg font-bold text-white">{value}</div>
              </div>
            ))}
          </div>

          {totalMs ? (
            <div className={"flex items-center gap-2 border-t px-5 py-3 text-xs " + (insideTarget ? "border-emerald-300/20 bg-emerald-400/10 text-emerald-200" : "border-amber-300/20 bg-amber-400/10 text-amber-100")}>
              {insideTarget ? <CheckCircle2 size={16} /> : <Gauge size={16} />}
              {insideTarget
                ? "50 ms target achieved on this device."
                : "Above 50 ms on this device. Try YOLO26-N, WebGPU, or a stronger GPU."}
            </div>
          ) : null}
        </section>

        <aside className="flex flex-col gap-5">
          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold"><Gauge size={18} className="text-cyan-300" /> 1 · Choose live model</h2>
            <div className="grid gap-2">
              {(Object.keys(MODEL_CHOICES) as ModelChoice[]).map((key) => {
                const item = MODEL_CHOICES[key];
                const active = modelChoice === key;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => switchModel(key)}
                    className={"relative rounded-2xl border p-4 text-left " + (active ? "border-cyan-300/60 bg-cyan-300/10" : "border-white/10 bg-[#03161e]")}
                  >
                    <span className="block pr-20 text-sm font-semibold text-white">{item.name}</span>
                    <span className="mt-1 block text-xs leading-5 text-slate-400">{item.detail}</span>
                    <span className="absolute right-3 top-3 rounded-full bg-white/10 px-2 py-1 text-[9px] font-bold uppercase text-cyan-100">{item.badge}</span>
                  </button>
                );
              })}
            </div>
            <button
              type="button"
              onClick={() => void loadModel()}
              className="mt-3 flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm font-semibold hover:bg-white/10"
            >
              <RefreshCw size={16} /> Reload selected model
            </button>
          </section>

          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold"><Camera size={18} className="text-cyan-300" /> 2 · Source</h2>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => void startCamera("environment")}
                className="flex min-h-12 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm font-semibold hover:border-cyan-300/40"
              >
                <Smartphone size={17} /> Rear camera
              </button>
              <button
                type="button"
                onClick={() => void startCamera("user")}
                className="flex min-h-12 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm font-semibold hover:border-cyan-300/40"
              >
                <Camera size={17} /> Webcam
              </button>
            </div>
            <label className="mt-2 flex min-h-12 cursor-pointer items-center justify-center gap-2 rounded-xl bg-cyan-300 font-semibold text-[#021118] hover:bg-cyan-200">
              <Upload size={17} /> Test an image
              <input type="file" accept="image/*" className="hidden" onChange={handleImage} />
            </label>
            {sourceMode === "image" && imageUrl ? (
              <button
                type="button"
                disabled={modelState !== "ready"}
                onClick={() => void analyse()}
                className="mt-2 flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm disabled:opacity-50"
              >
                <Target size={16} /> Analyse image
              </button>
            ) : null}
            {cameraActive ? (
              <button
                type="button"
                onClick={() => setRunning((value) => !value)}
                className="mt-2 flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm"
              >
                {running ? <Pause size={17} /> : <Play size={17} />}
                {running ? "Pause detection" : "Resume detection"}
              </button>
            ) : null}
          </section>

          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold"><ShieldCheck size={18} className="text-cyan-300" /> 3 · Detection</h2>
            <label className="block rounded-2xl border border-white/10 bg-[#03161e] p-4">
              <span className="flex items-center justify-between text-xs text-slate-300">
                Confidence <strong className="text-cyan-200">{Math.round(threshold * 100)}%</strong>
              </span>
              <input
                type="range"
                min="5"
                max="70"
                value={Math.round(threshold * 100)}
                onChange={(event) => setThreshold(Number(event.target.value) / 100)}
                className="mt-4 w-full accent-cyan-300"
              />
            </label>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-xl bg-emerald-300/8 p-3 text-emerald-200"><UserRound size={17} className="mb-2" />Human</div>
              <div className="rounded-xl bg-amber-300/8 p-3 text-amber-200"><CarFront size={18} className="mb-2" />Vehicles</div>
            </div>
            <p className="mt-3 text-[11px] leading-5 text-slate-500">
              Runtime output is restricted to person, bicycle, car, motorcycle, bus and truck. The pretrained network is still COCO; a future CCTV-specific Human/Vehicle fine-tune is needed for a true reduced-class model.
            </p>
          </section>

          <section className="rounded-[22px] border border-amber-300/15 bg-amber-300/5 p-4 text-[10px] leading-5 text-amber-100/70">
            YOLO26 runtime/weights are AGPL-3.0. This page is an experimental browser benchmark; review Ultralytics licensing before using it in a closed-source commercial product.
          </section>
        </aside>
      </div>
    </main>
  );
}
