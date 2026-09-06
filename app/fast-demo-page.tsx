"use client";

import {
  Camera,
  Cpu,
  Gauge,
  Image as ImageIcon,
  LoaderCircle,
  Pause,
  Play,
  RefreshCw,
  Smartphone,
  Upload,
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

type FacingMode = "environment" | "user";
type SourceMode = "camera" | "image";
type TargetKind = "Human" | "Car" | "Motorcycle" | "Bus/Truck";

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
    input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    options?: { confThres?: number; iouThres?: number; maxDet?: number },
  ) => Promise<LibreResult>;
  release: () => Promise<void>;
  provider: "webgpu" | "wasm" | null;
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

type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  classId: number;
  kind: TargetKind;
};

const LIBRE_RUNTIME_URL =
  "https://esm.sh/libreyolo-web@0.0.6?bundle&deps=onnxruntime-web@1.24.3";
const MODEL_SOURCE = "LibreYOLOXn";
const LIVE_INPUT = 320;
const IMAGE_INPUT = 416;
const CAMERA_WIDTH = 1280;
const CAMERA_HEIGHT = 720;
const MIN_LOOP_GAP_MS = 45;
const CAMERA_WARMUP_RUNS = 3;

function classToKind(classId: number): TargetKind | null {
  if (classId === 0) return "Human";
  if (classId === 2) return "Car";
  if (classId === 3) return "Motorcycle";
  if (classId === 5 || classId === 7) return "Bus/Truck";
  return null;
}

function formatMs(value: number) {
  if (!value || !Number.isFinite(value)) return "—";
  return value < 10 ? value.toFixed(1) + " ms" : Math.round(value) + " ms";
}

function Stat({ label, value, note }: { label: string; value: string | number; note?: string }) {
  return (
    <div className="min-w-0 border-white/10 p-4 sm:p-5">
      <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-bold text-white">{value}</div>
      {note ? <div className="mt-1 text-[11px] text-slate-500">{note}</div> : null}
    </div>
  );
}

export default function FastDemoPage() {
  const [sourceMode, setSourceMode] = useState<SourceMode>("camera");
  const [threshold, setThreshold] = useState(0.2);
  const [running, setRunning] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [modelProgress, setModelProgress] = useState(0);
  const [provider, setProvider] = useState("waiting");
  const [activeInput, setActiveInput] = useState(LIVE_INPUT);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [frameMs, setFrameMs] = useState(0);
  const [avgMs, setAvgMs] = useState(0);
  const [frames, setFrames] = useState(0);
  const [warmupRemaining, setWarmupRemaining] = useState(CAMERA_WARMUP_RUNS);
  const [error, setError] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [sourceName, setSourceName] = useState("Rear camera ready");

  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<LibreModel | null>(null);
  const modelInputRef = useRef(0);
  const runtimeRef = useRef<LibreRuntime | null>(null);
  const modelPromiseRef = useRef<Promise<LibreModel> | null>(null);
  const promiseInputRef = useRef(0);
  const loadIdRef = useRef(0);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef("");
  const busyRef = useRef(false);
  const loopRef = useRef<number | null>(null);
  const lastStartRef = useRef(0);
  const timingRef = useRef<number[]>([]);
  const warmupRef = useRef(CAMERA_WARMUP_RUNS);

  const humanCount = useMemo(
    () => detections.filter((d) => d.kind === "Human").length,
    [detections],
  );
  const vehicleCount = detections.length - humanCount;
  const fps = avgMs ? 1000 / avgMs : 0;

  const stopCamera = useCallback(() => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setRunning(false);
  }, []);

  const clearOverlay = useCallback(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const resetStats = useCallback((warmups = 0) => {
    timingRef.current = [];
    warmupRef.current = warmups;
    setWarmupRemaining(warmups);
    setFrameMs(0);
    setAvgMs(0);
    setFrames(0);
    setDetections([]);
    clearOverlay();
  }, [clearOverlay]);

  const ensureModel = useCallback(async (inputSize: number): Promise<LibreModel> => {
    if (modelRef.current && modelInputRef.current === inputSize) return modelRef.current;
    if (modelPromiseRef.current && promiseInputRef.current === inputSize) return modelPromiseRef.current;

    const loadId = ++loadIdRef.current;
    setModelState("loading");
    setModelProgress(2);
    setError("");
    promiseInputRef.current = inputSize;

    const promise = (async () => {
      if (!runtimeRef.current) {
        const moduleUrl: string = LIBRE_RUNTIME_URL;
        runtimeRef.current = (await import(/* @vite-ignore */ moduleUrl)) as unknown as LibreRuntime;
      }

      const previous = modelRef.current;
      modelRef.current = null;
      modelInputRef.current = 0;
      if (previous) {
        try {
          await previous.release();
        } catch {
          // A previous session may already be closed.
        }
      }

      const model = await runtimeRef.current.loadModel(MODEL_SOURCE, {
        device: ["webgpu", "wasm"],
        inputSize,
        modelFamily: "yolox",
        confThres: 0.12,
        iouThres: 0.65,
        maxDet: 120,
        onProgress: (progress) => {
          if (loadId === loadIdRef.current) {
            setModelProgress(Math.max(2, Math.min(99, Math.round(progress * 100))));
          }
        },
      });

      if (loadId !== loadIdRef.current) {
        await model.release();
        throw new Error("Model load was superseded by a newer source mode.");
      }

      modelRef.current = model;
      modelInputRef.current = inputSize;
      setActiveInput(inputSize);
      setProvider(model.provider || "wasm");
      setModelProgress(100);
      setModelState("ready");
      return model;
    })();

    modelPromiseRef.current = promise;
    try {
      return await promise;
    } catch (caught) {
      if (loadId === loadIdRef.current) {
        setModelState("error");
        setError(caught instanceof Error ? caught.message : "YOLOX Nano could not load.");
      }
      throw caught;
    } finally {
      if (loadId === loadIdRef.current) {
        modelPromiseRef.current = null;
        promiseInputRef.current = 0;
      }
    }
  }, []);

  useEffect(() => {
    void ensureModel(LIVE_INPUT).catch(() => {
      // Error state is shown in the UI.
    });
  }, [ensureModel]);

  useEffect(() => {
    return () => {
      loadIdRef.current += 1;
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      void modelRef.current?.release();
    };
  }, []);

  const drawDetections = useCallback((next: Detection[], width: number, height: number) => {
    const canvas = overlayRef.current;
    if (!canvas || !width || !height) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = Math.max(2, width / 520);
    ctx.font = `700 ${Math.max(12, Math.round(width / 92))}px Arial`;
    ctx.textBaseline = "top";

    next.forEach((box) => {
      const human = box.kind === "Human";
      const color = human ? "#5eead4" : "#facc15";
      const x = Math.max(0, box.x1);
      const y = Math.max(0, box.y1);
      const w = Math.max(1, box.x2 - box.x1);
      const h = Math.max(1, box.y2 - box.y1);
      const label = `${box.kind} ${Math.round(box.conf * 100)}%`;
      const metrics = ctx.measureText(label);
      const labelH = Math.max(20, width / 58);
      ctx.strokeStyle = color;
      ctx.fillStyle = color + "12";
      ctx.strokeRect(x, y, w, h);
      ctx.fillRect(x, y, w, h);
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - labelH), metrics.width + 10, labelH);
      ctx.fillStyle = "#021018";
      ctx.fillText(label, x + 5, Math.max(1, y - labelH + 3));
    });
  }, []);

  const analyseOnce = useCallback(async () => {
    if (busyRef.current) return;
    const desiredInput = sourceMode === "camera" ? LIVE_INPUT : IMAGE_INPUT;
    const source = sourceMode === "camera" ? videoRef.current : imageRef.current;
    if (!source) return;
    const width = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
    const height = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
    if (!width || !height) return;

    busyRef.current = true;
    try {
      const model = await ensureModel(desiredInput);
      const started = performance.now();
      const result = await model.predict(source, {
        confThres: Math.max(0.08, threshold),
        iouThres: 0.65,
        maxDet: 120,
      });
      const elapsed = performance.now() - started;

      const next = result.detections
        .map((item): Detection | null => {
          const kind = classToKind(item.classId);
          if (!kind || item.confidence < threshold) return null;
          return {
            x1: item.bbox[0],
            y1: item.bbox[1],
            x2: item.bbox[2],
            y2: item.bbox[3],
            conf: item.confidence,
            classId: item.classId,
            kind,
          };
        })
        .filter((item): item is Detection => item !== null);

      setDetections(next);
      setProvider(model.provider || "wasm");
      drawDetections(next, width, height);

      if (sourceMode === "camera" && warmupRef.current > 0) {
        warmupRef.current -= 1;
        setWarmupRemaining(warmupRef.current);
        return;
      }

      timingRef.current = [...timingRef.current.slice(-29), elapsed];
      const average = timingRef.current.reduce((sum, value) => sum + value, 0) / timingRef.current.length;
      setFrameMs(elapsed);
      setAvgMs(average);
      setFrames((value) => value + 1);
    } catch (caught) {
      setRunning(false);
      setError(caught instanceof Error ? caught.message : "Live inference failed.");
    } finally {
      busyRef.current = false;
    }
  }, [drawDetections, ensureModel, sourceMode, threshold]);

  useEffect(() => {
    if (!running || sourceMode !== "camera") return;
    let cancelled = false;
    const tick = (now: number) => {
      if (cancelled) return;
      if (!busyRef.current && now - lastStartRef.current >= MIN_LOOP_GAP_MS) {
        lastStartRef.current = now;
        void analyseOnce();
      }
      loopRef.current = requestAnimationFrame(tick);
    };
    loopRef.current = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      loopRef.current = null;
    };
  }, [analyseOnce, running, sourceMode]);

  const startCamera = useCallback(async (facing: FacingMode) => {
    stopCamera();
    resetStats(CAMERA_WARMUP_RUNS);
    setSourceMode("camera");
    setError("");
    setSourceName(facing === "environment" ? "Rear camera" : "Front camera / webcam");
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access requires HTTPS and a supported browser.");
      }

      await ensureModel(LIVE_INPUT);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: facing },
          width: { ideal: CAMERA_WIDTH },
          height: { ideal: CAMERA_HEIGHT },
          frameRate: { ideal: 30, max: 30 },
        },
      });
      cameraStreamRef.current = stream;
      if (!videoRef.current) throw new Error("Camera viewport is not ready.");
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraActive(true);
      lastStartRef.current = 0;
      setRunning(true);
    } catch (caught) {
      stopCamera();
      setError(caught instanceof Error ? caught.message : "Camera permission was not granted.");
    }
  }, [ensureModel, resetStats, stopCamera]);

  const handleImage = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    stopCamera();
    resetStats(0);
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setImageUrl(url);
    setSourceMode("image");
    setSourceName(file.name);
    event.target.value = "";
  }, [resetStats, stopCamera]);

  const analyseLoadedImage = useCallback(async () => {
    try {
      await ensureModel(IMAGE_INPUT);
      await analyseOnce();
    } catch {
      // Error state is already surfaced by ensureModel/analyseOnce.
    }
  }, [analyseOnce, ensureModel]);

  const modelBadge = sourceMode === "camera"
    ? `YOLOX Nano · ${LIVE_INPUT} live`
    : `YOLOX Nano · ${IMAGE_INPUT} image`;

  return (
    <main className="min-h-screen bg-[#021018] text-slate-100">
      <header className="border-b border-white/10 bg-[#03151d]/95">
        <div className="mx-auto max-w-[1450px] px-5 py-6 sm:px-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.32em] text-cyan-300">
                <Zap size={14} /> Browser AI Vision Demo
              </div>
              <h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">Real-Time Human + Vehicle Detection</h1>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
                One lightweight YOLOX Nano detector. Live camera uses a speed-focused 320 input; image analysis automatically uses 416 for more detail.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-2 text-cyan-100">{modelBadge}</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-slate-300">{provider.toUpperCase()}</span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-slate-300">Human + 3 vehicle types</span>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1450px] gap-5 px-4 py-5 sm:px-8 lg:grid-cols-[minmax(0,1fr)_390px]">
        <section className="overflow-hidden rounded-[28px] border border-white/10 bg-[#061b24] shadow-2xl shadow-black/20">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
            <div>
              <div className="text-base font-semibold text-white">Live analysis viewport</div>
              <div className="mt-0.5 text-xs text-slate-400">{sourceName}</div>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className={`h-2 w-2 rounded-full ${running ? "bg-emerald-400 animate-pulse" : "bg-slate-600"}`} />
              <span className="text-slate-300">
                {running
                  ? warmupRemaining > 0
                    ? `Warming up · ${warmupRemaining}`
                    : "Analysing"
                  : sourceMode === "image"
                    ? "Image ready"
                    : cameraActive
                      ? "Paused"
                      : "Waiting"}
              </span>
            </div>
          </div>

          <div className="relative aspect-video w-full bg-black">
            {sourceMode === "camera" ? (
              <video ref={videoRef} playsInline muted autoPlay className="h-full w-full object-contain" />
            ) : imageUrl ? (
              <img
                ref={imageRef}
                src={imageUrl}
                alt="Detection source"
                className="h-full w-full object-contain"
                onLoad={() => void analyseLoadedImage()}
              />
            ) : (
              <div className="flex h-full items-center justify-center text-center text-sm text-slate-500">
                <div><ImageIcon className="mx-auto mb-3" /><div>Select a camera or image to begin.</div></div>
              </div>
            )}
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full object-contain" />
          </div>

          <div className="grid grid-cols-2 border-t border-white/10 sm:grid-cols-3 lg:grid-cols-6 [&>*]:border-r [&>*]:border-b sm:[&>*]:border-b-0">
            <Stat label="Detections" value={detections.length} />
            <Stat label="Humans" value={humanCount} />
            <Stat label="Vehicles" value={vehicleCount} />
            <Stat label="Current" value={formatMs(frameMs)} />
            <Stat
              label="Stable avg"
              value={formatMs(avgMs)}
              note={sourceMode === "camera" ? `${frames} timed · warm-up excluded` : `${frames} analysed`}
            />
            <Stat label="Effective" value={fps ? fps.toFixed(1) + " FPS" : "—"} />
          </div>
        </section>

        <aside className="space-y-5">
          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <div className="mb-4 flex items-center gap-2 text-sm font-semibold"><Camera size={18} className="text-cyan-300" /> Choose source</div>
            <div className="grid gap-3">
              <button onClick={() => void startCamera("environment")} className="flex min-h-14 items-center justify-center gap-2 rounded-2xl bg-cyan-300 font-semibold text-[#021018] transition hover:bg-cyan-200">
                <Smartphone size={18} /> Rear camera
              </button>
              <button onClick={() => void startCamera("user")} className="flex min-h-14 items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/5 font-medium text-white transition hover:bg-white/10">
                <Camera size={18} /> Front / webcam
              </button>
              <label className="flex min-h-14 cursor-pointer items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/5 font-medium text-white transition hover:bg-white/10">
                <Upload size={18} /> Upload image
                <input type="file" accept="image/*" className="hidden" onChange={handleImage} />
              </label>
            </div>
            {cameraActive ? (
              <div className="mt-3 grid grid-cols-2 gap-3">
                <button onClick={() => setRunning((value) => !value)} className="flex min-h-12 items-center justify-center gap-2 rounded-xl border border-white/10 bg-[#03161e] text-sm">
                  {running ? <Pause size={17} /> : <Play size={17} />} {running ? "Pause AI" : "Resume AI"}
                </button>
                <button onClick={() => void analyseOnce()} className="flex min-h-12 items-center justify-center gap-2 rounded-xl border border-white/10 bg-[#03161e] text-sm">
                  <RefreshCw size={17} /> Analyse now
                </button>
              </div>
            ) : sourceMode === "image" && imageUrl ? (
              <button onClick={() => void analyseLoadedImage()} className="mt-3 flex min-h-12 w-full items-center justify-center gap-2 rounded-xl border border-cyan-300/25 bg-cyan-300/10 text-sm font-semibold text-cyan-100">
                <Play size={17} /> Analyse image
              </button>
            ) : null}
          </section>

          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <div className="mb-4 flex items-center gap-2 text-sm font-semibold"><Gauge size={18} className="text-cyan-300" /> Detection settings</div>
            <div className="flex items-center justify-between text-sm"><span className="text-slate-300">Confidence</span><strong className="text-cyan-200">{Math.round(threshold * 100)}%</strong></div>
            <input
              type="range"
              min="10"
              max="70"
              step="1"
              value={Math.round(threshold * 100)}
              onChange={(event) => setThreshold(Number(event.target.value) / 100)}
              className="mt-4 w-full accent-cyan-300"
            />
            <div className="mt-5 grid grid-cols-2 gap-2 text-xs">
              {["Human", "Car", "Motorcycle", "Bus/Truck"].map((item) => (
                <div key={item} className="rounded-xl border border-white/10 bg-[#03161e] px-3 py-2.5 text-center text-slate-300">{item}</div>
              ))}
            </div>
          </section>

          <section className="rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5">
            <div className="flex items-center gap-2 text-sm font-semibold"><Cpu size={18} className="text-cyan-300" /> Runtime status</div>
            {modelState === "loading" ? (
              <div className="mt-4">
                <div className="flex items-center gap-2 text-sm text-slate-300"><LoaderCircle className="animate-spin" size={17} /> Loading YOLOX Nano {activeInput}px… {modelProgress}%</div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full bg-cyan-300 transition-all" style={{ width: `${modelProgress}%` }} /></div>
              </div>
            ) : modelState === "ready" ? (
              <div className="mt-4 rounded-xl border border-emerald-400/15 bg-emerald-400/5 p-3 text-xs leading-5 text-emerald-100">
                YOLOX Nano ready at {activeInput}px. {sourceMode === "camera" ? "The first 3 live passes are excluded from timing." : "Image mode uses the higher-detail 416 input."}
              </div>
            ) : (
              <div className="mt-4 rounded-xl border border-red-400/20 bg-red-400/5 p-3 text-xs leading-5 text-red-200">Model failed to load.</div>
            )}
            {error ? <div className="mt-3 rounded-xl border border-red-400/20 bg-red-400/5 p-3 text-xs leading-5 text-red-200">{error}</div> : null}
          </section>
        </aside>
      </div>

      <section className="mx-auto max-w-[1450px] px-4 pb-10 sm:px-8">
        <div className="grid gap-4 rounded-[28px] border border-white/10 bg-[#061b24] p-6 md:grid-cols-3">
          <div><div className="text-xs font-bold uppercase tracking-[.2em] text-cyan-300">Focused</div><p className="mt-2 text-sm leading-6 text-slate-400">Only four public target types are displayed: Human, Car, Motorcycle and Bus/Truck.</p></div>
          <div><div className="text-xs font-bold uppercase tracking-[.2em] text-cyan-300">Fast live path</div><p className="mt-2 text-sm leading-6 text-slate-400">Camera and webcam use YOLOX Nano at 320 with a 1280×720 capture request and one inference pass per analysed frame.</p></div>
          <div><div className="text-xs font-bold uppercase tracking-[.2em] text-cyan-300">Private</div><p className="mt-2 text-sm leading-6 text-slate-400">Camera and image inference runs locally in the browser. Media is not uploaded by this page.</p></div>
        </div>
      </section>
    </main>
  );
}
