"use client";

import {
  Aperture,
  Camera,
  CarFront,
  CheckCircle2,
  Cpu,
  FileImage,
  Gauge,
  Image as ImageIcon,
  Laptop,
  LoaderCircle,
  Maximize2,
  Pause,
  Play,
  RefreshCw,
  ScanSearch,
  ShieldCheck,
  Smartphone,
  Upload,
  UserRound,
  Video,
  XCircle,
  Zap,
} from "lucide-react";
import {
  ChangeEvent,
  ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Backend,
  countAgreement,
  createDetector,
  Detection,
  DetectionFilter,
  Detector,
  detectFrame,
  getPreferredBackend,
  isBackendCompatibilityError,
  isMobileDevice,
  MOBILE_MODEL_KEY,
  MODELS,
  ModelKey,
  ScanMode,
  SourceMode,
} from "./detection";

type FrameSize = { width: number; height: number };
type ModelState = "idle" | "loading" | "ready" | "error";
type AnalysisState = "idle" | "loading-model" | "analysing";
type FacingMode = "environment" | "user";

const EMPTY_SIZE: FrameSize = { width: 0, height: 0 };

function Panel({
  title,
  number,
  icon,
  children,
  className = "",
}: {
  title: string;
  number?: string;
  icon: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={"rounded-[26px] border border-white/10 bg-[#071e27]/90 p-5 " + className}>
      <h2 className="mb-4 flex items-center gap-2 text-[15px] font-semibold text-white">
        <span className="text-cyan-300">{icon}</span>
        {number ? number + " · " : ""}
        {title}
      </h2>
      {children}
    </section>
  );
}

function SourceTab({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "flex min-h-11 flex-1 items-center justify-center gap-2 rounded-xl px-3 text-sm transition " +
        (active
          ? "bg-cyan-300/15 text-cyan-100 ring-1 ring-cyan-300/20"
          : "text-slate-400 hover:bg-white/5 hover:text-white")
      }
    >
      {icon}
      {label}
    </button>
  );
}

function ChoiceButton({
  active,
  title,
  detail,
  badge,
  onClick,
}: {
  active: boolean;
  title: string;
  detail: string;
  badge?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "relative min-h-[82px] rounded-2xl border p-4 text-left transition " +
        (active
          ? "border-cyan-300/60 bg-cyan-300/10 shadow-[0_0_30px_rgba(34,211,238,0.08)]"
          : "border-white/10 bg-[#03161e] hover:border-white/20")
      }
    >
      <span className="block pr-16 text-sm font-semibold text-white">{title}</span>
      <span className="mt-1.5 block text-xs leading-5 text-slate-400">{detail}</span>
      {badge ? (
        <span
          className={
            "absolute right-3 top-3 rounded-full px-2 py-1 text-[9px] font-bold uppercase tracking-wider " +
            (active ? "bg-cyan-300 text-[#021016]" : "bg-white/10 text-slate-300")
          }
        >
          {badge}
        </span>
      ) : null}
    </button>
  );
}

function Stat({
  label,
  value,
  accent = "text-white",
}: {
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className="min-w-0 border-r border-white/10 px-4 py-4 last:border-r-0">
      <div className="truncate text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">
        {label}
      </div>
      <div className={"mt-1 text-xl font-bold " + accent}>{value}</div>
    </div>
  );
}

export default function Home() {
  const [mode, setMode] = useState<SourceMode>("image");
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceName, setSourceName] = useState("No media selected");
  const [modelKey, setModelKey] = useState<ModelKey>("dfine-s");
  const [backend, setBackend] = useState<Backend>("wasm");
  const [scanMode, setScanMode] = useState<ScanMode>("precision");
  const [filter, setFilter] = useState<DetectionFilter>("both");
  const [threshold, setThreshold] = useState(0.2);
  const [analysisFps, setAnalysisFps] = useState(1);
  const [facing, setFacing] = useState<FacingMode>("environment");
  const [cameraActive, setCameraActive] = useState(false);
  const [running, setRunning] = useState(false);
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");
  const [modelProgress, setModelProgress] = useState(0);
  const [modelFile, setModelFile] = useState("");
  const [passProgress, setPassProgress] = useState({
    completed: 0,
    total: 0,
    name: "",
  });
  const [detections, setDetections] = useState<Detection[]>([]);
  const [frameSize, setFrameSize] = useState<FrameSize>(EMPTY_SIZE);
  const [passCount, setPassCount] = useState(0);
  const [fullFrameCount, setFullFrameCount] = useState(0);
  const [inferenceMs, setInferenceMs] = useState(0);
  const [error, setError] = useState("");
  const [fallbackNotice, setFallbackNotice] = useState("");
  const [actualHumans, setActualHumans] = useState<number | null>(null);
  const [actualVehicles, setActualVehicles] = useState<number | null>(null);

  const imageRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef("");
  const detectorRef = useRef<Detector | null>(null);
  const detectorIdentityRef = useRef("");
  const detectorPromiseRef = useRef<Promise<Detector> | null>(null);
  const modelLoadIdRef = useRef(0);
  const analysisLockRef = useRef(false);

  const humanCount = useMemo(
    () => detections.filter((detection) => detection.kind === "human").length,
    [detections],
  );
  const vehicleCount = useMemo(
    () => detections.filter((detection) => detection.kind === "vehicle").length,
    [detections],
  );
  const agreement = countAgreement(
    humanCount,
    vehicleCount,
    actualHumans,
    actualVehicles,
  );
  const tileGain = Math.max(0, detections.length - fullFrameCount);
  const model = MODELS[modelKey];

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const preferred = getPreferredBackend();
      setBackend(preferred);
      if (isMobileDevice()) {
        setModelKey(MOBILE_MODEL_KEY);
        setFallbackNotice(
          "Mobile-compatible RT-DETRv2-R18 selected. It avoids the D-FINE MaxPool operation that this phone's browser runtime cannot execute.",
        );
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  const stopCamera = useCallback(() => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setRunning(false);
  }, []);

  useEffect(() => {
    return () => {
      cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  const resetResults = useCallback(() => {
    setDetections([]);
    setFrameSize(EMPTY_SIZE);
    setPassCount(0);
    setFullFrameCount(0);
    setInferenceMs(0);
    setPassProgress({ completed: 0, total: 0, name: "" });
    setError("");
  }, []);

  const selectMode = useCallback(
    (next: SourceMode) => {
      if (mode === "camera" && next !== "camera") stopCamera();
      setMode(next);
      setSourceUrl("");
      setSourceName("No media selected");
      setRunning(false);
      resetResults();
    },
    [mode, resetResults, stopCamera],
  );

  const changeModel = useCallback((next: ModelKey) => {
    void Promise.resolve(detectorRef.current?.dispose?.()).catch(() => {
      // The previous inference session may already be closed.
    });
    modelLoadIdRef.current += 1;
    detectorRef.current = null;
    detectorIdentityRef.current = "";
    detectorPromiseRef.current = null;
    setModelKey(next);
    setModelState("idle");
    setModelProgress(0);
    setModelFile("");
    setError("");
  }, []);

  const ensureDetector = useCallback(async (
    requestedBackend = backend,
    requestedModelKey = modelKey,
  ): Promise<Detector> => {
    const identity = requestedModelKey + ":" + requestedBackend;
    if (detectorRef.current && detectorIdentityRef.current === identity) {
      return detectorRef.current;
    }
    if (detectorPromiseRef.current && detectorIdentityRef.current === identity) {
      return detectorPromiseRef.current;
    }

    const loadId = ++modelLoadIdRef.current;
    setModelState("loading");
    setAnalysisState("loading-model");
    setModelProgress(0);
    setError("");
    detectorIdentityRef.current = identity;

    const promise = createDetector(requestedModelKey, requestedBackend, (progress) => {
      if (loadId !== modelLoadIdRef.current) return;
      if (typeof progress.progress === "number") {
        const value =
          progress.progress <= 1 ? progress.progress * 100 : progress.progress;
        setModelProgress(Math.max(0, Math.min(100, Math.round(value))));
      }
      if (progress.file) setModelFile(progress.file);
    });
    detectorPromiseRef.current = promise;

    try {
      const detector = await promise;
      if (loadId === modelLoadIdRef.current) {
        detectorRef.current = detector;
        detectorIdentityRef.current = identity;
        detectorPromiseRef.current = null;
        setModelState("ready");
        setModelProgress(100);
      }
      return detector;
    } catch (caught) {
      if (loadId === modelLoadIdRef.current) {
        detectorPromiseRef.current = null;
        detectorIdentityRef.current = "";
        setModelState("error");
      }
      const message =
        caught instanceof Error ? caught.message : "The AI model could not load.";
      throw new Error(
        message +
          (requestedModelKey === MOBILE_MODEL_KEY
            ? " Check the internet connection, then reload the page."
            : " This browser may not support the D-FINE graph; try RT-DETRv2-R18."),
      );
    }
  }, [backend, modelKey]);

  const captureFrame = useCallback((): HTMLCanvasElement => {
    let source: CanvasImageSource;
    let width = 0;
    let height = 0;

    if (mode === "image") {
      const image = imageRef.current;
      if (!image?.complete || !image.naturalWidth) {
        throw new Error("Wait for the image to finish loading.");
      }
      source = image;
      width = image.naturalWidth;
      height = image.naturalHeight;
    } else {
      const video = videoRef.current;
      if (!video?.videoWidth) {
        throw new Error("Wait for the video or camera frame to become ready.");
      }
      source = video;
      width = video.videoWidth;
      height = video.videoHeight;
    }

    const maxSide = modelKey === "dfine-s" ? 2560 : 1920;
    const scale = Math.min(1, maxSide / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("This browser could not capture the frame.");
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    setFrameSize({ width: canvas.width, height: canvas.height });
    return canvas;
  }, [mode, modelKey]);

  const analyseOnce = useCallback(async () => {
    if (analysisLockRef.current) return;
    analysisLockRef.current = true;
    setError("");
    setAnalysisState("analysing");
    const startedAt = performance.now();

    try {
      const frame = captureFrame();
      const runDetection = async (
        requestedModelKey: ModelKey,
        requestedBackend: Backend,
      ) => {
        const activeDetector = await ensureDetector(
          requestedBackend,
          requestedModelKey,
        );
        setAnalysisState("analysing");
        return detectFrame({
          detector: activeDetector,
          frame,
          mode: scanMode,
          filter,
          threshold,
          onPass: (completed, total, name) =>
            setPassProgress({ completed, total, name }),
        });
      };

      let result: Awaited<ReturnType<typeof detectFrame>>;
      try {
        result = await runDetection(modelKey, backend);
      } catch (initialError) {
        if (
          modelKey === MOBILE_MODEL_KEY ||
          !isBackendCompatibilityError(initialError)
        ) {
          throw initialError;
        }

        setFallbackNotice(
          "D-FINE is not supported by this browser runtime. This frame is being retried with RT-DETRv2-R18 on mobile-safe WASM.",
        );
        try {
          await detectorRef.current?.dispose?.();
        } catch {
          // A failed inference session may already be closed.
        }
        modelLoadIdRef.current += 1;
        detectorRef.current = null;
        detectorIdentityRef.current = "";
        detectorPromiseRef.current = null;
        setBackend("wasm");
        setModelKey(MOBILE_MODEL_KEY);
        setModelState("idle");
        setModelProgress(0);
        setModelFile("");
        result = await runDetection(MOBILE_MODEL_KEY, "wasm");
      }
      setDetections(result.detections);
      setPassCount(result.passCount);
      setFullFrameCount(result.fullFrameCount);
      setInferenceMs(Math.round(performance.now() - startedAt));
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Analysis did not complete.",
      );
    } finally {
      analysisLockRef.current = false;
      setAnalysisState("idle");
    }
  }, [backend, captureFrame, ensureDetector, filter, modelKey, scanMode, threshold]);

  useEffect(() => {
    if (!running || mode === "image") return;
    let cancelled = false;
    let timer = 0;

    const loop = async () => {
      if (cancelled) return;
      await analyseOnce();
      if (!cancelled) {
        timer = window.setTimeout(loop, Math.round(1000 / analysisFps));
      }
    };
    void loop();

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [analyseOnce, analysisFps, mode, running]);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    if (!frameSize.width || !frameSize.height) {
      canvas.width = 1;
      canvas.height = 1;
      return;
    }
    canvas.width = frameSize.width;
    canvas.height = frameSize.height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.lineWidth = Math.max(3, canvas.width / 420);
    context.font =
      "700 " + Math.max(15, Math.round(canvas.width / 74)) + "px Arial";
    context.textBaseline = "bottom";

    detections.forEach((detection) => {
      const { xmin, ymin, xmax, ymax } = detection.box;
      const color = detection.kind === "human" ? "#42f5d0" : "#ffd45c";
      const label =
        (detection.kind === "human" ? "Human" : detection.label) +
        " #" +
        detection.id +
        " · " +
        Math.round(detection.score * 100) +
        "%";
      const metrics = context.measureText(label);
      const labelHeight = Math.max(24, canvas.width / 46);
      const labelY = Math.max(labelHeight, ymin);
      context.strokeStyle = color;
      context.fillStyle = color + "22";
      context.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
      context.fillRect(xmin, ymin, xmax - xmin, ymax - ymin);
      context.fillStyle = color;
      context.fillRect(
        xmin,
        labelY - labelHeight,
        metrics.width + 18,
        labelHeight,
      );
      context.fillStyle = "#03202a";
      context.fillText(label, xmin + 9, labelY - 4);
      context.beginPath();
      context.arc(
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        Math.max(4, canvas.width / 250),
        0,
        Math.PI * 2,
      );
      context.fillStyle = color;
      context.fill();
    });
  }, [detections, frameSize]);

  const handleFile = useCallback(
    (event: ChangeEvent<HTMLInputElement>, nextMode: "image" | "video") => {
      const file = event.target.files?.[0];
      if (!file) return;
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      setMode(nextMode);
      setSourceUrl(url);
      setSourceName(file.name);
      setRunning(nextMode === "video");
      resetResults();
      event.target.value = "";
    },
    [resetResults],
  );

  const startCamera = useCallback(
    async (requestedFacing: FacingMode = facing) => {
      stopCamera();
      setMode("camera");
      setFacing(requestedFacing);
      resetResults();
      setError("");
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error(
            "Camera access needs HTTPS and a supported mobile or desktop browser.",
          );
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: requestedFacing },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
        });
        cameraStreamRef.current = stream;
        setCameraActive(true);
        setSourceName(
          requestedFacing === "environment"
            ? "Live rear camera"
            : "Live front camera",
        );
        await new Promise<void>((resolve) =>
          window.requestAnimationFrame(() => resolve()),
        );
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setRunning(true);
      } catch (caught) {
        stopCamera();
        setError(
          caught instanceof Error
            ? caught.message
            : "Camera permission was not granted.",
        );
      }
    },
    [facing, resetResults, stopCamera],
  );

  const analysisLabel =
    analysisState === "loading-model"
      ? "Loading " + model.name
      : analysisState === "analysing"
        ? passProgress.total
          ? "Scanning " +
            Math.min(passProgress.completed + 1, passProgress.total) +
            "/" +
            passProgress.total
          : "Preparing frame"
        : running
          ? analysisFps + " FPS target"
          : "Paused";

  return (
    <main className="min-h-screen overflow-x-hidden bg-[#021018] text-slate-100">
      <header className="border-b border-white/10 bg-[#03151d]/95">
        <div className="mx-auto flex max-w-[1580px] flex-col gap-4 px-5 py-5 sm:px-8 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <div className="grid h-14 w-14 shrink-0 place-items-center rounded-[20px] border border-cyan-300/35 bg-cyan-300/5 text-cyan-200">
              <Aperture size={30} strokeWidth={1.8} />
            </div>
            <div>
              <div className="mb-1 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.34em] text-cyan-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_12px_#34d399]" />
                Small-target CCTV AI
              </div>
              <h1 className="text-xl font-bold tracking-tight sm:text-2xl">
                D-FINE Tiny-Target Accuracy Lab
              </h1>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <span className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-slate-300">
              <ShieldCheck size={13} />
              Media stays in your browser
            </span>
            <span className="flex items-center gap-1.5 rounded-full border border-emerald-300/25 bg-emerald-400/10 px-3 py-1.5 text-emerald-200">
              <CheckCircle2 size={13} />
              {backend.toUpperCase()} · {model.name}
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1580px] gap-5 px-4 py-6 sm:px-8 xl:grid-cols-[minmax(0,1fr)_420px]">
        <section className="min-w-0 overflow-hidden rounded-[28px] border border-white/10 bg-[#061d26]">
          <div className="flex flex-col gap-3 border-b border-white/10 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 items-center gap-3">
              <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-cyan-300/10 text-cyan-200">
                <ScanSearch size={20} />
              </div>
              <div className="min-w-0">
                <h2 className="text-sm font-semibold text-white">Analysis viewport</h2>
                <p className="truncate text-xs text-sky-200/70">{sourceName}</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-sky-200/75">
              <span
                className={
                  "h-2 w-2 rounded-full " +
                  (analysisState !== "idle"
                    ? "animate-pulse bg-amber-300"
                    : running
                      ? "bg-emerald-400"
                      : "bg-slate-500")
                }
              />
              {analysisLabel}
              <span className="text-slate-600">·</span>
              {scanMode === "precision" ? "5-pass precision" : "1-pass fast"}
            </div>
          </div>

          <div className="relative flex min-h-[340px] items-center justify-center overflow-hidden bg-black lg:min-h-[540px]">
            {mode === "image" && sourceUrl ? (
              <div className="relative w-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  ref={imageRef}
                  src={sourceUrl}
                  alt="Selected frame for object detection"
                  className="block h-auto w-full"
                  onLoad={() => void analyseOnce()}
                />
                <canvas
                  ref={overlayRef}
                  className="pointer-events-none absolute inset-0 h-full w-full"
                />
              </div>
            ) : mode === "video" && sourceUrl ? (
              <div className="relative w-full">
                <video
                  ref={videoRef}
                  src={sourceUrl}
                  controls
                  playsInline
                  className="block h-auto w-full"
                  onLoadedData={() => {
                    setError("");
                    setRunning(true);
                  }}
                />
                <canvas
                  ref={overlayRef}
                  className="pointer-events-none absolute inset-0 h-full w-full"
                />
              </div>
            ) : mode === "camera" ? (
              <div className="relative w-full">
                <video
                  ref={videoRef}
                  muted
                  playsInline
                  autoPlay
                  className={
                    "block h-auto min-h-[340px] w-full object-contain lg:min-h-[540px] " +
                    (cameraActive ? "" : "opacity-0")
                  }
                />
                <canvas
                  ref={overlayRef}
                  className="pointer-events-none absolute inset-0 h-full w-full"
                />
                {!cameraActive ? (
                  <div className="absolute inset-0 grid place-items-center p-8 text-center">
                    <div>
                      <Camera
                        size={54}
                        strokeWidth={1.3}
                        className="mx-auto mb-4 text-cyan-300/60"
                      />
                      <p className="font-semibold text-white">Camera is not started</p>
                      <p className="mt-2 text-sm text-slate-400">
                        Choose rear or front camera in the source controls.
                      </p>
                    </div>
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="max-w-md px-8 py-20 text-center">
                <div className="mx-auto mb-5 grid h-20 w-20 place-items-center rounded-[28px] border border-dashed border-cyan-300/30 bg-cyan-300/5 text-cyan-200/70">
                  {mode === "image" ? (
                    <FileImage size={38} strokeWidth={1.4} />
                  ) : (
                    <Video size={38} strokeWidth={1.4} />
                  )}
                </div>
                <h3 className="text-lg font-semibold text-white">
                  Add a {mode === "image" ? "test image" : "test video"}
                </h3>
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  Use CCTV footage with distant people or vehicles to compare
                  the full-frame result against the precision tile scan.
                </p>
              </div>
            )}

            {analysisState !== "idle" ? (
              <div className="absolute left-4 top-4 flex items-center gap-3 rounded-2xl border border-cyan-200/20 bg-[#03151ded] px-4 py-3 shadow-2xl backdrop-blur">
                <LoaderCircle className="animate-spin text-cyan-300" size={20} />
                <div>
                  <div className="text-xs font-semibold text-white">{analysisLabel}</div>
                  <div className="mt-0.5 text-[10px] text-slate-400">
                    {analysisState === "loading-model"
                      ? modelFile || "Downloading model files once"
                      : passProgress.name || "Preparing pixels"}
                  </div>
                </div>
              </div>
            ) : null}
          </div>

          {fallbackNotice ? (
            <div className="flex items-start gap-2 border-t border-cyan-300/20 bg-cyan-300/8 px-5 py-3 text-xs leading-5 text-cyan-100">
              <Cpu size={16} className="mt-0.5 shrink-0 text-cyan-300" />
              {fallbackNotice}
            </div>
          ) : null}

          {error ? (
            <div className="flex items-start gap-2 border-t border-red-300/20 bg-red-400/10 px-5 py-3 text-xs leading-5 text-red-200">
              <XCircle size={16} className="mt-0.5 shrink-0" />
              {error}
            </div>
          ) : null}

          <div className="grid grid-cols-3 border-t border-white/10 sm:grid-cols-6">
            <Stat label="Detections" value={detections.length} accent="text-cyan-200" />
            <Stat label="Humans" value={humanCount} accent="text-emerald-300" />
            <Stat label="Vehicles" value={vehicleCount} accent="text-amber-300" />
            <Stat label="Passes" value={passCount || "—"} />
            <Stat label="Tile gain" value={"+" + tileGain} accent="text-cyan-300" />
            <Stat
              label="Inference"
              value={inferenceMs ? inferenceMs + " ms" : "—"}
              accent="text-yellow-200"
            />
          </div>
        </section>

        <aside className="flex min-w-0 flex-col gap-5">
          <Panel title="Select the source" number="1" icon={<Camera size={18} />}>
            <div className="mb-3 flex rounded-2xl bg-[#03141b] p-1">
              <SourceTab
                active={mode === "image"}
                icon={<ImageIcon size={17} />}
                label="Image"
                onClick={() => selectMode("image")}
              />
              <SourceTab
                active={mode === "video"}
                icon={<Video size={17} />}
                label="Video"
                onClick={() => selectMode("video")}
              />
              <SourceTab
                active={mode === "camera"}
                icon={<Camera size={17} />}
                label="Camera"
                onClick={() => selectMode("camera")}
              />
            </div>

            {mode === "image" ? (
              <label className="flex min-h-12 cursor-pointer items-center justify-center gap-2 rounded-xl bg-cyan-300 font-semibold text-[#021118] transition hover:bg-cyan-200">
                <Upload size={18} />
                Choose an image
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(event) => handleFile(event, "image")}
                />
              </label>
            ) : mode === "video" ? (
              <label className="flex min-h-12 cursor-pointer items-center justify-center gap-2 rounded-xl bg-cyan-300 font-semibold text-[#021118] transition hover:bg-cyan-200">
                <Upload size={18} />
                Choose a video
                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(event) => handleFile(event, "video")}
                />
              </label>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => void startCamera("environment")}
                  className={
                    "flex min-h-12 items-center justify-center gap-2 rounded-xl border text-sm font-semibold " +
                    (cameraActive && facing === "environment"
                      ? "border-cyan-300 bg-cyan-300 text-[#021118]"
                      : "border-white/10 bg-white/5 text-white hover:border-cyan-300/40")
                  }
                >
                  <Smartphone size={18} />
                  Rear camera
                </button>
                <button
                  type="button"
                  onClick={() => void startCamera("user")}
                  className={
                    "flex min-h-12 items-center justify-center gap-2 rounded-xl border text-sm font-semibold " +
                    (cameraActive && facing === "user"
                      ? "border-cyan-300 bg-cyan-300 text-[#021118]"
                      : "border-white/10 bg-white/5 text-white hover:border-cyan-300/40")
                  }
                >
                  <Laptop size={18} />
                  Front/webcam
                </button>
              </div>
            )}

            {(sourceUrl || cameraActive) && (
              <div className="mt-3 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setRunning((value) => !value)}
                  className="flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm text-white hover:bg-white/10"
                >
                  {running ? <Pause size={17} /> : <Play size={17} />}
                  {running ? "Pause AI" : "Run AI"}
                </button>
                <button
                  type="button"
                  disabled={analysisState !== "idle"}
                  onClick={() => void analyseOnce()}
                  className="flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 text-sm text-white hover:bg-white/10 disabled:opacity-50"
                >
                  <RefreshCw size={16} />
                  Analyse now
                </button>
              </div>
            )}
            {mode === "camera" ? (
              <p className="mt-3 text-[11px] leading-5 text-slate-500">
                Works on Android, iPhone, laptop and desktop. Browser permission
                and HTTPS are required.
              </p>
            ) : null}
          </Panel>

          <Panel title="Choose the AI" number="2" icon={<Cpu size={18} />}>
            <div className="grid gap-2">
              {(Object.keys(MODELS) as ModelKey[]).map((key) => (
                <ChoiceButton
                  key={key}
                  active={modelKey === key}
                  title={MODELS[key].name}
                  detail={MODELS[key].note}
                  badge={MODELS[key].badge}
                  onClick={() => changeModel(key)}
                />
              ))}
            </div>
            <p className="mt-3 text-[11px] leading-5 text-slate-500">
              D-FINE-S is the desktop accuracy test. Phones automatically use
              RT-DETRv2-R18 because its deployment graph is compatible with the
              browser&apos;s mobile WASM engine.
            </p>
          </Panel>

          <Panel title="Detection filter" number="3" icon={<UserRound size={18} />}>
            <div className="mb-4 grid grid-cols-3 rounded-2xl bg-[#03141b] p-1">
              {(["both", "human", "vehicle"] as DetectionFilter[]).map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setFilter(value)}
                  className={
                    "min-h-10 rounded-xl text-sm capitalize transition " +
                    (filter === value
                      ? "bg-lime-300/15 text-lime-100 ring-1 ring-lime-300/20"
                      : "text-slate-400 hover:text-white")
                  }
                >
                  {value}
                </button>
              ))}
            </div>
            <label className="block rounded-2xl border border-white/10 bg-[#03161e] p-4">
              <span className="flex items-center justify-between text-xs text-sky-100/80">
                Confidence threshold
                <strong className="text-cyan-200">{Math.round(threshold * 100)}%</strong>
              </span>
              <input
                type="range"
                min="5"
                max="80"
                step="1"
                value={Math.round(threshold * 100)}
                onChange={(event) => setThreshold(Number(event.target.value) / 100)}
                className="mt-4 w-full accent-cyan-300"
              />
            </label>
            <p className="mt-3 text-[11px] leading-5 text-slate-500">
              Start at 20%. Lower values expose weak small targets but also
              increase false detections.
            </p>
          </Panel>

          <Panel title="Small-target scan" number="4" icon={<Maximize2 size={18} />}>
            <div className="grid grid-cols-2 gap-2">
              <ChoiceButton
                active={scanMode === "fast"}
                title="Fast · 1 pass"
                detail="Whole frame once. Best for a live phone camera."
                onClick={() => setScanMode("fast")}
              />
              <ChoiceButton
                active={scanMode === "precision"}
                title="Precision · 5 passes"
                detail="Full frame plus four overlapping enlarged tiles."
                badge="Small pixels"
                onClick={() => setScanMode("precision")}
              />
            </div>
            <label className="mt-4 block rounded-2xl border border-white/10 bg-[#03161e] p-4">
              <span className="flex items-center justify-between text-xs text-sky-100/80">
                Frames analysed per second
                <strong className="text-cyan-200">{analysisFps} FPS</strong>
              </span>
              <input
                type="range"
                min="1"
                max="3"
                step="1"
                value={analysisFps}
                onChange={(event) => setAnalysisFps(Number(event.target.value))}
                className="mt-4 w-full accent-cyan-300"
              />
            </label>
          </Panel>
        </aside>
      </div>
      <div className="mx-auto grid max-w-[1580px] gap-5 px-4 pb-6 sm:px-8 xl:grid-cols-[minmax(0,1fr)_420px]">
        <div className="flex min-w-0 flex-col gap-5">
          <Panel
            title="What the AI is doing in the background"
            icon={<Zap size={18} />}
          >
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              {[
                {
                  step: "01",
                  title: "Capture",
                  text: "Read the current image, video or camera pixels.",
                  active: Boolean(sourceUrl || cameraActive),
                },
                {
                  step: "02",
                  title: "Crop & enlarge",
                  text:
                    scanMode === "precision"
                      ? "Create four overlapping crops so tiny targets occupy more model pixels."
                      : "Keep the complete frame for the fastest result.",
                  active: analysisState === "analysing",
                },
                {
                  step: "03",
                  title: model.name + " inference",
                  text: "A transformer predicts classes, confidence and box coordinates.",
                  active: modelState === "ready",
                },
                {
                  step: "04",
                  title: "Class filter",
                  text: "Keep only person, car, truck, bus, bicycle and motorcycle.",
                  active: detections.length > 0,
                },
                {
                  step: "05",
                  title: "Tile merge",
                  text: "Remove repeated boxes created where precision tiles overlap.",
                  active: passCount > 0,
                },
              ].map((stage) => (
                <div
                  key={stage.step}
                  className={
                    "relative rounded-2xl border p-4 " +
                    (stage.active
                      ? "border-cyan-300/30 bg-cyan-300/8"
                      : "border-white/8 bg-[#03161e]")
                  }
                >
                  <div className="mb-6 flex items-center justify-between">
                    <span className="text-[10px] font-bold tracking-[0.2em] text-cyan-300">
                      STEP {stage.step}
                    </span>
                    <span
                      className={
                        "h-2 w-2 rounded-full " +
                        (stage.active ? "bg-emerald-400" : "bg-slate-700")
                      }
                    />
                  </div>
                  <h3 className="text-sm font-semibold text-white">{stage.title}</h3>
                  <p className="mt-2 text-[11px] leading-5 text-slate-400">{stage.text}</p>
                </div>
              ))}
            </div>
            <div className="mt-4 flex flex-col gap-2 rounded-2xl border border-cyan-300/15 bg-cyan-300/5 px-4 py-3 text-[11px] leading-5 text-sky-100/70 sm:flex-row sm:items-center sm:justify-between">
              <span>
                Current path: {model.name} → {backend.toUpperCase()} →{" "}
                {scanMode === "precision" ? "5 model passes" : "1 model pass"} →{" "}
                {filter} results
              </span>
              <span className="shrink-0 text-cyan-200">
                Input frame:{" "}
                {frameSize.width
                  ? frameSize.width + " × " + frameSize.height
                  : "waiting"}
              </span>
            </div>
          </Panel>

          <Panel title="Live detection output" icon={<ScanSearch size={18} />}>
            {detections.length ? (
              <div className="grid max-h-[390px] gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
                {detections.map((detection) => (
                  <div
                    key={detection.id}
                    className="flex items-center gap-3 rounded-2xl border border-white/8 bg-[#03161e] p-3"
                  >
                    <div
                      className={
                        "grid h-10 w-10 shrink-0 place-items-center rounded-xl " +
                        (detection.kind === "human"
                          ? "bg-emerald-300/10 text-emerald-300"
                          : "bg-amber-300/10 text-amber-300")
                      }
                    >
                      {detection.kind === "human" ? (
                        <UserRound size={20} />
                      ) : (
                        <CarFront size={21} />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <strong className="truncate text-xs capitalize text-white">
                          {detection.kind === "human" ? "Human" : detection.label} #
                          {detection.id}
                        </strong>
                        <span className="text-xs font-bold text-cyan-200">
                          {Math.round(detection.score * 100)}%
                        </span>
                      </div>
                      <div className="mt-1 truncate font-mono text-[9px] text-slate-500">
                        x {Math.round(detection.box.xmin)}, y{" "}
                        {Math.round(detection.box.ymin)}, w{" "}
                        {Math.round(detection.box.xmax - detection.box.xmin)}, h{" "}
                        {Math.round(detection.box.ymax - detection.box.ymin)} ·{" "}
                        {detection.origin}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-white/10 bg-[#03161e] px-5 py-10 text-center">
                <ScanSearch className="mx-auto text-slate-600" size={30} />
                <p className="mt-3 text-sm font-semibold text-slate-300">
                  Detection records will appear here
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Each record shows the class, confidence, box pixels and scan origin.
                </p>
              </div>
            )}
          </Panel>
        </div>

        <aside className="flex min-w-0 flex-col gap-5">
          <Panel title="Model download & engine" icon={<Gauge size={18} />}>
            <div className="rounded-2xl border border-white/10 bg-[#03161e] p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-white">{model.name}</div>
                  <div className="mt-1 text-[10px] text-slate-500">
                    {backend.toUpperCase()} browser inference
                  </div>
                </div>
                <span
                  className={
                    "rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider " +
                    (modelState === "ready"
                      ? "bg-emerald-400/15 text-emerald-300"
                      : modelState === "error"
                        ? "bg-red-400/15 text-red-300"
                        : "bg-cyan-300/10 text-cyan-200")
                  }
                >
                  {modelState}
                </span>
              </div>
              <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-cyan-300 transition-all duration-300"
                  style={{ width: modelProgress + "%" }}
                />
              </div>
              <div className="mt-2 flex justify-between text-[9px] text-slate-500">
                <span className="max-w-[260px] truncate">
                  {modelFile || "Model loads only when analysis starts"}
                </span>
                <span>{modelProgress}%</span>
              </div>
            </div>
            {modelState !== "ready" ? (
              <button
                type="button"
                disabled={modelState === "loading"}
                onClick={() =>
                  void ensureDetector().catch((caught) =>
                    setError(
                      caught instanceof Error
                        ? caught.message
                        : "The AI model could not load.",
                    ),
                  )
                }
                className="mt-3 flex min-h-11 w-full items-center justify-center gap-2 rounded-xl bg-cyan-300 text-sm font-semibold text-[#021118] hover:bg-cyan-200 disabled:cursor-wait disabled:opacity-60"
              >
                {modelState === "loading" ? (
                  <LoaderCircle size={17} className="animate-spin" />
                ) : (
                  <Cpu size={17} />
                )}
                {modelState === "loading" ? "Downloading model" : "Load selected AI"}
              </button>
            ) : null}
          </Panel>

          <Panel title="Check this frame’s count" icon={<Gauge size={18} />}>
            <p className="mb-4 text-[11px] leading-5 text-slate-400">
              Count the real targets yourself, then enter the values. This gives
              an immediate count-agreement check—not formal mAP.
            </p>
            <div className="grid grid-cols-2 gap-3">
              <label className="rounded-2xl border border-white/10 bg-[#03161e] p-3">
                <span className="flex items-center gap-2 text-[11px] text-emerald-200">
                  <UserRound size={14} />
                  Actual humans
                </span>
                <input
                  type="number"
                  min="0"
                  inputMode="numeric"
                  value={actualHumans ?? ""}
                  placeholder="—"
                  onChange={(event) =>
                    setActualHumans(
                      event.target.value === ""
                        ? null
                        : Math.max(0, Number(event.target.value)),
                    )
                  }
                  className="mt-2 w-full bg-transparent text-2xl font-bold text-white outline-none"
                />
              </label>
              <label className="rounded-2xl border border-white/10 bg-[#03161e] p-3">
                <span className="flex items-center gap-2 text-[11px] text-amber-200">
                  <CarFront size={14} />
                  Actual vehicles
                </span>
                <input
                  type="number"
                  min="0"
                  inputMode="numeric"
                  value={actualVehicles ?? ""}
                  placeholder="—"
                  onChange={(event) =>
                    setActualVehicles(
                      event.target.value === ""
                        ? null
                        : Math.max(0, Number(event.target.value)),
                    )
                  }
                  className="mt-2 w-full bg-transparent text-2xl font-bold text-white outline-none"
                />
              </label>
            </div>
            <div className="mt-3 rounded-2xl border border-cyan-300/20 bg-cyan-300/8 p-4">
              <div className="flex items-end justify-between">
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                    Count agreement
                  </div>
                  <div className="mt-1 text-3xl font-bold text-cyan-200">
                    {agreement === null ? "—" : agreement + "%"}
                  </div>
                </div>
                <div className="text-right text-[10px] leading-5 text-slate-400">
                  <div>AI humans: {humanCount}</div>
                  <div>AI vehicles: {vehicleCount}</div>
                </div>
              </div>
            </div>
          </Panel>

          <Panel title="What improves tiny targets" icon={<Aperture size={18} />}>
            <div className="space-y-3 text-[11px] leading-5 text-slate-400">
              <div className="flex gap-3">
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-cyan-300/10 font-bold text-cyan-200">
                  1
                </span>
                <p>
                  <strong className="text-slate-200">D-FINE-S</strong> remains
                  the desktop accuracy choice. <strong className="text-slate-200">RT-DETRv2-R18</strong>{" "}
                  provides the compatible mobile path when D-FINE cannot execute.
                </p>
              </div>
              <div className="flex gap-3">
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-cyan-300/10 font-bold text-cyan-200">
                  2
                </span>
                <p>
                  <strong className="text-slate-200">Precision tiles</strong>{" "}
                  enlarge distant regions before inference, then map boxes back
                  to the original frame.
                </p>
              </div>
              <div className="flex gap-3">
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-cyan-300/10 font-bold text-cyan-200">
                  3
                </span>
                <p>
                  <strong className="text-slate-200">A CCTV fine-tune</strong>{" "}
                  is the next step for night scenes and very small targets. The
                  current weights are pretrained on general COCO images.
                </p>
              </div>
            </div>
            <div className="mt-4 rounded-xl border border-amber-300/15 bg-amber-300/5 p-3 text-[10px] leading-5 text-amber-100/70">
              If a person is only a few indistinct pixels, no detector can
              recreate missing visual information. Better source resolution,
              camera placement and domain training still matter.
            </div>
          </Panel>
        </aside>
      </div>

      <footer className="mx-auto max-w-[1580px] px-5 pb-10 pt-2 text-center text-[11px] leading-5 text-slate-500 sm:px-8">
        Educational browser test using pretrained COCO weights. Count agreement
        helps compare frames, but formal detector accuracy requires annotated
        bounding boxes and mAP evaluation.
      </footer>
    </main>
  );
}
