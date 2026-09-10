import type { LIBREYOLO } from "libreyolo-web";
import { type ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import { COCO_DISPLAY_LABELS, ROAD_VEHICLE_CLASS_IDS } from "./coco";

type FacingMode = "environment" | "user";
type SourceMode = "camera" | "image";
type DisplayMode = "focus" | "all";
type DetectionGroup = "Human" | "Vehicle" | "Other";

type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  label: string;
  group: DetectionGroup;
};

type DetectionCounts = {
  total: number;
  humans: number;
  vehicles: number;
  other: number;
};

type TimingStats = {
  current: number;
  average: number;
  frames: number;
};

const MODEL_NAME = "LibreYOLOXs" as const;
const MODEL_INPUT = 640;
const DEFAULT_CONFIDENCE = 0.6;
const NMS_IOU_THRESHOLD = 0.65;
const MAX_DETECTIONS = 120;
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 360;
const CAMERA_WARMUP_RUNS = 3;
const TIMING_WINDOW = 30;
const EMPTY_COUNTS: DetectionCounts = { total: 0, humans: 0, vehicles: 0, other: 0 };
const EMPTY_TIMING: TimingStats = { current: 0, average: 0, frames: 0 };

function mapDetection(classId: number, mode: DisplayMode): { label: string; group: DetectionGroup } | null {
  if (classId < 0 || classId >= COCO_DISPLAY_LABELS.length) return null;

  if (mode === "focus") {
    if (classId === 0) return { label: "Human", group: "Human" };
    if (ROAD_VEHICLE_CLASS_IDS.has(classId)) return { label: "Vehicle", group: "Vehicle" };
    return null;
  }

  return {
    label: COCO_DISPLAY_LABELS[classId],
    group: classId === 0 ? "Human" : ROAD_VEHICLE_CLASS_IDS.has(classId) ? "Vehicle" : "Other",
  };
}

function formatMs(value: number) {
  if (!value || !Number.isFinite(value)) return "—";
  return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function waitForPaint() {
  return new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

async function requestCameraStream(facingMode: FacingMode) {
  const video: MediaTrackConstraints = {
    facingMode: { ideal: facingMode },
    width: { ideal: CAMERA_WIDTH },
    height: { ideal: CAMERA_HEIGHT },
    frameRate: { ideal: 30, max: 30 },
  };

  try {
    return await navigator.mediaDevices.getUserMedia({ audio: false, video });
  } catch (firstError) {
    try {
      return await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: CAMERA_WIDTH },
          height: { ideal: CAMERA_HEIGHT },
          frameRate: { ideal: 30, max: 30 },
        },
      });
    } catch {
      throw firstError;
    }
  }
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

export default function App() {
  const [sourceMode, setSourceMode] = useState<SourceMode>("camera");
  const [displayMode, setDisplayMode] = useState<DisplayMode>("all");
  const [selectedCamera, setSelectedCamera] = useState<FacingMode | null>(null);
  const [cameraAspect, setCameraAspect] = useState("16 / 9");
  const [threshold, setThreshold] = useState(DEFAULT_CONFIDENCE);
  const [running, setRunning] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [modelProgress, setModelProgress] = useState(0);
  const [provider, setProvider] = useState("waiting");
  const [counts, setCounts] = useState<DetectionCounts>(EMPTY_COUNTS);
  const [timing, setTiming] = useState<TimingStats>(EMPTY_TIMING);
  const [warmupRemaining, setWarmupRemaining] = useState(CAMERA_WARMUP_RUNS);
  const [error, setError] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [sourceName, setSourceName] = useState("Choose a source to begin");

  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<LIBREYOLO | null>(null);
  const modelPromiseRef = useRef<Promise<LIBREYOLO> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef("");
  const busyRef = useRef(false);
  const frameRef = useRef<number | null>(null);
  const timingRef = useRef<number[]>([]);
  const warmupRef = useRef(CAMERA_WARMUP_RUNS);
  const analysisEpochRef = useRef(0);

  const effectiveFps = timing.average ? 1000 / timing.average : 0;

  const clearOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const resetAnalysis = useCallback(
    (warmups = 0) => {
      analysisEpochRef.current += 1;
      timingRef.current = [];
      warmupRef.current = warmups;
      setWarmupRemaining(warmups);
      setCounts(EMPTY_COUNTS);
      setTiming(EMPTY_TIMING);
      clearOverlay();
    },
    [clearOverlay],
  );

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setRunning(false);
  }, []);

  const clearImage = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = "";
    }
    setImageUrl("");
  }, []);

  const syncCameraAspect = useCallback(() => {
    const video = videoRef.current;
    if (!video?.videoWidth || !video.videoHeight) return;
    setCameraAspect(`${video.videoWidth} / ${video.videoHeight}`);
  }, []);

  const ensureModel = useCallback(async () => {
    if (modelRef.current) return modelRef.current;
    if (modelPromiseRef.current) return modelPromiseRef.current;

    setModelState("loading");
    setModelProgress(2);
    setError("");

    const promise = import("libreyolo-web").then(({ loadModel }) =>
      loadModel(MODEL_NAME, {
        device: ["webgpu", "wasm"],
        modelFamily: "yolox",
        confThres: DEFAULT_CONFIDENCE,
        iouThres: NMS_IOU_THRESHOLD,
        maxDet: MAX_DETECTIONS,
        onProgress: (progress) => {
          setModelProgress(Math.max(2, Math.min(99, Math.round(progress * 100))));
        },
      }),
    );

    modelPromiseRef.current = promise;

    try {
      const model = await promise;
      modelRef.current = model;
      setProvider(model.provider || "wasm");
      setModelProgress(100);
      setModelState("ready");
      return model;
    } catch (caught) {
      setModelState("error");
      setError(caught instanceof Error ? caught.message : "Could not load YOLOX-S.");
      throw caught;
    } finally {
      modelPromiseRef.current = null;
    }
  }, []);

  useEffect(() => {
    void ensureModel().catch(() => undefined);
  }, [ensureModel]);

  useEffect(() => {
    return () => {
      analysisEpochRef.current += 1;
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const model = modelRef.current;
      modelRef.current = null;
      void model?.release();
    };
  }, []);

  const drawDetections = useCallback((items: Detection[], width: number, height: number) => {
    const canvas = canvasRef.current;
    if (!canvas || !width || !height) return;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const context = canvas.getContext("2d");
    if (!context) return;

    context.clearRect(0, 0, width, height);
    context.lineWidth = Math.max(2, width / 520);
    context.font = `700 ${Math.max(12, Math.round(width / 90))}px Arial`;
    context.textBaseline = "top";

    for (const item of items) {
      const color = item.group === "Human" ? "#58e2d3" : item.group === "Vehicle" ? "#f4cf52" : "#79bdf2";
      const x = Math.max(0, item.x1);
      const y = Math.max(0, item.y1);
      const boxWidth = Math.max(1, item.x2 - item.x1);
      const boxHeight = Math.max(1, item.y2 - item.y1);
      const label = `${item.label} ${Math.round(item.confidence * 100)}%`;
      const labelHeight = Math.max(19, width / 60);
      const labelWidth = context.measureText(label).width + 10;
      const labelY = Math.max(0, y - labelHeight);

      context.strokeStyle = color;
      context.fillStyle = `${color}12`;
      context.strokeRect(x, y, boxWidth, boxHeight);
      context.fillRect(x, y, boxWidth, boxHeight);
      context.fillStyle = color;
      context.fillRect(x, labelY, labelWidth, labelHeight);
      context.fillStyle = "#04151b";
      context.fillText(label, x + 5, labelY + 3);
    }
  }, []);

  const analyseSource = useCallback(
    async (source: HTMLVideoElement | HTMLImageElement, live: boolean) => {
      if (busyRef.current) return false;

      const width = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
      const height = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
      if (!width || !height) return live;

      const epoch = analysisEpochRef.current;
      busyRef.current = true;

      try {
        const model = await ensureModel();
        const started = performance.now();
        const result = await model.predict(source, {
          confThres: threshold,
          iouThres: NMS_IOU_THRESHOLD,
          maxDet: MAX_DETECTIONS,
        });
        const elapsed = performance.now() - started;

        if (epoch !== analysisEpochRef.current) return false;

        const items: Detection[] = [];
        const nextCounts: DetectionCounts = { total: 0, humans: 0, vehicles: 0, other: 0 };

        for (const item of result.detections) {
          const mapped = mapDetection(item.classId, displayMode);
          if (!mapped) continue;

          items.push({
            x1: item.bbox[0],
            y1: item.bbox[1],
            x2: item.bbox[2],
            y2: item.bbox[3],
            confidence: item.confidence,
            label: mapped.label,
            group: mapped.group,
          });

          nextCounts.total += 1;
          if (mapped.group === "Human") nextCounts.humans += 1;
          else if (mapped.group === "Vehicle") nextCounts.vehicles += 1;
          else nextCounts.other += 1;
        }

        setCounts(nextCounts);
        drawDetections(items, width, height);

        if (live && warmupRef.current > 0) {
          warmupRef.current -= 1;
          setWarmupRemaining(warmupRef.current);
          return true;
        }

        if (live) {
          timingRef.current = [...timingRef.current.slice(-(TIMING_WINDOW - 1)), elapsed];
          setTiming((previous) => ({
            current: elapsed,
            average: mean(timingRef.current),
            frames: previous.frames + 1,
          }));
        } else {
          timingRef.current = [elapsed];
          setTiming({ current: elapsed, average: elapsed, frames: 1 });
        }

        return true;
      } catch (caught) {
        if (epoch === analysisEpochRef.current) {
          setRunning(false);
          setError(caught instanceof Error ? caught.message : "Inference failed.");
        }
        return false;
      } finally {
        busyRef.current = false;
      }
    },
    [displayMode, drawDetections, ensureModel, threshold],
  );

  useEffect(() => {
    if (!running || sourceMode !== "camera") return;

    let cancelled = false;

    const scheduleNext = () => {
      if (cancelled) return;
      frameRef.current = requestAnimationFrame(() => void tick());
    };

    const tick = async () => {
      if (cancelled) return;

      if (busyRef.current || !videoRef.current) {
        scheduleNext();
        return;
      }

      const shouldContinue = await analyseSource(videoRef.current, true);
      if (!cancelled && shouldContinue) scheduleNext();
    };

    scheduleNext();

    return () => {
      cancelled = true;
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    };
  }, [analyseSource, running, sourceMode]);

  async function startCamera(facingMode: FacingMode) {
    stopCamera();
    clearImage();
    resetAnalysis(CAMERA_WARMUP_RUNS);
    setSourceMode("camera");
    setSelectedCamera(facingMode);
    setSourceName(facingMode === "environment" ? "Back camera" : "Front camera / Webcam");
    setError("");

    const epoch = analysisEpochRef.current;
    let stream: MediaStream | null = null;

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access requires HTTPS and a supported browser.");
      }

      const modelPromise = ensureModel();
      stream = await requestCameraStream(facingMode);
      await modelPromise;

      if (epoch !== analysisEpochRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;
      await waitForPaint();

      const video = videoRef.current;
      if (!video) throw new Error("Camera viewport is not ready.");

      video.srcObject = stream;
      await video.play();
      syncCameraAspect();

      const track = stream.getVideoTracks()[0];
      const actualFacing = track?.getSettings().facingMode;

      if (facingMode === "environment" && actualFacing && actualFacing !== "environment") {
        setSourceName(track.label || "Available camera");
      } else if (facingMode === "user" && actualFacing && actualFacing !== "user") {
        setSourceName(track.label || "Webcam");
      }

      setCameraActive(true);
      setRunning(true);
    } catch (caught) {
      stream?.getTracks().forEach((track) => track.stop());
      if (epoch !== analysisEpochRef.current) return;

      stopCamera();
      setSelectedCamera(null);
      setError(caught instanceof Error ? caught.message : "Camera permission was not granted.");
    }
  }

  function handleImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    stopCamera();
    setSelectedCamera(null);
    resetAnalysis(0);

    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);

    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setImageUrl(url);
    setSourceMode("image");
    setSourceName(file.name);
    setError("");
    event.target.value = "";
  }

  async function analyseImage() {
    const image = imageRef.current;
    if (!image || !imageUrl) return;

    resetAnalysis(0);
    setError("");

    try {
      if (!image.complete || !image.naturalWidth) await image.decode();
      if (!image.naturalWidth || !image.naturalHeight) throw new Error("The selected image could not be decoded.");
      await analyseSource(image, false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The selected image could not be decoded.");
    }
  }

  function changeDisplayMode(mode: DisplayMode) {
    if (mode === displayMode) return;
    setDisplayMode(mode);
    resetAnalysis(0);
  }

  function changeThreshold(event: ChangeEvent<HTMLInputElement>) {
    setThreshold(Number(event.target.value) / 100);
    resetAnalysis(0);
  }

  return (
    <main className="page-shell">
      <header className="hero">
        <p className="eyebrow">Browser computer vision demo</p>
        <h1>COCO Object Detection</h1>
        <p className="intro">
          YOLOX-S runs locally in the browser with a 640 × 640 model input. All 80 COCO classes are shown by
          default, with an optional Human + Vehicle view for CCTV-focused testing.
        </p>
        <div className="badges">
          <span>YOLOX-S · {MODEL_INPUT}px</span>
          <span>{provider.toUpperCase()}</span>
          <span>{displayMode === "focus" ? "Human + Vehicle" : "All COCO · 80 classes"}</span>
        </div>
      </header>

      <section className="panel viewport-panel">
        <div className="panel-head">
          <div>
            <h2>Detection viewport</h2>
            <p>{sourceName}</p>
          </div>
          <div className={`live-state ${running ? "active" : ""}`}>
            <i />
            {modelState === "loading"
              ? `Loading model ${modelProgress}%`
              : modelState === "error"
                ? "Model error"
                : running
                  ? "Analysing"
                  : sourceMode === "image" && imageUrl
                    ? "Image ready"
                    : "Ready"}
          </div>
        </div>

        <div className="media-stage">
          {sourceMode === "camera" ? (
            <div className="media-layer camera-layer" style={{ aspectRatio: cameraAspect }}>
              <video ref={videoRef} playsInline muted className="media" onLoadedMetadata={syncCameraAspect} />
              {!cameraActive ? <div className="empty-state">Select a camera below to start.</div> : null}
              <canvas ref={canvasRef} className="overlay" />
            </div>
          ) : imageUrl ? (
            <div className="media-layer image-layer">
              <img ref={imageRef} src={imageUrl} alt="Selected for detection" className="media" />
              <canvas ref={canvasRef} className="overlay" />
            </div>
          ) : (
            <div className="empty-state image-empty">Choose an image below.</div>
          )}
        </div>

        <div className="metrics">
          <Metric label="Detections" value={counts.total} />
          <Metric label="Humans" value={counts.humans} />
          <Metric label="Vehicles" value={counts.vehicles} />
          {displayMode === "all" ? <Metric label="Other objects" value={counts.other} /> : null}
          {sourceMode === "image" ? (
            <Metric label="Analysis time" value={formatMs(timing.current)} />
          ) : (
            <>
              <Metric label="Current" value={formatMs(timing.current)} />
              <Metric
                label="Stable average"
                value={formatMs(timing.average)}
                note={timing.frames ? `${timing.frames} timed frames` : undefined}
              />
              <Metric label="Effective" value={effectiveFps ? `${effectiveFps.toFixed(1)} FPS` : "—"} />
            </>
          )}
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>1</span>
          <div>
            <h2>Choose source</h2>
            <p>Use a phone camera, webcam, or an uploaded image.</p>
          </div>
        </div>

        <div className="source-grid">
          <button
            className={sourceMode === "camera" && selectedCamera === "environment" ? "primary" : undefined}
            onClick={() => void startCamera("environment")}
          >
            Back camera
          </button>
          <button
            className={sourceMode === "camera" && selectedCamera === "user" ? "primary" : undefined}
            onClick={() => void startCamera("user")}
          >
            Front camera / Webcam
          </button>
          <label className="button-like">
            Upload image
            <input type="file" accept="image/*" onChange={handleImage} />
          </label>
          {sourceMode === "image" ? (
            <button onClick={() => void analyseImage()} disabled={!imageUrl || modelState === "loading"}>
              Analyse image
            </button>
          ) : cameraActive ? (
            <button onClick={() => setRunning((value) => !value)}>{running ? "Pause AI" : "Resume AI"}</button>
          ) : null}
        </div>

        <p className="source-note">
          The visible preview keeps the source aspect ratio. YOLOX-S runs internally at 640 × 640.
        </p>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>2</span>
          <div>
            <h2>Detection set</h2>
            <p>YOLOX-S is the only model. This control only changes which detections are displayed.</p>
          </div>
        </div>
        <div className="source-grid">
          <button className={displayMode === "all" ? "primary" : undefined} onClick={() => changeDisplayMode("all")}>
            All COCO objects (80)
          </button>
          <button
            className={displayMode === "focus" ? "primary" : undefined}
            onClick={() => changeDisplayMode("focus")}
          >
            Human + Vehicle
          </button>
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>3</span>
          <div>
            <h2>Detection confidence</h2>
            <p>Default is 60%. Lower values find more candidates but may also add false detections.</p>
          </div>
        </div>
        <div className="slider-row">
          <input
            type="range"
            min="10"
            max="80"
            step="1"
            value={Math.round(threshold * 100)}
            onChange={changeThreshold}
          />
          <strong>{Math.round(threshold * 100)}%</strong>
        </div>
      </section>

      {error ? <div className="error-box">{error}</div> : null}

      <footer className="footer-card">
        <div>
          <strong>YOLOX-S</strong>
          <p>One accuracy-focused model running at its native 640 × 640 input.</p>
        </div>
        <div>
          <strong>WebGPU first</strong>
          <p>WebGPU is preferred when available, with WASM as the fallback runtime.</p>
        </div>
        <div>
          <strong>Local media</strong>
          <p>Camera frames and uploaded images stay in the browser during inference.</p>
        </div>
      </footer>

      {sourceMode === "camera" && warmupRemaining > 0 && cameraActive ? (
        <div className="warmup-note">Warming up… {warmupRemaining} pass{warmupRemaining === 1 ? "" : "es"} left</div>
      ) : null}
    </main>
  );
}
