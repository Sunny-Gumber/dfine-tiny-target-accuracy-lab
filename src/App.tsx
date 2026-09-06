import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type FacingMode = "environment" | "user";
type SourceMode = "camera" | "image";
type TargetKind = "Human" | "Vehicle";

type RuntimeDetection = {
  classId: number;
  confidence: number;
  bbox: [number, number, number, number];
};

type RuntimeResult = {
  detections: RuntimeDetection[];
};

type RuntimeModel = {
  predict: (
    input: HTMLVideoElement | HTMLImageElement,
    options?: { confThres?: number; iouThres?: number; maxDet?: number },
  ) => Promise<RuntimeResult>;
  release: () => Promise<void>;
  provider: "webgpu" | "wasm" | null;
};

type RuntimeModule = {
  loadModel: (
    source: string,
    options?: {
      confThres?: number;
      iouThres?: number;
      maxDet?: number;
      device?: "auto" | "webgpu" | "wasm" | ("webgpu" | "wasm")[];
      modelFamily?: "yolox" | "auto";
      onProgress?: (progress: number) => void;
    },
  ) => Promise<RuntimeModel>;
};

type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  kind: TargetKind;
};

const RUNTIME_URL = "https://esm.sh/libreyolo-web@0.0.6?bundle&deps=onnxruntime-web@1.24.3";
const MODEL_NAME = "LibreYOLOXn";
const MODEL_INPUT = 416;
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 360;
const CAMERA_WARMUP_RUNS = 3;
const LOOP_GAP_MS = 16;
const ROAD_VEHICLE_CLASSES = new Set([1, 2, 3, 5, 7]);

function targetKind(classId: number): TargetKind | null {
  if (classId === 0) return "Human";
  if (ROAD_VEHICLE_CLASSES.has(classId)) return "Vehicle";
  return null;
}

function formatMs(value: number) {
  if (!value || !Number.isFinite(value)) return "—";
  return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;
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
  const [threshold, setThreshold] = useState(0.3);
  const [running, setRunning] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [modelProgress, setModelProgress] = useState(0);
  const [provider, setProvider] = useState("waiting");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [currentMs, setCurrentMs] = useState(0);
  const [averageMs, setAverageMs] = useState(0);
  const [analysedFrames, setAnalysedFrames] = useState(0);
  const [warmupRemaining, setWarmupRemaining] = useState(CAMERA_WARMUP_RUNS);
  const [error, setError] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [sourceName, setSourceName] = useState("Choose a source to begin");

  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<RuntimeModel | null>(null);
  const runtimeRef = useRef<RuntimeModule | null>(null);
  const modelPromiseRef = useRef<Promise<RuntimeModel> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef("");
  const busyRef = useRef(false);
  const loopRef = useRef<number | null>(null);
  const lastStartRef = useRef(0);
  const timingRef = useRef<number[]>([]);
  const warmupRef = useRef(CAMERA_WARMUP_RUNS);

  const humans = useMemo(
    () => detections.filter((item) => item.kind === "Human").length,
    [detections],
  );
  const vehicles = detections.length - humans;
  const effectiveFps = averageMs ? 1000 / averageMs : 0;

  const clearOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const resetStats = useCallback(
    (warmups = 0) => {
      timingRef.current = [];
      warmupRef.current = warmups;
      setWarmupRemaining(warmups);
      setCurrentMs(0);
      setAverageMs(0);
      setAnalysedFrames(0);
      setDetections([]);
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

  const ensureModel = useCallback(async () => {
    if (modelRef.current) return modelRef.current;
    if (modelPromiseRef.current) return modelPromiseRef.current;

    setModelState("loading");
    setModelProgress(2);
    setError("");

    const promise = (async () => {
      if (!runtimeRef.current) {
        const moduleUrl: string = RUNTIME_URL;
        runtimeRef.current = (await import(/* @vite-ignore */ moduleUrl)) as unknown as RuntimeModule;
      }

      const model = await runtimeRef.current.loadModel(MODEL_NAME, {
        device: ["webgpu", "wasm"],
        modelFamily: "yolox",
        confThres: 0.12,
        iouThres: 0.65,
        maxDet: 120,
        onProgress: (progress) => {
          setModelProgress(Math.max(2, Math.min(99, Math.round(progress * 100))));
        },
      });

      modelRef.current = model;
      setProvider(model.provider || "wasm");
      setModelProgress(100);
      setModelState("ready");
      return model;
    })();

    modelPromiseRef.current = promise;
    try {
      return await promise;
    } catch (caught) {
      setModelState("error");
      const message = caught instanceof Error ? caught.message : "Could not load YOLOX Nano.";
      setError(message);
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
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      void modelRef.current?.release();
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
      const color = item.kind === "Human" ? "#58e2d3" : "#f4cf52";
      const x = Math.max(0, item.x1);
      const y = Math.max(0, item.y1);
      const boxWidth = Math.max(1, item.x2 - item.x1);
      const boxHeight = Math.max(1, item.y2 - item.y1);
      const label = `${item.kind} ${Math.round(item.confidence * 100)}%`;
      const labelHeight = Math.max(19, width / 60);
      const labelWidth = context.measureText(label).width + 10;

      context.strokeStyle = color;
      context.fillStyle = `${color}12`;
      context.strokeRect(x, y, boxWidth, boxHeight);
      context.fillRect(x, y, boxWidth, boxHeight);
      context.fillStyle = color;
      context.fillRect(x, Math.max(0, y - labelHeight), labelWidth, labelHeight);
      context.fillStyle = "#04151b";
      context.fillText(label, x + 5, Math.max(1, y - labelHeight + 3));
    }
  }, []);

  const analyseSource = useCallback(
    async (source: HTMLVideoElement | HTMLImageElement, live: boolean) => {
      if (busyRef.current) return;

      const width = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
      const height = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
      if (!width || !height) return;

      busyRef.current = true;
      try {
        const model = await ensureModel();
        const started = performance.now();
        const result = await model.predict(source, {
          confThres: Math.max(0.08, threshold),
          iouThres: 0.65,
          maxDet: 120,
        });
        const elapsed = performance.now() - started;

        const items = result.detections
          .map((item): Detection | null => {
            const kind = targetKind(item.classId);
            if (!kind || item.confidence < threshold) return null;
            return {
              x1: item.bbox[0],
              y1: item.bbox[1],
              x2: item.bbox[2],
              y2: item.bbox[3],
              confidence: item.confidence,
              kind,
            };
          })
          .filter((item): item is Detection => item !== null);

        setDetections(items);
        setProvider(model.provider || "wasm");
        drawDetections(items, width, height);

        if (live && warmupRef.current > 0) {
          warmupRef.current -= 1;
          setWarmupRemaining(warmupRef.current);
          return;
        }

        if (live) {
          timingRef.current = [...timingRef.current.slice(-29), elapsed];
          const mean = timingRef.current.reduce((sum, value) => sum + value, 0) / timingRef.current.length;
          setCurrentMs(elapsed);
          setAverageMs(mean);
          setAnalysedFrames((count) => count + 1);
        } else {
          timingRef.current = [elapsed];
          setCurrentMs(elapsed);
          setAverageMs(elapsed);
          setAnalysedFrames(1);
        }
      } catch (caught) {
        setRunning(false);
        setError(caught instanceof Error ? caught.message : "Inference failed.");
      } finally {
        busyRef.current = false;
      }
    },
    [drawDetections, ensureModel, threshold],
  );

  useEffect(() => {
    if (!running || sourceMode !== "camera") return;

    let cancelled = false;
    const tick = (now: number) => {
      if (cancelled) return;
      if (!busyRef.current && now - lastStartRef.current >= LOOP_GAP_MS && videoRef.current) {
        lastStartRef.current = now;
        void analyseSource(videoRef.current, true);
      }
      loopRef.current = requestAnimationFrame(tick);
    };

    loopRef.current = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      loopRef.current = null;
    };
  }, [analyseSource, running, sourceMode]);

  const startCamera = useCallback(
    async (facingMode: FacingMode) => {
      stopCamera();
      resetStats(CAMERA_WARMUP_RUNS);
      setSourceMode("camera");
      setSourceName(facingMode === "environment" ? "Rear camera" : "Front camera / webcam");
      setError("");

      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("Camera access requires HTTPS and a supported browser.");
        }

        await ensureModel();
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: CAMERA_WIDTH },
            height: { ideal: CAMERA_HEIGHT },
            frameRate: { ideal: 30, max: 30 },
          },
        });

        streamRef.current = stream;
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
    },
    [ensureModel, resetStats, stopCamera],
  );

  const handleImage = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
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
      setError("");
      event.target.value = "";
    },
    [resetStats, stopCamera],
  );

  const analyseImage = useCallback(async () => {
    if (!imageRef.current || !imageUrl) return;
    resetStats(0);
    await analyseSource(imageRef.current, false);
  }, [analyseSource, imageUrl, resetStats]);

  return (
    <main className="page-shell">
      <header className="hero">
        <p className="eyebrow">Browser computer vision demo</p>
        <h1>Human + Vehicle Detection</h1>
        <p className="intro">
          A lightweight YOLOX Nano demo that runs directly in the browser. Use a phone camera,
          webcam or an uploaded image without sending the media to an application server.
        </p>
        <div className="badges">
          <span>YOLOX Nano · {MODEL_INPUT}px</span>
          <span>{provider.toUpperCase()}</span>
          <span>Human + Vehicle</span>
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
              : running
                ? "Analysing"
                : sourceMode === "image" && imageUrl
                  ? "Image ready"
                  : "Ready"}
          </div>
        </div>

        <div className="media-stage">
          {sourceMode === "camera" ? (
            <div className="media-layer camera-layer">
              <video ref={videoRef} playsInline muted className="media" />
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

        <div className={`metrics ${sourceMode === "image" ? "image-metrics" : ""}`}>
          <Metric label="Detections" value={detections.length} />
          <Metric label="Humans" value={humans} />
          <Metric label="Vehicles" value={vehicles} />
          {sourceMode === "image" ? (
            <Metric label="Analysis time" value={formatMs(currentMs)} />
          ) : (
            <>
              <Metric label="Current" value={formatMs(currentMs)} />
              <Metric
                label="Stable average"
                value={formatMs(averageMs)}
                note={analysedFrames ? `${analysedFrames} timed frames` : undefined}
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
            <p>Camera input is requested at 640 × 360. Images use the complete uploaded frame.</p>
          </div>
        </div>

        <div className="source-grid">
          <button className="primary" onClick={() => void startCamera("environment")}>Rear camera</button>
          <button onClick={() => void startCamera("user")}>Front / webcam</button>
          <label className="button-like">
            Upload image
            <input type="file" accept="image/*" onChange={handleImage} />
          </label>
          {sourceMode === "image" ? (
            <button onClick={() => void analyseImage()} disabled={!imageUrl || modelState !== "ready"}>
              Analyse image
            </button>
          ) : cameraActive ? (
            <button onClick={() => setRunning((value) => !value)}>{running ? "Pause AI" : "Resume AI"}</button>
          ) : null}
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>2</span>
          <div>
            <h2>Detection confidence</h2>
            <p>Lower values find more candidates but can also increase false detections.</p>
          </div>
        </div>
        <div className="slider-row">
          <input
            type="range"
            min="10"
            max="80"
            step="1"
            value={Math.round(threshold * 100)}
            onChange={(event) => setThreshold(Number(event.target.value) / 100)}
          />
          <strong>{Math.round(threshold * 100)}%</strong>
        </div>
      </section>

      {error ? <div className="error-box">{error}</div> : null}

      <footer className="footer-card">
        <div>
          <strong>One model</strong>
          <p>YOLOX Nano runs one inference pass at its native 416 × 416 input.</p>
        </div>
        <div>
          <strong>Simple labels</strong>
          <p>COCO road-vehicle classes are grouped into a single Vehicle label.</p>
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
