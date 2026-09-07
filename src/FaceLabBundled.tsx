import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";
import { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";

type RunningMode = "IMAGE" | "VIDEO";
type FacingMode = "environment" | "user";
type SourceMode = "camera" | "image";

type FaceDetection = {
  boundingBox?: {
    originX?: number;
    originY?: number;
    width?: number;
    height?: number;
  };
  categories?: Array<{ score?: number }>;
  keypoints?: Array<{ x?: number; y?: number }>;
};

type FaceCrop = {
  id: number;
  dataUrl: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  cropWidth: number;
  cropHeight: number;
  frameWidth: number;
  frameHeight: number;
  keypoints: number;
  capturedAt: string;
};

const WASM_URLS = [
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm",
  "https://unpkg.com/@mediapipe/tasks-vision@1.0.1/wasm",
];
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 480;
const LIVE_INTERVAL_MS = 250;
const CROP_MARGIN = 0.18;

function formatMs(value: number) {
  if (!value || !Number.isFinite(value)) return "—";
  return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

async function loadVisionFileset() {
  let lastError: unknown = null;
  for (const wasmUrl of WASM_URLS) {
    try {
      return await FilesetResolver.forVisionTasks(wasmUrl);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("MediaPipe WASM files could not be loaded.");
}

export default function FaceLabBundled() {
  const [sourceMode, setSourceMode] = useState<SourceMode>("camera");
  const [threshold, setThreshold] = useState(0.5);
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [provider, setProvider] = useState("GPU");
  const [sourceName, setSourceName] = useState("Choose a source to begin");
  const [imageUrl, setImageUrl] = useState("");
  const [cameraActive, setCameraActive] = useState(false);
  const [running, setRunning] = useState(false);
  const [cameraAspect, setCameraAspect] = useState("4 / 3");
  const [faceCount, setFaceCount] = useState(0);
  const [analysisMs, setAnalysisMs] = useState(0);
  const [frameSize, setFrameSize] = useState("—");
  const [crops, setCrops] = useState<FaceCrop[]>([]);
  const [error, setError] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectorRef = useRef<FaceDetector | null>(null);
  const detectorPromiseRef = useRef<Promise<FaceDetector> | null>(null);
  const detectorModeRef = useRef<RunningMode>("IMAGE");
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef("");
  const loopRef = useRef<number | null>(null);
  const lastRunRef = useRef(0);
  const busyRef = useRef(false);

  const clearResults = useCallback(() => {
    setFaceCount(0);
    setAnalysisMs(0);
    setFrameSize("—");
    setCrops([]);
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setRunning(false);
  }, []);

  const syncCameraAspect = useCallback(() => {
    const video = videoRef.current;
    if (!video?.videoWidth || !video.videoHeight) return;
    setCameraAspect(`${video.videoWidth} / ${video.videoHeight}`);
  }, []);

  const ensureDetector = useCallback(async () => {
    if (detectorRef.current) return detectorRef.current;
    if (detectorPromiseRef.current) return detectorPromiseRef.current;

    setModelState("loading");
    setError("");

    const promise = (async () => {
      const vision = await loadVisionFileset();

      try {
        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "IMAGE",
          minDetectionConfidence: threshold,
        });
        detectorRef.current = detector;
        detectorModeRef.current = "IMAGE";
        setProvider("GPU");
        setModelState("ready");
        return detector;
      } catch {
        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
          runningMode: "IMAGE",
          minDetectionConfidence: threshold,
        });
        detectorRef.current = detector;
        detectorModeRef.current = "IMAGE";
        setProvider("CPU");
        setModelState("ready");
        return detector;
      }
    })();

    detectorPromiseRef.current = promise;
    try {
      return await promise;
    } catch (caught) {
      setModelState("error");
      const message = caught instanceof Error ? caught.message : "Face detector could not be loaded.";
      setError(message);
      throw caught;
    } finally {
      detectorPromiseRef.current = null;
    }
  }, [threshold]);

  const setDetectorMode = useCallback(
    async (mode: RunningMode) => {
      const detector = await ensureDetector();
      if (detectorModeRef.current !== mode) {
        await detector.setOptions({ runningMode: mode });
        detectorModeRef.current = mode;
      }
      await detector.setOptions({ minDetectionConfidence: threshold });
      return detector;
    },
    [ensureDetector, threshold],
  );

  useEffect(() => {
    void ensureDetector().catch(() => undefined);
  }, [ensureDetector]);

  useEffect(() => {
    return () => {
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      detectorRef.current?.close();
    };
  }, []);

  const drawAndCrop = useCallback(
    (
      source: HTMLVideoElement | HTMLImageElement,
      detections: FaceDetection[],
      width: number,
      height: number,
    ) => {
      const overlay = canvasRef.current;
      if (!overlay) return;
      overlay.width = width;
      overlay.height = height;
      const context = overlay.getContext("2d");
      if (!context) return;

      context.clearRect(0, 0, width, height);
      context.lineWidth = Math.max(2, width / 420);
      context.font = `700 ${Math.max(13, Math.round(width / 65))}px Arial`;
      context.textBaseline = "top";

      const capturedAt = new Date().toLocaleTimeString();
      const nextCrops: FaceCrop[] = [];

      detections.forEach((detection, index) => {
        const box = detection.boundingBox;
        if (!box) return;

        const x = Math.max(0, Math.round(box.originX ?? 0));
        const y = Math.max(0, Math.round(box.originY ?? 0));
        const boxWidth = Math.max(1, Math.round(box.width ?? 0));
        const boxHeight = Math.max(1, Math.round(box.height ?? 0));
        const confidence = detection.categories?.[0]?.score ?? 0;

        context.strokeStyle = "#58e2d3";
        context.fillStyle = "rgba(88, 226, 211, 0.08)";
        context.strokeRect(x, y, boxWidth, boxHeight);
        context.fillRect(x, y, boxWidth, boxHeight);

        const label = `Face ${index + 1} · ${Math.round(confidence * 100)}%`;
        const labelHeight = Math.max(22, width / 48);
        const labelWidth = context.measureText(label).width + 12;
        context.fillStyle = "#58e2d3";
        context.fillRect(x, Math.max(0, y - labelHeight), labelWidth, labelHeight);
        context.fillStyle = "#04151b";
        context.fillText(label, x + 6, Math.max(1, y - labelHeight + 4));

        const expandedWidth = boxWidth * (1 + CROP_MARGIN * 2);
        const expandedHeight = boxHeight * (1 + CROP_MARGIN * 2);
        const side = Math.min(Math.max(expandedWidth, expandedHeight), width, height);
        const centerX = x + boxWidth / 2;
        const centerY = y + boxHeight / 2;
        const cropX = Math.max(0, Math.min(width - side, centerX - side / 2));
        const cropY = Math.max(0, Math.min(height - side, centerY - side / 2));
        const cropSize = Math.max(1, Math.round(side));

        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = cropSize;
        cropCanvas.height = cropSize;
        const cropContext = cropCanvas.getContext("2d");
        if (!cropContext) return;
        cropContext.drawImage(source, cropX, cropY, side, side, 0, 0, cropSize, cropSize);

        nextCrops.push({
          id: index + 1,
          dataUrl: cropCanvas.toDataURL("image/jpeg", 0.9),
          confidence,
          x,
          y,
          width: boxWidth,
          height: boxHeight,
          cropWidth: cropSize,
          cropHeight: cropSize,
          frameWidth: width,
          frameHeight: height,
          keypoints: detection.keypoints?.length ?? 0,
          capturedAt,
        });
      });

      setCrops(nextCrops);
    },
    [],
  );

  const processSource = useCallback(
    async (source: HTMLVideoElement | HTMLImageElement, live: boolean) => {
      if (busyRef.current) return;
      const width = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
      const height = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
      if (!width || !height) return;

      busyRef.current = true;
      try {
        const detector = await setDetectorMode(live ? "VIDEO" : "IMAGE");
        const started = performance.now();
        const result = live
          ? detector.detectForVideo(source as HTMLVideoElement, performance.now())
          : detector.detect(source as HTMLImageElement);
        const elapsed = performance.now() - started;
        const detections = (result.detections ?? []) as FaceDetection[];

        setFaceCount(detections.length);
        setAnalysisMs(elapsed);
        setFrameSize(`${width} × ${height}`);
        drawAndCrop(source, detections, width, height);
      } catch (caught) {
        setRunning(false);
        setError(caught instanceof Error ? caught.message : "Face detection failed.");
      } finally {
        busyRef.current = false;
      }
    },
    [drawAndCrop, setDetectorMode],
  );

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

  const startCamera = useCallback(
    async (facingMode: FacingMode) => {
      stopCamera();
      clearResults();
      setSourceMode("camera");
      setSourceName(facingMode === "environment" ? "Back camera" : "Front camera / Webcam");
      setError("");

      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("Camera access requires HTTPS and a supported browser.");
        }
        await setDetectorMode("VIDEO");

        let stream: MediaStream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
              facingMode: { ideal: facingMode },
              width: { ideal: CAMERA_WIDTH },
              height: { ideal: CAMERA_HEIGHT },
              frameRate: { ideal: 30, max: 30 },
            },
          });
        } catch {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { width: { ideal: CAMERA_WIDTH }, height: { ideal: CAMERA_HEIGHT } },
          });
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
    },
    [clearResults, setDetectorMode, stopCamera, syncCameraAspect],
  );

  const handleImage = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      stopCamera();
      clearResults();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      setImageUrl(url);
      setSourceMode("image");
      setSourceName(file.name);
      setError("");
      event.target.value = "";
    },
    [clearResults, stopCamera],
  );

  const analyseImage = useCallback(async () => {
    if (!imageRef.current || !imageUrl) return;
    clearResults();
    await processSource(imageRef.current, false);
  }, [clearResults, imageUrl, processSource]);

  return (
    <main className="page-shell face-lab">
      <header className="hero">
        <p className="eyebrow">Browser face detection test lab</p>
        <h1>Face Detection + Crop Metadata</h1>
        <p className="intro">
          Detect faces locally in the browser, crop every detected face and inspect the detector metadata.
          This is face detection only — it does not identify or recognise a person.
        </p>
        <div className="badges">
          <span>MediaPipe Face Detector</span>
          <span>{modelState === "ready" ? provider : modelState.toUpperCase()}</span>
          <span>Bundled JS · local inference</span>
        </div>
        <a className="lab-link" href={window.location.pathname}>← Human + Vehicle demo</a>
      </header>

      <section className="panel viewport-panel">
        <div className="panel-head">
          <div>
            <h2>Face detection viewport</h2>
            <p>{sourceName}</p>
          </div>
          <div className={`live-state ${running ? "active" : ""}`}>
            <i />
            {modelState === "loading" ? "Loading detector" : running ? "Detecting" : "Ready"}
          </div>
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
          ) : (
            <div className="empty-state image-empty">Choose an image or camera below.</div>
          )}
        </div>

        <div className="metrics face-metrics">
          <Metric label="Faces" value={faceCount} />
          <Metric label="Detection time" value={formatMs(analysisMs)} />
          <Metric label="Source" value={frameSize} />
          <Metric label="Confidence" value={`${Math.round(threshold * 100)}%`} />
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>1</span>
          <div>
            <h2>Choose source</h2>
            <p>Use a camera for live testing or upload an image for repeatable face-crop tests.</p>
          </div>
        </div>
        <div className="source-grid">
          <button onClick={() => void startCamera("environment")}>Back camera</button>
          <button className="primary" onClick={() => void startCamera("user")}>Front camera / Webcam</button>
          <label className="button-like">
            Upload image
            <input type="file" accept="image/*" onChange={handleImage} />
          </label>
          {sourceMode === "image" ? (
            <button onClick={() => void analyseImage()} disabled={!imageUrl || modelState !== "ready"}>
              Analyse image
            </button>
          ) : cameraActive ? (
            <button onClick={() => setRunning((value) => !value)}>{running ? "Pause detection" : "Resume detection"}</button>
          ) : null}
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>2</span>
          <div>
            <h2>Face confidence</h2>
            <p>Lower confidence can find more difficult faces but may increase false detections.</p>
          </div>
        </div>
        <div className="slider-row">
          <input
            type="range"
            min="20"
            max="90"
            step="1"
            value={Math.round(threshold * 100)}
            onChange={(event) => setThreshold(Number(event.target.value) / 100)}
          />
          <strong>{Math.round(threshold * 100)}%</strong>
        </div>
      </section>

      <section className="panel face-results-panel">
        <div className="panel-head">
          <div>
            <h2>Detected face crops</h2>
            <p>Latest detected faces with metadata from the current image/frame.</p>
          </div>
          <strong className="crop-count">{crops.length} crop{crops.length === 1 ? "" : "s"}</strong>
        </div>

        {crops.length ? (
          <div className="face-crop-grid">
            {crops.map((crop) => (
              <article className="face-card" key={`${crop.id}-${crop.capturedAt}`}>
                <img src={crop.dataUrl} alt={`Detected face ${crop.id}`} />
                <div className="face-card-body">
                  <div className="face-card-title">
                    <strong>Face {crop.id}</strong>
                    <span>{Math.round(crop.confidence * 100)}%</span>
                  </div>
                  <dl className="metadata-grid">
                    <div><dt>Box X / Y</dt><dd>{crop.x} / {crop.y}</dd></div>
                    <div><dt>Face W × H</dt><dd>{crop.width} × {crop.height} px</dd></div>
                    <div><dt>Crop</dt><dd>{crop.cropWidth} × {crop.cropHeight} px</dd></div>
                    <div><dt>Frame</dt><dd>{crop.frameWidth} × {crop.frameHeight}</dd></div>
                    <div><dt>Keypoints</dt><dd>{crop.keypoints}</dd></div>
                    <div><dt>Captured</dt><dd>{crop.capturedAt}</dd></div>
                  </dl>
                  <a className="crop-download" href={crop.dataUrl} download={`face-${crop.id}.jpg`}>Save crop</a>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="face-empty">Detected face crops will appear here.</div>
        )}
      </section>

      {error ? <div className="error-box">{error}</div> : null}

      <footer className="footer-card">
        <div>
          <strong>Face detection</strong>
          <p>Returns face location, confidence and detector keypoints. No identity matching is performed.</p>
        </div>
        <div>
          <strong>Automatic crop</strong>
          <p>Each face is cropped with a small margin so we can inspect the image available to later analytics.</p>
        </div>
        <div>
          <strong>Private test</strong>
          <p>Camera frames and uploaded images stay in the browser while the test page is running.</p>
        </div>
      </footer>
    </main>
  );
}
