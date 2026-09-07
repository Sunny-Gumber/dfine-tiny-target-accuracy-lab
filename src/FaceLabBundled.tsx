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

type FaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type TrackedFace = FaceBox & {
  trackId: number;
  lastSeen: number;
};

type FaceCapture = FaceBox & {
  id: number;
  dataUrl: string;
  confidence: number;
  frameWidth: number;
  frameHeight: number;
  keypoints: number;
  capturedAt: string;
  source: "Camera" | "Image";
  detectionMs: number;
  coveragePercent: number;
  faceRatio: number;
  brightnessPercent: number;
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
const TRACK_TTL_MS = 1600;
const TRACK_IOU_THRESHOLD = 0.24;
const FIXED_CROP_WIDTH = 160;
const FIXED_CROP_HEIGHT = 320;
const MAX_CAPTURE_LOG = 100;

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

function boxIou(a: FaceBox, b: FaceBox) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function getFaceBox(detection: FaceDetection): FaceBox | null {
  const box = detection.boundingBox;
  if (!box) return null;
  return {
    x: Math.max(0, Math.round(box.originX ?? 0)),
    y: Math.max(0, Math.round(box.originY ?? 0)),
    width: Math.max(1, Math.round(box.width ?? 0)),
    height: Math.max(1, Math.round(box.height ?? 0)),
  };
}

function getBrightnessPercent(canvas: HTMLCanvasElement) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return 0;
  const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
  let total = 0;
  let pixels = 0;
  for (let index = 0; index < data.length; index += 16) {
    const red = data[index] ?? 0;
    const green = data[index + 1] ?? 0;
    const blue = data[index + 2] ?? 0;
    total += 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    pixels += 1;
  }
  return pixels ? Math.round((total / pixels / 255) * 100) : 0;
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
  const [captures, setCaptures] = useState<FaceCapture[]>([]);
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
  const tracksRef = useRef<TrackedFace[]>([]);
  const trackIdRef = useRef(1);
  const captureIdRef = useRef(1);

  const clearViewport = useCallback(() => {
    setFaceCount(0);
    setAnalysisMs(0);
    setFrameSize("—");
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const clearCaptureLog = useCallback(() => {
    setCaptures([]);
    tracksRef.current = [];
    trackIdRef.current = 1;
    captureIdRef.current = 1;
  }, []);

  const resetSession = useCallback(() => {
    clearViewport();
    clearCaptureLog();
  }, [clearCaptureLog, clearViewport]);

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

  const createCapture = useCallback(
    (
      source: HTMLVideoElement | HTMLImageElement,
      detection: FaceDetection,
      box: FaceBox,
      frameWidth: number,
      frameHeight: number,
      detectionMs: number,
      sourceType: "Camera" | "Image",
    ): FaceCapture | null => {
      const targetRatio = FIXED_CROP_WIDTH / FIXED_CROP_HEIGHT;
      let cropWidthInSource = box.width * 1.55;
      let cropHeightInSource = cropWidthInSource / targetRatio;

      const minimumHeight = box.height * 1.8;
      if (cropHeightInSource < minimumHeight) {
        cropHeightInSource = minimumHeight;
        cropWidthInSource = cropHeightInSource * targetRatio;
      }

      const frameScale = Math.min(
        1,
        frameWidth / cropWidthInSource,
        frameHeight / cropHeightInSource,
      );
      cropWidthInSource *= frameScale;
      cropHeightInSource *= frameScale;

      const centerX = box.x + box.width / 2;
      const centerY = box.y + box.height / 2 + box.height * 0.16;
      const cropX = Math.max(0, Math.min(frameWidth - cropWidthInSource, centerX - cropWidthInSource / 2));
      const cropY = Math.max(0, Math.min(frameHeight - cropHeightInSource, centerY - cropHeightInSource / 2));

      const cropCanvas = document.createElement("canvas");
      cropCanvas.width = FIXED_CROP_WIDTH;
      cropCanvas.height = FIXED_CROP_HEIGHT;
      const cropContext = cropCanvas.getContext("2d");
      if (!cropContext) return null;

      cropContext.drawImage(
        source,
        cropX,
        cropY,
        cropWidthInSource,
        cropHeightInSource,
        0,
        0,
        FIXED_CROP_WIDTH,
        FIXED_CROP_HEIGHT,
      );

      const confidence = detection.categories?.[0]?.score ?? 0;
      const coveragePercent = (box.width * box.height * 100) / (frameWidth * frameHeight);

      return {
        id: captureIdRef.current++,
        dataUrl: cropCanvas.toDataURL("image/jpeg", 0.92),
        confidence,
        ...box,
        frameWidth,
        frameHeight,
        keypoints: detection.keypoints?.length ?? 0,
        capturedAt: new Date().toLocaleTimeString(),
        source: sourceType,
        detectionMs,
        coveragePercent,
        faceRatio: box.width / box.height,
        brightnessPercent: getBrightnessPercent(cropCanvas),
      };
    },
    [],
  );

  const drawCurrentFaces = useCallback((detections: FaceDetection[], width: number, height: number) => {
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

    detections.forEach((detection, index) => {
      const box = getFaceBox(detection);
      if (!box) return;
      const confidence = detection.categories?.[0]?.score ?? 0;
      context.strokeStyle = "#58e2d3";
      context.fillStyle = "rgba(88, 226, 211, 0.08)";
      context.strokeRect(box.x, box.y, box.width, box.height);
      context.fillRect(box.x, box.y, box.width, box.height);

      const label = `Face ${index + 1} · ${Math.round(confidence * 100)}%`;
      const labelHeight = Math.max(22, width / 48);
      const labelWidth = context.measureText(label).width + 12;
      context.fillStyle = "#58e2d3";
      context.fillRect(box.x, Math.max(0, box.y - labelHeight), labelWidth, labelHeight);
      context.fillStyle = "#04151b";
      context.fillText(label, box.x + 6, Math.max(1, box.y - labelHeight + 4));
    });
  }, []);

  const captureNewLiveFaces = useCallback(
    (
      source: HTMLVideoElement,
      detections: FaceDetection[],
      frameWidth: number,
      frameHeight: number,
      detectionMs: number,
    ) => {
      const now = performance.now();
      const tracks = tracksRef.current.filter((track) => now - track.lastSeen < TRACK_TTL_MS);
      const usedTracks = new Set<number>();
      const newCaptures: FaceCapture[] = [];

      detections.forEach((detection) => {
        const box = getFaceBox(detection);
        if (!box) return;

        let bestTrack: TrackedFace | null = null;
        let bestScore = 0;
        for (const track of tracks) {
          if (usedTracks.has(track.trackId)) continue;
          const score = boxIou(box, track);
          if (score > bestScore) {
            bestScore = score;
            bestTrack = track;
          }
        }

        if (bestTrack && bestScore >= TRACK_IOU_THRESHOLD) {
          bestTrack.x = box.x;
          bestTrack.y = box.y;
          bestTrack.width = box.width;
          bestTrack.height = box.height;
          bestTrack.lastSeen = now;
          usedTracks.add(bestTrack.trackId);
          return;
        }

        const track: TrackedFace = {
          trackId: trackIdRef.current++,
          ...box,
          lastSeen: now,
        };
        tracks.push(track);
        usedTracks.add(track.trackId);

        const capture = createCapture(
          source,
          detection,
          box,
          frameWidth,
          frameHeight,
          detectionMs,
          "Camera",
        );
        if (capture) newCaptures.push(capture);
      });

      tracksRef.current = tracks;
      if (newCaptures.length) {
        setCaptures((current) => [...current, ...newCaptures].slice(-MAX_CAPTURE_LOG));
      }
    },
    [createCapture],
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
        drawCurrentFaces(detections, width, height);

        if (live) {
          captureNewLiveFaces(source as HTMLVideoElement, detections, width, height, elapsed);
        } else {
          const imageCaptures = detections
            .map((detection) => {
              const box = getFaceBox(detection);
              if (!box) return null;
              return createCapture(source, detection, box, width, height, elapsed, "Image");
            })
            .filter((capture): capture is FaceCapture => capture !== null);
          setCaptures(imageCaptures.slice(0, MAX_CAPTURE_LOG));
        }
      } catch (caught) {
        setRunning(false);
        setError(caught instanceof Error ? caught.message : "Face detection failed.");
      } finally {
        busyRef.current = false;
      }
    },
    [captureNewLiveFaces, createCapture, drawCurrentFaces, setDetectorMode],
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
      resetSession();
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
    [resetSession, setDetectorMode, stopCamera, syncCameraAspect],
  );

  const handleImage = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      stopCamera();
      resetSession();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      setImageUrl(url);
      setSourceMode("image");
      setSourceName(file.name);
      setError("");
      event.target.value = "";
    },
    [resetSession, stopCamera],
  );

  const analyseImage = useCallback(async () => {
    if (!imageRef.current || !imageUrl) return;
    clearViewport();
    setCaptures([]);
    captureIdRef.current = 1;
    await processSource(imageRef.current, false);
  }, [clearViewport, imageUrl, processSource]);

  return (
    <main className="page-shell face-lab">
      <header className="hero">
        <p className="eyebrow">Browser face detection test lab</p>
        <h1>Face Capture Log + Metadata</h1>
        <p className="intro">
          Detect faces locally, keep one fixed portrait crop when a new face enters the camera and build a persistent capture log.
          This page performs face detection only; it does not identify a person.
        </p>
        <div className="badges">
          <span>MediaPipe Face Detector</span>
          <span>{modelState === "ready" ? provider : modelState.toUpperCase()}</span>
          <span>Fixed crop · 1:2 · {FIXED_CROP_WIDTH} × {FIXED_CROP_HEIGHT}px</span>
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
          <Metric label="Faces now" value={faceCount} />
          <Metric label="Captured" value={captures.length} />
          <Metric label="Detection time" value={formatMs(analysisMs)} />
          <Metric label="Source" value={frameSize} />
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>1</span>
          <div>
            <h2>Choose source</h2>
            <p>Each new face is added once to the log. A face that stays in view is not continuously re-captured.</p>
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
            <p>Lower confidence may capture more difficult faces but can also add false detections to the log.</p>
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
            <h2>Face capture log</h2>
            <p>Rows stay in the list after capture. Fixed crop ratio is 1:2, equivalent to the 10 mm × 20 mm proportion.</p>
          </div>
          <div className="log-actions">
            <strong className="crop-count">{captures.length} captured</strong>
            <button className="small-button" onClick={clearCaptureLog} disabled={!captures.length}>Clear list</button>
          </div>
        </div>

        {captures.length ? (
          <div className="face-log">
            {captures.map((capture) => (
              <article className="face-log-row" key={capture.id}>
                <div className="face-sequence">#{capture.id}</div>
                <img className="face-fixed-thumb" src={capture.dataUrl} alt={`Captured face ${capture.id}`} />
                <div className="face-log-main">
                  <div className="face-log-title">
                    <strong>Captured face {capture.id}</strong>
                    <span>{Math.round(capture.confidence * 100)}% confidence</span>
                  </div>
                  <div className="face-meta-line">
                    <span><b>Time</b> {capture.capturedAt}</span>
                    <span><b>Source</b> {capture.source}</span>
                    <span><b>Face</b> {capture.width} × {capture.height}px</span>
                    <span><b>Box</b> X {capture.x}, Y {capture.y}</span>
                    <span><b>Frame</b> {capture.frameWidth} × {capture.frameHeight}</span>
                    <span><b>Coverage</b> {capture.coveragePercent.toFixed(2)}%</span>
                    <span><b>Face ratio</b> {capture.faceRatio.toFixed(2)}</span>
                    <span><b>Keypoints</b> {capture.keypoints}</span>
                    <span><b>Brightness</b> {capture.brightnessPercent}%</span>
                    <span><b>Detection</b> {formatMs(capture.detectionMs)}</span>
                    <span><b>Crop</b> {FIXED_CROP_WIDTH} × {FIXED_CROP_HEIGHT}px (1:2)</span>
                  </div>
                </div>
                <a className="crop-download compact" href={capture.dataUrl} download={`face-${capture.id}.jpg`}>Save</a>
              </article>
            ))}
          </div>
        ) : (
          <div className="face-empty">New face captures will be added here line by line.</div>
        )}
      </section>

      <section className="panel controls attribute-note">
        <div className="section-title">
          <span>3</span>
          <div>
            <h2>What metadata can we add?</h2>
            <p>
              Face detection itself gives geometry, confidence and landmarks. The log also derives image-quality information such as brightness and face coverage.
              Estimated age range would require a second attribute model and should be labelled as an estimate, not as FD metadata.
            </p>
          </div>
        </div>
      </section>

      {error ? <div className="error-box">{error}</div> : null}

      <footer className="footer-card">
        <div>
          <strong>Persistent capture</strong>
          <p>A crop is frozen when a new face appears and stays unchanged in the capture history.</p>
        </div>
        <div>
          <strong>Fixed portrait ratio</strong>
          <p>Every saved crop uses the same 1:2 frame so face samples are visually comparable.</p>
        </div>
        <div>
          <strong>Local processing</strong>
          <p>Camera frames, crops and metadata stay in the browser while the page is running.</p>
        </div>
      </footer>
    </main>
  );
}
