import { COCO_CLASSES, loadModel, type LIBREYOLO } from "libreyolo-web";
import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  analyseRelationships,
  type RelationLoadUpdate,
  type SceneRelation,
} from "./relations";
import {
  deduplicateDetections,
  IoUTracker,
  RelationSmoother,
  resolveRelationBoxes,
  type TrackedDetection,
} from "./tracking";

type FacingMode = "environment" | "user";
type SourceMode = "camera" | "image";
type DisplayMode = "focus" | "all";
type DetectionGroup = "Human" | "Vehicle" | "Other";
type RelationState = "idle" | "loading" | "running" | "done" | "error";

type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  label: string;
  group: DetectionGroup;
  trackId?: number;
  trackAge?: number;
};

const MODEL_NAME = "LibreYOLOXs" as const;
const MODEL_INPUT = 640;
const DEFAULT_CONFIDENCE = 0.6;
const DEFAULT_RELATION_THRESHOLD = 0.56;
const DEFAULT_RELATION_CADENCE_MS = 1500;
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 360;
const CAMERA_WARMUP_RUNS = 3;
const LOOP_GAP_MS = 16;
const RELATION_MIN_TRACK_AGE = 2;
const ROAD_VEHICLE_CLASSES = new Set([1, 2, 3, 5, 7]);

function titleCase(value: string) {
  return value.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function mapDetection(classId: number): { label: string; group: DetectionGroup } | null {
  if (classId < 0 || classId >= COCO_CLASSES.length) return null;
  return {
    label: titleCase(COCO_CLASSES[classId]),
    group: classId === 0 ? "Human" : ROAD_VEHICLE_CLASSES.has(classId) ? "Vehicle" : "Other",
  };
}

function formatMs(value: number) {
  if (!value || !Number.isFinite(value)) return "—";
  return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;
}

function relationObjectName(item: SceneRelation["subject"]) {
  return item.trackId === undefined ? item.label : `#${item.trackId} ${item.label}`;
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
  const [relationThreshold, setRelationThreshold] = useState(DEFAULT_RELATION_THRESHOLD);
  const [relationCadenceMs, setRelationCadenceMs] = useState(DEFAULT_RELATION_CADENCE_MS);
  const [running, setRunning] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [liveSceneEnabled, setLiveSceneEnabled] = useState(false);
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
  const [sceneRelations, setSceneRelations] = useState<SceneRelation[]>([]);
  const [relationMs, setRelationMs] = useState(0);
  const [relationState, setRelationState] = useState<RelationState>("idle");
  const [relationStatus, setRelationStatus] = useState(
    "Phase 2 is ready: upload an image or enable Live Scene AI on a camera.",
  );
  const [relationUpdates, setRelationUpdates] = useState(0);
  const [activeTrackCount, setActiveTrackCount] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<LIBREYOLO | null>(null);
  const modelPromiseRef = useRef<Promise<LIBREYOLO> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef("");
  const busyRef = useRef(false);
  const relationBusyRef = useRef(false);
  const loopRef = useRef<number | null>(null);
  const lastStartRef = useRef(0);
  const lastRelationAtRef = useRef(0);
  const timingRef = useRef<number[]>([]);
  const warmupRef = useRef(CAMERA_WARMUP_RUNS);
  const generationRef = useRef(0);
  const trackerRef = useRef(new IoUTracker<Detection>({ iouThreshold: 0.28, maxMisses: 8 }));
  const smootherRef = useRef(new RelationSmoother({ emaWeight: 0.62, maxMisses: 1, maxRelations: 8 }));
  const currentRelationsRef = useRef<SceneRelation[]>([]);

  const humans = useMemo(() => detections.filter((item) => item.group === "Human").length, [detections]);
  const vehicles = useMemo(() => detections.filter((item) => item.group === "Vehicle").length, [detections]);
  const otherObjects = useMemo(() => detections.filter((item) => item.group === "Other").length, [detections]);
  const effectiveFps = averageMs ? 1000 / averageMs : 0;
  const relationBusy = relationState === "loading" || relationState === "running";

  const clearOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const drawFrame = useCallback(
    (items: Detection[], relations: SceneRelation[], width: number, height: number) => {
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
        const trackPrefix = item.trackId === undefined ? "" : `#${item.trackId} `;
        const label = `${trackPrefix}${item.label} ${Math.round(item.confidence * 100)}%`;
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

      context.save();
      context.lineWidth = Math.max(2, width / 600);
      context.font = `700 ${Math.max(11, Math.round(width / 100))}px Arial`;
      context.textBaseline = "middle";

      relations.slice(0, 6).forEach((relation) => {
        const sx = (relation.subject.x1 + relation.subject.x2) / 2;
        const sy = (relation.subject.y1 + relation.subject.y2) / 2;
        const ox = (relation.object.x1 + relation.object.x2) / 2;
        const oy = (relation.object.y1 + relation.object.y2) / 2;
        const angle = Math.atan2(oy - sy, ox - sx);
        const arrowSize = Math.max(8, width / 80);

        context.strokeStyle = "#ffad66";
        context.fillStyle = "#ffad66";
        context.beginPath();
        context.moveTo(sx, sy);
        context.lineTo(ox, oy);
        context.stroke();
        context.beginPath();
        context.moveTo(ox, oy);
        context.lineTo(ox - arrowSize * Math.cos(angle - Math.PI / 6), oy - arrowSize * Math.sin(angle - Math.PI / 6));
        context.lineTo(ox - arrowSize * Math.cos(angle + Math.PI / 6), oy - arrowSize * Math.sin(angle + Math.PI / 6));
        context.closePath();
        context.fill();

        const text = `${relation.predicate} ${Math.round(relation.score * 100)}%`;
        const textWidth = context.measureText(text).width + 10;
        const tx = Math.min(Math.max(2, (sx + ox) / 2 - textWidth / 2), Math.max(2, width - textWidth - 2));
        const ty = Math.min(Math.max(12, (sy + oy) / 2), height - 12);
        context.fillStyle = "rgba(4, 19, 25, 0.9)";
        context.fillRect(tx, ty - 11, textWidth, 22);
        context.fillStyle = "#ffd2ad";
        context.fillText(text, tx + 5, ty);
      });

      context.restore();
    },
    [],
  );

  const clearSceneResults = useCallback((message?: string) => {
    smootherRef.current.reset();
    currentRelationsRef.current = [];
    setSceneRelations([]);
    setRelationMs(0);
    setRelationUpdates(0);
    setRelationState("idle");
    setRelationStatus(message ?? "Phase 2 is ready: upload an image or enable Live Scene AI on a camera.");
  }, []);

  const resetTracking = useCallback(() => {
    trackerRef.current.reset();
    smootherRef.current.reset();
    currentRelationsRef.current = [];
    lastRelationAtRef.current = 0;
    setActiveTrackCount(0);
    setSceneRelations([]);
    setRelationUpdates(0);
  }, []);

  const resetStats = useCallback(
    (warmups = 0) => {
      generationRef.current += 1;
      timingRef.current = [];
      warmupRef.current = warmups;
      setWarmupRemaining(warmups);
      setCurrentMs(0);
      setAverageMs(0);
      setAnalysedFrames(0);
      setDetections([]);
      setRelationMs(0);
      resetTracking();
      clearSceneResults();
      clearOverlay();
    },
    [clearOverlay, clearSceneResults, resetTracking],
  );

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setRunning(false);
    setLiveSceneEnabled(false);
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

    const promise = loadModel(MODEL_NAME, {
      device: ["webgpu", "wasm"],
      modelFamily: "yolox",
      confThres: 0.12,
      iouThres: 0.65,
      maxDet: 40,
      onProgress: (progress) => {
        setModelProgress(Math.max(2, Math.min(99, Math.round(progress * 100))));
      },
    });

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
      const message = caught instanceof Error ? caught.message : "Could not load YOLOX-S.";
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
      generationRef.current += 1;
      if (loopRef.current !== null) cancelAnimationFrame(loopRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      void modelRef.current?.release();
    };
  }, []);

  const runLiveRelations = useCallback(
    async (source: HTMLVideoElement, trackedItems: TrackedDetection<Detection>[]) => {
      const relationCandidates = trackedItems.filter((item) => item.trackAge >= RELATION_MIN_TRACK_AGE);
      if (!liveSceneEnabled || relationBusyRef.current || relationCandidates.length < 2) return;

      const generation = generationRef.current;
      relationBusyRef.current = true;
      setRelationState("loading");
      setRelationStatus("Live Scene AI is preparing the relationship model…");

      const handleUpdate = (update: RelationLoadUpdate) => {
        if (generation !== generationRef.current) return;
        setRelationState(update.stage === "inference" ? "running" : "loading");
        setRelationStatus(
          update.stage === "inference"
            ? "Analysing the latest tracked camera frame…"
            : update.message,
        );
      };

      try {
        const result = await analyseRelationships(source, relationCandidates, relationThreshold, handleUpdate);
        if (generation !== generationRef.current) return;

        const stable = smootherRef.current.update(result.relations);
        currentRelationsRef.current = stable;
        setSceneRelations(stable);
        setRelationMs(result.inferenceMs);
        setRelationUpdates((count) => count + 1);
        setRelationState("done");
        setRelationStatus(
          stable.length
            ? `Live Scene AI active · ${stable.length} stable relationship${stable.length === 1 ? "" : "s"}.`
            : "Live Scene AI active · no relationship crossed the current threshold.",
        );

        const latest = trackerRef.current;
        setActiveTrackCount(latest.activeTrackCount);
      } catch (caught) {
        if (generation !== generationRef.current) return;
        const message = caught instanceof Error ? caught.message : "Live relation inference failed.";
        setRelationState("error");
        setRelationStatus(message);
        setError(`RelateAnything: ${message}`);
        setLiveSceneEnabled(false);
      } finally {
        relationBusyRef.current = false;
      }
    },
    [liveSceneEnabled, relationThreshold],
  );

  const analyseSource = useCallback(
    async (source: HTMLVideoElement | HTMLImageElement, live: boolean): Promise<Detection[]> => {
      if (busyRef.current) return [];

      const width = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
      const height = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
      if (!width || !height) return [];

      busyRef.current = true;

      try {
        const model = await ensureModel();
        const started = performance.now();
        const result = await model.predict(source, {
          confThres: Math.max(0.08, threshold),
          iouThres: 0.65,
          maxDet: 40,
        });
        const elapsed = performance.now() - started;

        const rawItems = result.detections
          .map((item): Detection | null => {
            if (item.confidence < threshold) return null;
            const mapped = mapDetection(item.classId);
            if (!mapped) return null;
            return {
              x1: item.bbox[0],
              y1: item.bbox[1],
              x2: item.bbox[2],
              y2: item.bbox[3],
              confidence: item.confidence,
              label: mapped.label,
              group: mapped.group,
            };
          })
          .filter((item): item is Detection => item !== null);

        const deduplicatedItems = deduplicateDetections(rawItems, 0.45, 0.72);
        const allItems: Detection[] = live
          ? trackerRef.current.update(deduplicatedItems)
          : deduplicatedItems;
        if (live) setActiveTrackCount(trackerRef.current.activeTrackCount);

        const visibleItems = displayMode === "all" ? allItems : allItems.filter((item) => item.group !== "Other");
        const movingRelations = live
          ? resolveRelationBoxes(currentRelationsRef.current, allItems as TrackedDetection<Detection>[])
          : currentRelationsRef.current;

        setDetections(visibleItems);
        setProvider(model.provider || "wasm");
        drawFrame(liveSceneEnabled && live ? [] : visibleItems, movingRelations, width, height);

        if (live && warmupRef.current > 0) {
          warmupRef.current -= 1;
          setWarmupRemaining(warmupRef.current);
          return allItems;
        }

        if (live) {
          timingRef.current = [...timingRef.current.slice(-29), elapsed];
          const mean = timingRef.current.reduce((sum, value) => sum + value, 0) / timingRef.current.length;
          setCurrentMs(elapsed);
          setAverageMs(mean);
          setAnalysedFrames((count) => count + 1);

          if (liveSceneEnabled && allItems.length >= 2 && source instanceof HTMLVideoElement) {
            const now = performance.now();
            if (!relationBusyRef.current && now - lastRelationAtRef.current >= relationCadenceMs) {
              lastRelationAtRef.current = now;
              void runLiveRelations(source, allItems as TrackedDetection<Detection>[]);
            }
          }
        } else {
          timingRef.current = [elapsed];
          setCurrentMs(elapsed);
          setAverageMs(elapsed);
          setAnalysedFrames(1);
        }

        return allItems;
      } catch (caught) {
        setRunning(false);
        setError(caught instanceof Error ? caught.message : "Inference failed.");
        return [];
      } finally {
        busyRef.current = false;
      }
    },
    [
      displayMode,
      drawFrame,
      ensureModel,
      liveSceneEnabled,
      relationCadenceMs,
      runLiveRelations,
      threshold,
    ],
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

  const requestCameraStream = useCallback(async (facingMode: FacingMode) => {
    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: CAMERA_WIDTH },
        height: { ideal: CAMERA_HEIGHT },
        frameRate: { ideal: 30, max: 30 },
      },
    };

    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
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
  }, []);

  const startCamera = useCallback(
    async (facingMode: FacingMode) => {
      stopCamera();
      resetStats(CAMERA_WARMUP_RUNS);
      setSourceMode("camera");
      setSelectedCamera(facingMode);
      setSourceName(facingMode === "environment" ? "Back camera" : "Front camera / Webcam");
      setError("");

      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("Camera access requires HTTPS and a supported browser.");
        }

        await ensureModel();
        const stream = await requestCameraStream(facingMode);
        streamRef.current = stream;

        if (!videoRef.current) throw new Error("Camera viewport is not ready.");
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        syncCameraAspect();

        const track = stream.getVideoTracks()[0];
        const actualFacing = track?.getSettings().facingMode;

        if (facingMode === "environment" && actualFacing && actualFacing !== "environment") {
          setSourceName(track.label || "Available camera");
        }
        if (facingMode === "user" && actualFacing && actualFacing !== "user") {
          setSourceName(track.label || "Webcam");
        }

        setCameraActive(true);
        lastStartRef.current = 0;
        setRunning(true);
        setRelationStatus("Camera tracking is active. Enable Live Scene AI when you want relationship inference.");
      } catch (caught) {
        stopCamera();
        setSelectedCamera(null);
        setError(caught instanceof Error ? caught.message : "Camera permission was not granted.");
      }
    },
    [ensureModel, requestCameraStream, resetStats, stopCamera, syncCameraAspect],
  );

  const toggleLiveScene = useCallback(() => {
    if (!cameraActive) return;

    if (liveSceneEnabled) {
      setLiveSceneEnabled(false);
      smootherRef.current.reset();
      currentRelationsRef.current = [];
      setSceneRelations([]);
      setRelationState("idle");
      setRelationStatus("Live Scene AI paused. Object detection and tracking continue.");
      clearOverlay();
      return;
    }

    generationRef.current += 1;
    smootherRef.current.reset();
    currentRelationsRef.current = [];
    lastRelationAtRef.current = 0;
    setSceneRelations([]);
    setRelationMs(0);
    setRelationUpdates(0);
    setError("");
    setLiveSceneEnabled(true);
    clearOverlay();
    setRelationState("loading");
    setRelationStatus(
      "Live Scene AI enabled. Detector boxes are hidden; only final relationships will be drawn.",
    );
  }, [cameraActive, clearOverlay, liveSceneEnabled]);

  const handleImage = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      stopCamera();
      setSelectedCamera(null);
      resetStats(0);

      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);

      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      setImageUrl(url);
      setSourceMode("image");
      setSourceName(file.name);
      setError("");
      setRelationStatus("Image ready. Detect objects or run scene understanding.");
      event.target.value = "";
    },
    [resetStats, stopCamera],
  );

  const analyseImage = useCallback(async () => {
    if (!imageRef.current || !imageUrl) return;
    resetStats(0);
    await analyseSource(imageRef.current, false);
  }, [analyseSource, imageUrl, resetStats]);

  const analyseScene = useCallback(async () => {
    const image = imageRef.current;
    if (!image || !imageUrl || relationBusyRef.current) return;

    resetStats(0);
    const generation = generationRef.current;
    relationBusyRef.current = true;
    setRelationState("loading");
    setRelationStatus("Running YOLOX-S first so RelateAnything receives real detected boxes…");
    setError("");

    const allItems = await analyseSource(image, false);
    if (generation !== generationRef.current) {
      relationBusyRef.current = false;
      return;
    }

    if (allItems.length < 2) {
      relationBusyRef.current = false;
      setRelationState("done");
      setRelationStatus("Scene understanding needs at least two detected objects. Try a lower detection confidence.");
      return;
    }

    const handleUpdate = (update: RelationLoadUpdate) => {
      if (generation !== generationRef.current) return;
      setRelationState(update.stage === "inference" ? "running" : "loading");
      setRelationStatus(update.message);
    };

    try {
      const result = await analyseRelationships(image, allItems, relationThreshold, handleUpdate);
      if (generation !== generationRef.current) return;

      currentRelationsRef.current = result.relations;
      setSceneRelations(result.relations);
      setRelationMs(result.inferenceMs);
      setRelationUpdates(1);
      setRelationState("done");
      setRelationStatus(
        result.relations.length
          ? `Found ${result.relations.length} ranked relationship${result.relations.length === 1 ? "" : "s"}.`
          : "No relationship crossed the current threshold. Try lowering the relationship confidence.",
      );
      const visibleItems = displayMode === "all" ? allItems : allItems.filter((item) => item.group !== "Other");
      drawFrame([], result.relations, image.naturalWidth, image.naturalHeight);
    } catch (caught) {
      if (generation !== generationRef.current) return;
      const message = caught instanceof Error ? caught.message : "Scene-understanding inference failed.";
      setRelationState("error");
      setRelationStatus(message);
      setError(`RelateAnything: ${message}`);
    } finally {
      relationBusyRef.current = false;
    }
  }, [analyseSource, displayMode, drawFrame, imageUrl, relationThreshold, resetStats]);

  const changeDisplayMode = useCallback(
    (mode: DisplayMode) => {
      setDisplayMode(mode);
      resetTracking();
      clearOverlay();
      setRelationStatus(
        sourceMode === "camera"
          ? "Display filter changed. Tracking has restarted and Live Scene AI will rebuild relationships."
          : "Display filter changed. Run the image analysis again to refresh the overlay.",
      );
    },
    [clearOverlay, resetTracking, sourceMode],
  );

  const changeDetectionThreshold = useCallback(
    (value: number) => {
      setThreshold(value);
      resetTracking();
      setRelationMs(0);
      setRelationState("idle");
      setRelationStatus(
        sourceMode === "camera"
          ? "Detection confidence changed. Tracking and live relationships will rebuild automatically."
          : "Detection confidence changed. Run scene understanding again for fresh relationships.",
      );
    },
    [resetTracking, sourceMode],
  );

  const changeRelationThreshold = useCallback(
    (value: number) => {
      setRelationThreshold(value);
      smootherRef.current.reset();
      currentRelationsRef.current = [];
      setSceneRelations([]);
      setRelationMs(0);
      setRelationUpdates(0);
      setRelationState("idle");
      lastRelationAtRef.current = 0;
      setRelationStatus(
        sourceMode === "camera" && liveSceneEnabled
          ? "Relationship confidence changed. The next live relation pass will use the new threshold."
          : "Relationship confidence changed. Run scene understanding again to apply it.",
      );
    },
    [liveSceneEnabled, sourceMode],
  );

  const changeRelationCadence = useCallback((value: number) => {
    setRelationCadenceMs(value);
    lastRelationAtRef.current = 0;
    setRelationStatus(`Live relation cadence set to ${(value / 1000).toFixed(1)} seconds.`);
  }, []);

  return (
    <main className="page-shell">
      <header className="hero">
        <p className="eyebrow">CCTV AI browser lab</p>
        <h1>Object Detection + Live Scene Understanding</h1>
        <p className="intro">
          Phase 2 keeps YOLOX-S detection running continuously, assigns lightweight track IDs, and runs RelateAnything
          at a controlled cadence so visual relationships can persist across live camera frames instead of flickering
          frame by frame.
        </p>
        <div className="badges">
          <span>YOLOX-S · {MODEL_INPUT}px</span>
          <span>{provider.toUpperCase()}</span>
          <span>RelateAnything · Phase 2</span>
          <span>IoU tracking + relation smoothing</span>
          <span>{displayMode === "focus" ? "Human + Vehicle view" : "All COCO · 80 classes"}</span>
        </div>
      </header>

      <section className="panel viewport-panel">
        <div className="panel-head">
          <div>
            <h2>AI viewport</h2>
            <p>{sourceName}</p>
          </div>
          <div className={`live-state ${running || relationBusy ? "active" : ""}`}>
            <i />
            {sourceMode === "camera" && liveSceneEnabled && relationBusy
              ? "Live scene analysis"
              : modelState === "loading"
                ? `Loading detector ${modelProgress}%`
                : running
                  ? liveSceneEnabled
                    ? "Detection + tracking + scene AI"
                    : "Detection + tracking"
                  : sourceMode === "image" && relationBusy
                    ? "Understanding scene"
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
              <img ref={imageRef} src={imageUrl} alt="Selected for AI analysis" className="media" />
              <canvas ref={canvasRef} className="overlay" />
            </div>
          ) : (
            <div className="empty-state image-empty">Choose an image below.</div>
          )}
        </div>

        <div className="metrics">
          <Metric label="Detections" value={detections.length} />
          {sourceMode === "camera" ? (
            <>
              <Metric label="Active tracks" value={activeTrackCount} />
              <Metric label="Relationships" value={sceneRelations.length || "—"} />
              <Metric label="Relation updates" value={relationUpdates || "—"} />
              <Metric label="Detector avg" value={formatMs(averageMs)} />
              <Metric label="Relation time" value={formatMs(relationMs)} />
              <Metric label="Detector rate" value={effectiveFps ? `${effectiveFps.toFixed(1)} FPS` : "—"} />
            </>
          ) : (
            <>
              <Metric label="Humans" value={humans} />
              <Metric label="Vehicles" value={vehicles} />
              {displayMode === "all" ? <Metric label="Other objects" value={otherObjects} /> : null}
              <Metric label="Detection time" value={formatMs(currentMs)} />
              <Metric label="Relationships" value={sceneRelations.length || "—"} />
              <Metric label="Relation time" value={formatMs(relationMs)} />
            </>
          )}
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>1</span>
          <div>
            <h2>Choose source</h2>
            <p>Use live camera tracking or upload a still image. Both can now use RelateAnything.</p>
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
            <button onClick={() => void analyseImage()} disabled={!imageUrl || modelState !== "ready" || relationBusy}>
              Detect objects
            </button>
          ) : cameraActive ? (
            <button onClick={() => setRunning((value) => !value)}>{running ? "Pause detector" : "Resume detector"}</button>
          ) : null}

          {sourceMode === "image" ? (
            <button
              className="scene-action"
              onClick={() => void analyseScene()}
              disabled={!imageUrl || modelState !== "ready" || relationBusy}
            >
              {relationBusy ? "Working…" : "Understand scene"}
            </button>
          ) : cameraActive ? (
            <button className={liveSceneEnabled ? "scene-action active-scene" : "scene-action"} onClick={toggleLiveScene}>
              {liveSceneEnabled ? "Stop Live Scene AI" : "Start Live Scene AI"}
            </button>
          ) : null}
        </div>

        <p className="source-note">
          Live relation inference is intentionally slower than the detector. When Live Scene AI is enabled, detector boxes
          are hidden from the viewport, duplicate detections are suppressed, and only the strongest eight objects are offered
          to RelateAnything. Tracking still runs internally so final relationship arrows can follow moving objects.
        </p>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>2</span>
          <div>
            <h2>Detection set</h2>
            <p>The display can focus on humans and road vehicles. Scene understanding still receives all detected COCO objects.</p>
          </div>
        </div>
        <div className="source-grid">
          <button
            className={displayMode === "all" ? "primary" : undefined}
            onClick={() => changeDisplayMode("all")}
          >
            All COCO objects (80)
          </button>
          <button
            className={displayMode === "focus" ? "primary" : undefined}
            onClick={() => changeDisplayMode("focus")}
          >
            Human + Vehicle view
          </button>
        </div>
      </section>

      <section className="panel controls">
        <div className="section-title">
          <span>3</span>
          <div>
            <h2>Detection confidence</h2>
            <p>Default is 60%. Live track IDs are rebuilt when this threshold changes.</p>
          </div>
        </div>
        <div className="slider-row">
          <input
            type="range"
            min="10"
            max="80"
            step="1"
            value={Math.round(threshold * 100)}
            onChange={(event) => changeDetectionThreshold(Number(event.target.value) / 100)}
          />
          <strong>{Math.round(threshold * 100)}%</strong>
        </div>
      </section>

      <section className="panel controls scene-panel">
        <div className="section-title">
          <span>4</span>
          <div>
            <h2>Scene understanding · Phase 2</h2>
            <p>Clean camera frame + deduplicated tracked boxes → RelateAnything → final relationship overlay only.</p>
          </div>
        </div>

        <div className={`scene-status ${relationBusy ? "active" : relationState === "error" ? "error" : ""}`}>
          <strong>
            {relationBusy
              ? "Working"
              : sourceMode === "camera" && liveSceneEnabled
                ? "Live"
                : relationState === "done"
                  ? "Result"
                  : relationState === "error"
                    ? "Error"
                    : "Ready"}
          </strong>
          <span>{relationStatus}</span>
        </div>

        {sourceMode === "camera" ? (
          <div className="cadence-block">
            <div>
              <strong>Live relation cadence</strong>
              <small>
                Faster updates feel more live but increase CPU/GPU load. Balanced is the recommended starting point on phones.
              </small>
            </div>
            <div className="cadence-grid">
              {[1000, 1500, 3000].map((value) => (
                <button
                  key={value}
                  className={relationCadenceMs === value ? "primary" : undefined}
                  onClick={() => changeRelationCadence(value)}
                >
                  {value === 1000 ? "Fast · 1.0s" : value === 1500 ? "Balanced · 1.5s" : "Light · 3.0s"}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        <div className="relation-threshold">
          <div>
            <strong>Relationship confidence</strong>
            <small>Released calibrated relation score. Lower values show more relationships but can become noisy.</small>
          </div>
          <div className="slider-row compact">
            <input
              type="range"
              min="30"
              max="90"
              step="1"
              value={Math.round(relationThreshold * 100)}
              onChange={(event) => changeRelationThreshold(Number(event.target.value) / 100)}
            />
            <strong>{Math.round(relationThreshold * 100)}%</strong>
          </div>
        </div>

        {sceneRelations.length ? (
          <div className="relation-list">
            {sceneRelations.map((relation, index) => (
              <div
                className="relation-row"
                key={`${relation.subject.trackId ?? relation.subject.label}-${relation.predicate}-${relation.object.trackId ?? relation.object.label}-${index}`}
              >
                <div className="relation-chain">
                  <strong>{relationObjectName(relation.subject)}</strong>
                  <span>→ {relation.predicate} →</span>
                  <strong>{relationObjectName(relation.object)}</strong>
                </div>
                <span className="relation-score">{Math.round(relation.score * 100)}%</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-relations">
            {sourceMode === "camera"
              ? cameraActive
                ? liveSceneEnabled
                  ? "Tracking is active. Relationships will appear after the next RelateAnything pass."
                  : "Press “Start Live Scene AI” to add live relationships on top of object tracking."
                : "Start a camera to use Phase 2 live scene understanding."
              : imageUrl
                ? "Press “Understand scene” to detect objects and infer relationships."
                : "Upload an image to begin."}
          </div>
        )}

        <div className="phase-flow">
          <span>Detector</span>
          <i>→</i>
          <span>Track IDs</span>
          <i>→</i>
          <span>RelateAnything</span>
          <i>→</i>
          <span>Temporal smoothing</span>
        </div>

        <p className="scene-note">
          The detector boxes are now internal guidance only and are hidden while Live Scene AI is active. RelateAnything
          receives the clean camera pixels plus deduplicated box coordinates, not a frame with yellow rectangles painted on it.
          Only stable tracks are considered for live relation passes, and at most eight high-confidence objects are evaluated.
        </p>
      </section>

      {error ? <div className="error-box">{error}</div> : null}

      <footer className="footer-card">
        <div>
          <strong>Continuous detection</strong>
          <p>YOLOX-S keeps running at 640 × 640 while lightweight IoU tracking assigns short-lived object IDs.</p>
        </div>
        <div>
          <strong>Periodic relation AI</strong>
          <p>RelateAnything runs every 1–3 seconds instead of every detector frame, reducing live browser load.</p>
        </div>
        <div>
          <strong>Local media</strong>
          <p>Camera frames and uploaded images are analysed in-browser; no image inference API was added.</p>
        </div>
      </footer>

      {sourceMode === "camera" && warmupRemaining > 0 && cameraActive ? (
        <div className="warmup-note">Warming up… {warmupRemaining} pass{warmupRemaining === 1 ? "" : "es"} left</div>
      ) : null}
    </main>
  );
}
