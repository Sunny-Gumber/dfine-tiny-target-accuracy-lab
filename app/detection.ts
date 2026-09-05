export type SourceMode = "image" | "video" | "camera";
export type ModelKey = "dfine-s" | "dfine-n" | "rtdetr-r18";
export type ScanMode = "fast" | "precision";
export type DetectionFilter = "both" | "human" | "vehicle";
export type Backend = "webgpu" | "wasm";

export type Box = {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
};

export type ModelOutput = {
  score: number;
  label: string;
  box: Box;
};

export type Detection = ModelOutput & {
  id: number;
  kind: "human" | "vehicle";
  origin: string;
};

export type Detector = {
  (
    input: HTMLCanvasElement,
    options?: Record<string, unknown>,
  ): Promise<ModelOutput[]>;
  dispose?: () => Promise<void> | void;
};

export type ModelProgress = {
  status?: string;
  file?: string;
  progress?: number;
};

type TransformersRuntime = {
  pipeline: (
    task: string,
    model: string,
    options: Record<string, unknown>,
  ) => Promise<Detector>;
  env?: {
    allowLocalModels?: boolean;
    allowRemoteModels?: boolean;
  };
};

export const MODELS = {
  "dfine-s": {
    name: "D-FINE-S",
    id: "onnx-community/dfine_s_coco-ONNX",
    note: "Accuracy first · 10M parameters · 48.5 COCO AP",
    badge: "Recommended",
  },
  "dfine-n": {
    name: "D-FINE-N",
    id: "onnx-community/dfine_n_coco-ONNX",
    note: "Faster D-FINE · 4M parameters · 42.8 COCO AP",
    badge: "Faster",
  },
  "rtdetr-r18": {
    name: "RT-DETRv2-R18",
    id: "onnx-community/rtdetr_v2_r18vd-ONNX",
    note: "Mobile fallback · deployment-safe DETR · quantized WASM",
    badge: "Mobile",
  },
} as const;

export const MOBILE_MODEL_KEY: ModelKey = "rtdetr-r18";

const HUMAN_LABELS = new Set(["person"]);
const VEHICLE_LABELS = new Set([
  "bicycle",
  "car",
  "motorcycle",
  "bus",
  "truck",
]);

const CDN_MODULE =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

export function isMobileDevice(): boolean {
  return (
    typeof navigator !== "undefined" &&
    /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)
  );
}

export function getPreferredBackend(): Backend {
  if (typeof navigator === "undefined") return "wasm";

  if (!isMobileDevice() && "gpu" in navigator) {
    return "webgpu";
  }
  return "wasm";
}

export function isBackendCompatibilityError(error: unknown): boolean {
  const message =
    error instanceof Error ? error.message.toLowerCase() : String(error).toLowerCase();

  return (
    /ceil\(\).*shape computation.*maxpool/.test(message) ||
    /maxpool.*not yet supported/.test(message) ||
    /unsupported.*maxpool/.test(message) ||
    /webgpu.*(unsupported|not supported|failed)/.test(message) ||
    /no available backend found.*webgpu/.test(message)
  );
}

export async function createDetector(
  modelKey: ModelKey,
  backend: Backend,
  onProgress: (progress: ModelProgress) => void,
): Promise<Detector> {
  const moduleUrl: string = CDN_MODULE;
  const runtime = (await import(
    /* @vite-ignore */ moduleUrl
  )) as unknown as TransformersRuntime;

  if (runtime.env) {
    runtime.env.allowLocalModels = false;
    runtime.env.allowRemoteModels = true;
  }

  return runtime.pipeline("object-detection", MODELS[modelKey].id, {
    device: backend,
    dtype: backend === "wasm" ? "q8" : "fp32",
    progress_callback: onProgress,
  });
}

function normalizeLabel(label: string): string {
  return label.toLowerCase().trim().replaceAll("_", " ");
}

export function getKind(label: string): Detection["kind"] | null {
  const normalized = normalizeLabel(label);
  if (HUMAN_LABELS.has(normalized)) return "human";
  if (VEHICLE_LABELS.has(normalized)) return "vehicle";
  return null;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function boxArea(box: Box): number {
  return Math.max(0, box.xmax - box.xmin) * Math.max(0, box.ymax - box.ymin);
}

function intersectionArea(a: Box, b: Box): number {
  const width = Math.max(0, Math.min(a.xmax, b.xmax) - Math.max(a.xmin, b.xmin));
  const height = Math.max(
    0,
    Math.min(a.ymax, b.ymax) - Math.max(a.ymin, b.ymin),
  );
  return width * height;
}

function duplicateOverlap(a: Box, b: Box): boolean {
  const intersection = intersectionArea(a, b);
  if (!intersection) return false;
  const aArea = boxArea(a);
  const bArea = boxArea(b);
  const union = aArea + bArea - intersection;
  const iou = union ? intersection / union : 0;
  const containment = intersection / Math.max(1, Math.min(aArea, bArea));
  return iou >= 0.46 || containment >= 0.82;
}

type Region = {
  x: number;
  y: number;
  width: number;
  height: number;
  name: string;
};

function scanRegions(
  width: number,
  height: number,
  mode: ScanMode,
): Region[] {
  const full = { x: 0, y: 0, width, height, name: "Full frame" };
  if (mode === "fast") return [full];

  const tileWidth = Math.ceil(width * 0.56);
  const tileHeight = Math.ceil(height * 0.56);
  return [
    full,
    { x: 0, y: 0, width: tileWidth, height: tileHeight, name: "Top left" },
    {
      x: width - tileWidth,
      y: 0,
      width: tileWidth,
      height: tileHeight,
      name: "Top right",
    },
    {
      x: 0,
      y: height - tileHeight,
      width: tileWidth,
      height: tileHeight,
      name: "Bottom left",
    },
    {
      x: width - tileWidth,
      y: height - tileHeight,
      width: tileWidth,
      height: tileHeight,
      name: "Bottom right",
    },
  ];
}

function cropFrame(frame: HTMLCanvasElement, region: Region): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = region.width;
  canvas.height = region.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("This browser could not prepare the AI frame.");
  context.drawImage(
    frame,
    region.x,
    region.y,
    region.width,
    region.height,
    0,
    0,
    region.width,
    region.height,
  );
  return canvas;
}

export async function detectFrame(args: {
  detector: Detector;
  frame: HTMLCanvasElement;
  mode: ScanMode;
  filter: DetectionFilter;
  threshold: number;
  onPass: (completed: number, total: number, name: string) => void;
}): Promise<{
  detections: Detection[];
  passCount: number;
  fullFrameCount: number;
}> {
  const { detector, frame, mode, filter, threshold, onPass } = args;
  const regions = scanRegions(frame.width, frame.height, mode);
  const candidates: Detection[] = [];
  let fullFrameCount = 0;

  for (let index = 0; index < regions.length; index += 1) {
    const region = regions[index];
    onPass(index, regions.length, region.name);
    const input = cropFrame(frame, region);
    const outputs = await detector(input, {
      threshold: Math.min(0.05, threshold),
      percentage: false,
    });

    let acceptedInPass = 0;
    for (const output of outputs) {
      const kind = getKind(output.label);
      if (!kind || output.score < threshold) continue;
      if (filter !== "both" && filter !== kind) continue;

      const box = {
        xmin: clamp(output.box.xmin + region.x, 0, frame.width),
        ymin: clamp(output.box.ymin + region.y, 0, frame.height),
        xmax: clamp(output.box.xmax + region.x, 0, frame.width),
        ymax: clamp(output.box.ymax + region.y, 0, frame.height),
      };
      if (boxArea(box) < 16) continue;

      candidates.push({
        id: 0,
        score: output.score,
        label: normalizeLabel(output.label),
        box,
        kind,
        origin: region.name,
      });
      acceptedInPass += 1;
    }
    if (index === 0) fullFrameCount = acceptedInPass;
    onPass(index + 1, regions.length, region.name);
  }

  const merged: Detection[] = [];
  const sorted = [...candidates].sort((a, b) => b.score - a.score);
  for (const candidate of sorted) {
    const isDuplicate = merged.some(
      (kept) =>
        kept.kind === candidate.kind &&
        duplicateOverlap(kept.box, candidate.box),
    );
    if (!isDuplicate) merged.push(candidate);
  }

  return {
    detections: merged.map((detection, index) => ({
      ...detection,
      id: index + 1,
    })),
    passCount: regions.length,
    fullFrameCount,
  };
}

export function countAgreement(
  predictedHumans: number,
  predictedVehicles: number,
  actualHumans: number | null,
  actualVehicles: number | null,
): number | null {
  if (actualHumans === null || actualVehicles === null) return null;
  const error =
    Math.abs(predictedHumans - actualHumans) +
    Math.abs(predictedVehicles - actualVehicles);
  const actualTotal = actualHumans + actualVehicles;
  if (actualTotal === 0) return error === 0 ? 100 : 0;
  return Math.max(0, Math.round((1 - error / actualTotal) * 100));
}
