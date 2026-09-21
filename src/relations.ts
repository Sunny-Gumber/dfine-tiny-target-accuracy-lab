import * as ort from "onnxruntime-web";

export type RelationSource = HTMLImageElement | HTMLVideoElement | HTMLCanvasElement;

export type RelationDetection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  label: string;
  trackId?: number;
  trackAge?: number;
};

export type SceneRelation = {
  subject: RelationDetection;
  predicate: string;
  object: RelationDetection;
  score: number;
  rankScore: number;
};

export type RelationLoadUpdate = {
  stage: "bank" | "model" | "inference";
  message: string;
};

type PredicateBank = {
  predicates: string[];
  W: Float32Array;
  alpha: Float32Array;
};

type ParsedNpy = {
  shape: number[];
  data: Float32Array;
};

const MODEL_URL = "https://huggingface.co/maelic/relsgg-vits16plus/resolve/main/relateanything.onnx";
const BANK_URL =
  "https://cdn.jsdelivr.net/gh/Maelic/RelateAnything@main/deploy/dist/relsgg-vits16plus/predicate_bank.npz";
const MODEL_SIZE = 448;
const MAX_BOXES = 32;
const MAX_RELATION_OBJECTS = 8;
const TEXT_DIM = 512;
const CALIBRATION_A = 0.5651;
const CALIBRATION_B = -1.9623;
const PAIR_WEIGHT = 1.0;
const MAX_RELATIONS = 12;

// build_predicate_bank.py writes PREDICATE_VOCAB first and only then appends
// additional trained predicates. The first rows of W/alpha therefore follow
// this exact order.
const DEFAULT_BANK_PREDICATES = [
  "wearing",
  "riding",
  "playing",
  "sitting on",
  "sitting at",
  "holding",
  "sitting in",
  "looking at",
  "using",
  "watching",
  "standing on",
  "carrying",
  "talking to",
  "smiling at",
  "standing beside",
  "walking past",
  "posing with",
  "leaning against",
  "part of",
  "resting on",
  "on",
  "covering",
  "inside",
  "on top of",
  "contained in",
  "hanging from",
  "surrounding",
  "attached to",
  "in front of",
  "beside",
  "to the left of",
  "to the right of",
  "behind",
  "above",
  "below",
] as const;

// A compact CCTV-oriented vocabulary keeps live output understandable and
// reduces the host-side score matrix. Every string is in the released bank.
const CCTV_PREDICATES = [
  "wearing",
  "riding",
  "sitting on",
  "holding",
  "looking at",
  "using",
  "standing on",
  "carrying",
  "standing beside",
  "walking past",
  "leaning against",
  "resting on",
  "on",
  "covering",
  "inside",
  "attached to",
  "in front of",
  "beside",
  "behind",
  "above",
  "below",
] as const;

let bankPromise: Promise<PredicateBank> | null = null;
let sessionPromise: Promise<ort.InferenceSession> | null = null;

function sigmoid(value: number) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

function findEndOfCentralDirectory(view: DataView) {
  const minOffset = Math.max(0, view.byteLength - 65_557);
  for (let offset = view.byteLength - 22; offset >= minOffset; offset -= 1) {
    if (view.getUint32(offset, true) === 0x06054b50) return offset;
  }
  throw new Error("Could not read the RelateAnything predicate bank ZIP directory.");
}

async function unzipEntry(buffer: ArrayBuffer, wantedName: string) {
  const view = new DataView(buffer);
  const eocd = findEndOfCentralDirectory(view);
  const totalEntries = view.getUint16(eocd + 10, true);
  let offset = view.getUint32(eocd + 16, true);
  const decoder = new TextDecoder();

  for (let index = 0; index < totalEntries; index += 1) {
    if (view.getUint32(offset, true) !== 0x02014b50) {
      throw new Error("Predicate bank ZIP directory is malformed.");
    }

    const method = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const fileNameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const localHeaderOffset = view.getUint32(offset + 42, true);
    const fileName = decoder.decode(new Uint8Array(buffer, offset + 46, fileNameLength));

    if (fileName === wantedName) {
      if (view.getUint32(localHeaderOffset, true) !== 0x04034b50) {
        throw new Error(`Predicate bank entry ${wantedName} has a malformed local header.`);
      }
      const localNameLength = view.getUint16(localHeaderOffset + 26, true);
      const localExtraLength = view.getUint16(localHeaderOffset + 28, true);
      const dataOffset = localHeaderOffset + 30 + localNameLength + localExtraLength;
      const compressed = buffer.slice(dataOffset, dataOffset + compressedSize);

      if (method === 0) return compressed;
      if (method !== 8) throw new Error(`Unsupported predicate bank compression method: ${method}.`);
      if (typeof DecompressionStream === "undefined") {
        throw new Error("This browser cannot decompress the RelateAnything predicate bank. Use a current Chrome/Edge browser.");
      }

      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
      return await new Response(stream).arrayBuffer();
    }

    offset += 46 + fileNameLength + extraLength + commentLength;
  }

  throw new Error(`Predicate bank entry ${wantedName} was not found.`);
}

function parseFloat32Npy(buffer: ArrayBuffer): ParsedNpy {
  const bytes = new Uint8Array(buffer);
  if (
    bytes.length < 12 ||
    bytes[0] !== 0x93 ||
    String.fromCharCode(...bytes.slice(1, 6)) !== "NUMPY"
  ) {
    throw new Error("Predicate bank contains an invalid NPY array.");
  }

  const major = bytes[6];
  const view = new DataView(buffer);
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(bytes.slice(headerStart, headerStart + headerLength));
  const descr = header.match(/'descr':\s*'([^']+)'/)?.[1];
  const fortran = header.match(/'fortran_order':\s*(True|False)/)?.[1];
  const shapeText = header.match(/'shape':\s*\(([^)]*)\)/)?.[1];

  if (descr !== "<f4" && descr !== "|f4" && descr !== "=f4") {
    throw new Error(`Expected float32 predicate-bank data, got ${descr ?? "unknown"}.`);
  }
  if (fortran === "True") throw new Error("Fortran-order predicate-bank arrays are not supported.");
  if (!shapeText) throw new Error("Predicate-bank NPY shape is missing.");

  const shape = shapeText
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);
  const count = shape.reduce((product, value) => product * value, 1);
  const dataStart = headerStart + headerLength;
  const dataEnd = dataStart + count * 4;
  if (dataEnd > buffer.byteLength) throw new Error("Predicate-bank NPY payload is truncated.");

  return { shape, data: new Float32Array(buffer.slice(dataStart, dataEnd)) };
}

async function loadPredicateBank(onUpdate?: (update: RelationLoadUpdate) => void) {
  if (bankPromise) return bankPromise;

  bankPromise = (async () => {
    onUpdate?.({ stage: "bank", message: "Loading CCTV relationship vocabulary…" });
    const response = await fetch(BANK_URL, { mode: "cors" });
    if (!response.ok) throw new Error(`Could not load predicate bank (${response.status}).`);
    const zip = await response.arrayBuffer();
    const [wBuffer, alphaBuffer] = await Promise.all([unzipEntry(zip, "W.npy"), unzipEntry(zip, "alpha.npy")]);
    const w = parseFloat32Npy(wBuffer);
    const alpha = parseFloat32Npy(alphaBuffer);

    if (w.shape.length !== 2 || w.shape[1] !== TEXT_DIM) {
      throw new Error(`Unexpected predicate embedding shape: ${w.shape.join(" × ")}.`);
    }
    if (w.shape[0] < DEFAULT_BANK_PREDICATES.length || alpha.data.length < DEFAULT_BANK_PREDICATES.length) {
      throw new Error("Released predicate bank is smaller than expected.");
    }

    const selectedIndices = CCTV_PREDICATES.map((name) => {
      const index = DEFAULT_BANK_PREDICATES.indexOf(name);
      if (index < 0) throw new Error(`Predicate ${name} is not in the released default bank.`);
      return index;
    });
    const W = new Float32Array(selectedIndices.length * TEXT_DIM);
    const selectedAlpha = new Float32Array(selectedIndices.length);

    selectedIndices.forEach((sourceIndex, targetIndex) => {
      W.set(w.data.subarray(sourceIndex * TEXT_DIM, (sourceIndex + 1) * TEXT_DIM), targetIndex * TEXT_DIM);
      selectedAlpha[targetIndex] = alpha.data[sourceIndex];
    });

    return { predicates: [...CCTV_PREDICATES], W, alpha: selectedAlpha };
  })().catch((error) => {
    bankPromise = null;
    throw error;
  });

  return bankPromise;
}

async function loadRelationSession(onUpdate?: (update: RelationLoadUpdate) => void) {
  if (sessionPromise) return sessionPromise;

  sessionPromise = (async () => {
    onUpdate?.({
      stage: "model",
      message: "Loading RelateAnything model… first run downloads a large ONNX file.",
    });
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    return await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  })().catch((error) => {
    sessionPromise = null;
    throw error;
  });

  return sessionPromise;
}

function sourceDimensions(source: RelationSource) {
  if (source instanceof HTMLVideoElement) {
    return { width: source.videoWidth, height: source.videoHeight };
  }
  if (source instanceof HTMLImageElement) {
    return { width: source.naturalWidth, height: source.naturalHeight };
  }
  return { width: source.width, height: source.height };
}

function makeImageTensor(source: RelationSource) {
  const canvas = document.createElement("canvas");
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Could not create an image preprocessing canvas.");
  context.drawImage(source, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const pixels = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const plane = MODEL_SIZE * MODEL_SIZE;
  const chw = new Float32Array(plane * 3);

  for (let index = 0; index < plane; index += 1) {
    const rgba = index * 4;
    chw[index] = pixels[rgba] / 255;
    chw[plane + index] = pixels[rgba + 1] / 255;
    chw[plane * 2 + index] = pixels[rgba + 2] / 255;
  }

  return new ort.Tensor("float32", chw, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function makeBoxesTensor(source: RelationSource, detections: RelationDetection[]) {
  const { width, height } = sourceDimensions(source);
  const padded = new Float32Array(MAX_BOXES * 4);

  detections.forEach((item, index) => {
    const x1 = item.x1 / Math.max(width, 1);
    const y1 = item.y1 / Math.max(height, 1);
    const x2 = item.x2 / Math.max(width, 1);
    const y2 = item.y2 / Math.max(height, 1);
    padded[index * 4] = (x1 + x2) / 2;
    padded[index * 4 + 1] = (y1 + y2) / 2;
    padded[index * 4 + 2] = x2 - x1;
    padded[index * 4 + 3] = y2 - y1;
  });

  return new ort.Tensor("float32", padded, [1, MAX_BOXES, 4]);
}

function boolAt(data: Uint8Array, index: number) {
  return Boolean(data[index]);
}

export async function analyseRelationships(
  source: RelationSource,
  detections: RelationDetection[],
  threshold: number,
  onUpdate?: (update: RelationLoadUpdate) => void,
) {
  const selectedDetections = [...detections]
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, MAX_RELATION_OBJECTS);

  if (selectedDetections.length < 2) {
    return { relations: [] as SceneRelation[], inferenceMs: 0, usedDetections: selectedDetections };
  }

  const dimensions = sourceDimensions(source);
  if (!dimensions.width || !dimensions.height) {
    throw new Error("Source frame is not ready for relation inference.");
  }

  const [bank, session] = await Promise.all([loadPredicateBank(onUpdate), loadRelationSession(onUpdate)]);
  onUpdate?.({ stage: "inference", message: "Finding relationships between detected objects…" });

  const feeds: Record<string, ort.Tensor> = {
    image: makeImageTensor(source),
    boxes: makeBoxesTensor(source, selectedDetections),
    box_counts: new ort.Tensor("int64", new BigInt64Array([BigInt(selectedDetections.length)]), [1]),
    W: new ort.Tensor("float32", bank.W, [bank.predicates.length, TEXT_DIM]),
    alpha: new ort.Tensor("float32", bank.alpha, [bank.predicates.length]),
  };

  const started = performance.now();
  const output = await session.run(feeds);
  const inferenceMs = performance.now() - started;

  const predTensor = output.pred_logits;
  const pairTensor = output.pair_logits;
  const subTensor = output.sub_idx;
  const objTensor = output.obj_idx;
  const validTensor = output.valid_mask;
  if (!predTensor || !pairTensor || !subTensor || !objTensor || !validTensor) {
    throw new Error("RelateAnything returned an unexpected output signature.");
  }

  const pred = predTensor.data as Float32Array;
  const pair = pairTensor.data as Float32Array;
  const sub = subTensor.data as BigInt64Array;
  const obj = objTensor.data as BigInt64Array;
  const valid = validTensor.data as unknown as Uint8Array;
  const pairCount = pairTensor.dims.at(-1) ?? pair.length;
  const predicateCount = bank.predicates.length;
  const candidates: SceneRelation[] = [];

  for (let pairIndex = 0; pairIndex < pairCount; pairIndex += 1) {
    if (!boolAt(valid, pairIndex)) continue;
    const subjectIndex = Number(sub[pairIndex]);
    const objectIndex = Number(obj[pairIndex]);
    if (
      subjectIndex < 0 ||
      objectIndex < 0 ||
      subjectIndex >= selectedDetections.length ||
      objectIndex >= selectedDetections.length ||
      subjectIndex === objectIndex
    ) {
      continue;
    }

    let bestPredicate = -1;
    let bestScore = -Infinity;
    const rowOffset = pairIndex * predicateCount;
    for (let predicateIndex = 0; predicateIndex < predicateCount; predicateIndex += 1) {
      const score = sigmoid(
        CALIBRATION_A * (pred[rowOffset + predicateIndex] + PAIR_WEIGHT * pair[pairIndex]) + CALIBRATION_B,
      );
      if (score > bestScore) {
        bestScore = score;
        bestPredicate = predicateIndex;
      }
    }

    if (bestPredicate < 0 || bestScore < threshold) continue;
    const subject = selectedDetections[subjectIndex];
    const object = selectedDetections[objectIndex];
    candidates.push({
      subject,
      predicate: bank.predicates[bestPredicate],
      object,
      score: bestScore,
      rankScore: bestScore * subject.confidence * object.confidence,
    });
  }

  candidates.sort((left, right) => right.rankScore - left.rankScore);
  return {
    relations: candidates.slice(0, MAX_RELATIONS),
    inferenceMs,
    usedDetections: selectedDetections,
  };
}
