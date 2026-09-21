import type { RelationDetection, SceneRelation } from "./relations";

export type TrackableDetection = RelationDetection & {
  group?: string;
};

export type TrackedDetection<T extends TrackableDetection = TrackableDetection> = T & {
  trackId: number;
  trackAge: number;
};

type InternalTrack<T extends TrackableDetection> = {
  id: number;
  detection: T;
  age: number;
  misses: number;
};

export type TrackerOptions = {
  iouThreshold?: number;
  maxMisses?: number;
};

function area(box: TrackableDetection) {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

export function intersectionOverUnion(a: TrackableDetection, b: TrackableDetection) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (!intersection) return 0;
  const union = area(a) + area(b) - intersection;
  return union > 0 ? intersection / union : 0;
}

export class IoUTracker<T extends TrackableDetection> {
  private readonly iouThreshold: number;
  private readonly maxMisses: number;
  private tracks: InternalTrack<T>[] = [];
  private nextId = 1;

  constructor(options: TrackerOptions = {}) {
    this.iouThreshold = options.iouThreshold ?? 0.28;
    this.maxMisses = options.maxMisses ?? 8;
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
  }

  update(detections: T[]): TrackedDetection<T>[] {
    const unmatchedTracks = new Set(this.tracks.map((_, index) => index));
    const assignments = new Map<number, number>();

    // Match high-confidence detections first. Label agreement plus IoU keeps
    // IDs stable enough for a lightweight browser tracker without adding a
    // second neural network.
    const detectionOrder = detections
      .map((detection, index) => ({ detection, index }))
      .sort((left, right) => right.detection.confidence - left.detection.confidence);

    for (const { detection, index: detectionIndex } of detectionOrder) {
      let bestTrackIndex = -1;
      let bestIou = this.iouThreshold;

      unmatchedTracks.forEach((trackIndex) => {
        const track = this.tracks[trackIndex];
        if (track.detection.label !== detection.label) return;
        const iou = intersectionOverUnion(track.detection, detection);
        if (iou > bestIou) {
          bestIou = iou;
          bestTrackIndex = trackIndex;
        }
      });

      if (bestTrackIndex >= 0) {
        assignments.set(detectionIndex, bestTrackIndex);
        unmatchedTracks.delete(bestTrackIndex);
      }
    }

    unmatchedTracks.forEach((trackIndex) => {
      this.tracks[trackIndex].misses += 1;
    });

    const output = detections.map((detection, detectionIndex) => {
      const matchedTrackIndex = assignments.get(detectionIndex);
      if (matchedTrackIndex !== undefined) {
        const track = this.tracks[matchedTrackIndex];
        track.detection = detection;
        track.age += 1;
        track.misses = 0;
        return { ...detection, trackId: track.id, trackAge: track.age };
      }

      const track: InternalTrack<T> = {
        id: this.nextId,
        detection,
        age: 1,
        misses: 0,
      };
      this.nextId += 1;
      this.tracks.push(track);
      return { ...detection, trackId: track.id, trackAge: track.age };
    });

    this.tracks = this.tracks.filter((track) => track.misses <= this.maxMisses);
    return output;
  }

  get activeTrackCount() {
    return this.tracks.filter((track) => track.misses === 0).length;
  }
}

type SmoothedEntry = {
  relation: SceneRelation;
  score: number;
  rankScore: number;
  hits: number;
  misses: number;
};

export type RelationSmootherOptions = {
  emaWeight?: number;
  maxMisses?: number;
  maxRelations?: number;
};

function relationKey(relation: SceneRelation) {
  const subjectId = relation.subject.trackId;
  const objectId = relation.object.trackId;
  if (subjectId === undefined || objectId === undefined) return null;
  return `${subjectId}|${relation.predicate}|${objectId}`;
}

export class RelationSmoother {
  private readonly emaWeight: number;
  private readonly maxMisses: number;
  private readonly maxRelations: number;
  private entries = new Map<string, SmoothedEntry>();

  constructor(options: RelationSmootherOptions = {}) {
    this.emaWeight = options.emaWeight ?? 0.62;
    this.maxMisses = options.maxMisses ?? 1;
    this.maxRelations = options.maxRelations ?? 8;
  }

  reset() {
    this.entries.clear();
  }

  update(relations: SceneRelation[]) {
    const seen = new Set<string>();

    for (const relation of relations) {
      const key = relationKey(relation);
      if (!key) continue;
      seen.add(key);
      const previous = this.entries.get(key);
      if (!previous) {
        this.entries.set(key, {
          relation,
          score: relation.score,
          rankScore: relation.rankScore,
          hits: 1,
          misses: 0,
        });
        continue;
      }

      const score = previous.score * this.emaWeight + relation.score * (1 - this.emaWeight);
      const rankScore = previous.rankScore * this.emaWeight + relation.rankScore * (1 - this.emaWeight);
      this.entries.set(key, {
        relation: { ...relation, score, rankScore },
        score,
        rankScore,
        hits: previous.hits + 1,
        misses: 0,
      });
    }

    for (const [key, entry] of this.entries) {
      if (seen.has(key)) continue;
      entry.misses += 1;
      if (entry.misses > this.maxMisses) this.entries.delete(key);
    }

    return this.values();
  }

  values() {
    return [...this.entries.values()]
      .sort((left, right) => right.rankScore - left.rankScore)
      .slice(0, this.maxRelations)
      .map((entry) => ({
        ...entry.relation,
        score: entry.score,
        rankScore: entry.rankScore,
      }));
  }
}

export function resolveRelationBoxes<T extends TrackedDetection>(
  relations: SceneRelation[],
  detections: T[],
) {
  const byId = new Map(detections.map((detection) => [detection.trackId, detection]));
  return relations
    .map((relation) => {
      const subjectId = relation.subject.trackId;
      const objectId = relation.object.trackId;
      if (subjectId === undefined || objectId === undefined) return relation;
      const subject = byId.get(subjectId);
      const object = byId.get(objectId);
      if (!subject || !object) return null;
      return {
        ...relation,
        subject: { ...relation.subject, ...subject },
        object: { ...relation.object, ...object },
      };
    })
    .filter((relation): relation is SceneRelation => relation !== null);
}
