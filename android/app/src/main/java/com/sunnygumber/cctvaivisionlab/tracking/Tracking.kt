package com.sunnygumber.cctvaivisionlab.tracking

import com.sunnygumber.cctvaivisionlab.core.BoundingBox
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import com.sunnygumber.cctvaivisionlab.core.Tracker
import kotlin.math.max
import kotlin.math.min

fun intersectionOverUnion(a: BoundingBox, b: BoundingBox): Float {
    val left = max(a.left, b.left)
    val top = max(a.top, b.top)
    val right = min(a.right, b.right)
    val bottom = min(a.bottom, b.bottom)
    val intersection = max(0f, right - left) * max(0f, bottom - top)
    if (intersection <= 0f) return 0f
    val union = a.area + b.area - intersection
    return if (union > 0f) intersection / union else 0f
}

private fun overlapOverSmallerArea(a: BoundingBox, b: BoundingBox): Float {
    val left = max(a.left, b.left)
    val top = max(a.top, b.top)
    val right = min(a.right, b.right)
    val bottom = min(a.bottom, b.bottom)
    val intersection = max(0f, right - left) * max(0f, bottom - top)
    if (intersection <= 0f) return 0f
    val smaller = min(a.area, b.area)
    return if (smaller > 0f) intersection / smaller else 0f
}

fun deduplicateDetections(
    detections: List<Detection>,
    iouThreshold: Float = 0.45f,
    containmentThreshold: Float = 0.72f,
): List<Detection> {
    val kept = ArrayList<Detection>()
    detections.sortedByDescending { it.confidence }.forEach { candidate ->
        val duplicate = kept.any { existing ->
            existing.label == candidate.label &&
                (intersectionOverUnion(existing.box, candidate.box) >= iouThreshold ||
                    overlapOverSmallerArea(existing.box, candidate.box) >= containmentThreshold)
        }
        if (!duplicate) kept += candidate
    }
    return kept
}

private data class InternalTrack(
    val id: Int,
    var detection: Detection,
    var age: Int,
    var misses: Int,
)

class IoUTracker(
    private val iouThreshold: Float = 0.28f,
    private val maxMisses: Int = 8,
) : Tracker {
    private val tracks = ArrayList<InternalTrack>()
    private var nextId = 1

    override fun update(detections: List<Detection>): List<Detection> {
        val unmatchedTracks = tracks.indices.toMutableSet()
        val assignments = mutableMapOf<Int, Int>()

        detections.indices
            .sortedByDescending { detections[it].confidence }
            .forEach { detectionIndex ->
                val detection = detections[detectionIndex]
                var bestTrack = -1
                var bestIou = iouThreshold

                unmatchedTracks.forEach { trackIndex ->
                    val track = tracks[trackIndex]
                    if (track.detection.label != detection.label) return@forEach
                    val iou = intersectionOverUnion(track.detection.box, detection.box)
                    if (iou > bestIou) {
                        bestIou = iou
                        bestTrack = trackIndex
                    }
                }

                if (bestTrack >= 0) {
                    assignments[detectionIndex] = bestTrack
                    unmatchedTracks.remove(bestTrack)
                }
            }

        unmatchedTracks.forEach { tracks[it].misses += 1 }

        val output = detections.mapIndexed { index, detection ->
            val matched = assignments[index]
            if (matched != null) {
                val track = tracks[matched]
                track.detection = detection
                track.age += 1
                track.misses = 0
                detection.copy(trackId = track.id, trackAge = track.age)
            } else {
                val track = InternalTrack(nextId++, detection, age = 1, misses = 0)
                tracks += track
                detection.copy(trackId = track.id, trackAge = 1)
            }
        }

        tracks.removeAll { it.misses > maxMisses }
        return output
    }

    override fun reset() {
        tracks.clear()
        nextId = 1
    }

    val activeTrackCount: Int
        get() = tracks.count { it.misses == 0 }
}

class RelationSmoother(
    private val emaWeight: Float = 0.62f,
    private val maxMisses: Int = 1,
    private val maxRelations: Int = 8,
) {
    private data class Entry(
        var relation: SceneRelation,
        var score: Float,
        var rankScore: Float,
        var misses: Int,
    )

    private val entries = linkedMapOf<String, Entry>()

    fun reset() = entries.clear()

    fun update(relations: List<SceneRelation>): List<SceneRelation> {
        val seen = mutableSetOf<String>()

        relations.forEach { relation ->
            val subjectId = relation.subject.trackId ?: return@forEach
            val objectId = relation.objectDetection.trackId ?: return@forEach
            val key = "$subjectId|${relation.predicate}|$objectId"
            seen += key

            val previous = entries[key]
            if (previous == null) {
                entries[key] = Entry(relation, relation.score, relation.rankScore, 0)
            } else {
                previous.score = previous.score * emaWeight + relation.score * (1f - emaWeight)
                previous.rankScore = previous.rankScore * emaWeight + relation.rankScore * (1f - emaWeight)
                previous.relation = relation.copy(score = previous.score, rankScore = previous.rankScore)
                previous.misses = 0
            }
        }

        entries.entries.toList().forEach { (key, entry) ->
            if (key !in seen) {
                entry.misses += 1
                if (entry.misses > maxMisses) entries.remove(key)
            }
        }

        return entries.values
            .sortedByDescending { it.rankScore }
            .take(maxRelations)
            .map { it.relation.copy(score = it.score, rankScore = it.rankScore) }
    }

    fun resolveBoxes(relations: List<SceneRelation>, detections: List<Detection>): List<SceneRelation> {
        val byId = detections.mapNotNull { detection ->
            detection.trackId?.let { it to detection }
        }.toMap()

        return relations.mapNotNull { relation ->
            val subjectId = relation.subject.trackId ?: return@mapNotNull relation
            val objectId = relation.objectDetection.trackId ?: return@mapNotNull relation
            val subject = byId[subjectId] ?: return@mapNotNull null
            val objectDetection = byId[objectId] ?: return@mapNotNull null
            relation.copy(subject = subject, objectDetection = objectDetection)
        }
    }
}
