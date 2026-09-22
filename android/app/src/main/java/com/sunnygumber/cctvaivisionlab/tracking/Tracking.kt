package com.sunnygumber.cctvaivisionlab.tracking

import com.sunnygumber.cctvaivisionlab.core.BoundingBox
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import com.sunnygumber.cctvaivisionlab.core.Tracker

fun intersectionOverUnion(a: BoundingBox, b: BoundingBox): Float {
    val left = maxOf(a.left, b.left)
    val top = maxOf(a.top, b.top)
    val right = minOf(a.right, b.right)
    val bottom = minOf(a.bottom, b.bottom)
    val intersection = maxOf(0f, right - left) * maxOf(0f, bottom - top)
    if (intersection <= 0f) return 0f
    val union = a.area + b.area - intersection
    return if (union > 0f) intersection / union else 0f
}

private fun overlapOverSmallerArea(a: BoundingBox, b: BoundingBox): Float {
    val left = maxOf(a.left, b.left)
    val top = maxOf(a.top, b.top)
    val right = minOf(a.right, b.right)
    val bottom = minOf(a.bottom, b.bottom)
    val intersection = maxOf(0f, right - left) * maxOf(0f, bottom - top)
    if (intersection <= 0f) return 0f
    val smaller = minOf(a.area, b.area)
    return if (smaller > 0f) intersection / smaller else 0f
}

fun deduplicateDetections(
    detections: List<Detection>,
    iouThreshold: Float = 0.45f,
    containmentThreshold: Float = 0.72f,
): List<Detection> {
    val kept = mutableListOf<Detection>()
    detections.sortedByDescending { it.confidence }.forEach { candidate ->
        val duplicate = kept.any { existing ->
            existing.label == candidate.label &&
                (
                    intersectionOverUnion(existing.box, candidate.box) >= iouThreshold ||
                        overlapOverSmallerArea(existing.box, candidate.box) >= containmentThreshold
                    )
        }
        if (!duplicate) kept += candidate
    }
    return kept
}

class IoUTracker(
    private val iouThreshold: Float = 0.28f,
    private val maxMisses: Int = 8,
) : Tracker {
    private data class Track(
        val id: Int,
        var detection: Detection,
        var age: Int,
        var misses: Int,
    )

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    override fun reset() {
        tracks.clear()
        nextId = 1
    }

    override fun update(detections: List<Detection>): List<Detection> {
        val unmatchedTracks = tracks.indices.toMutableSet()
        val assignments = mutableMapOf<Int, Int>()

        detections.withIndex()
            .sortedByDescending { it.value.confidence }
            .forEach { indexed ->
                var bestTrack = -1
                var bestIou = iouThreshold

                unmatchedTracks.forEach trackLoop@{ trackIndex ->
                    val track = tracks[trackIndex]
                    if (track.detection.label != indexed.value.label) return@trackLoop
                    val iou = intersectionOverUnion(track.detection.box, indexed.value.box)
                    if (iou > bestIou) {
                        bestIou = iou
                        bestTrack = trackIndex
                    }
                }

                if (bestTrack >= 0) {
                    assignments[indexed.index] = bestTrack
                    unmatchedTracks.remove(bestTrack)
                }
            }

        unmatchedTracks.forEach { tracks[it].misses += 1 }

        val output = detections.mapIndexed { detectionIndex, detection ->
            val matchedIndex = assignments[detectionIndex]
            if (matchedIndex != null) {
                val track = tracks[matchedIndex]
                track.detection = detection
                track.age += 1
                track.misses = 0
                detection.copy(trackId = track.id, trackAge = track.age)
            } else {
                val track = Track(
                    id = nextId++,
                    detection = detection,
                    age = 1,
                    misses = 0,
                )
                tracks += track
                detection.copy(trackId = track.id, trackAge = track.age)
            }
        }

        tracks.removeAll { it.misses > maxMisses }
        return output
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

    private val entries = mutableMapOf<String, Entry>()

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
                entries[key] = Entry(
                    relation = relation,
                    score = relation.score,
                    rankScore = relation.rankScore,
                    misses = 0,
                )
            } else {
                val score = previous.score * emaWeight + relation.score * (1f - emaWeight)
                val rankScore = previous.rankScore * emaWeight + relation.rankScore * (1f - emaWeight)
                previous.relation = relation.copy(score = score, rankScore = rankScore)
                previous.score = score
                previous.rankScore = rankScore
                previous.misses = 0
            }
        }

        val iterator = entries.iterator()
        while (iterator.hasNext()) {
            val item = iterator.next()
            if (item.key in seen) continue
            item.value.misses += 1
            if (item.value.misses > maxMisses) iterator.remove()
        }

        return values()
    }

    fun values(): List<SceneRelation> = entries.values
        .sortedByDescending { it.rankScore }
        .take(maxRelations)
        .map { entry ->
            entry.relation.copy(score = entry.score, rankScore = entry.rankScore)
        }
}

fun resolveRelationBoxes(
    relations: List<SceneRelation>,
    detections: List<Detection>,
): List<SceneRelation> {
    val byId = detections.mapNotNull { detection ->
        detection.trackId?.let { id -> id to detection }
    }.toMap()

    return relations.mapNotNull { relation ->
        val subjectId = relation.subject.trackId ?: return@mapNotNull relation
        val objectId = relation.objectDetection.trackId ?: return@mapNotNull relation
        val subject = byId[subjectId] ?: return@mapNotNull null
        val objectDetection = byId[objectId] ?: return@mapNotNull null
        relation.copy(subject = subject, objectDetection = objectDetection)
    }
}
