package com.sunnygumber.cctvaivisionlab.tracking

import android.os.SystemClock
import com.sunnygumber.cctvaivisionlab.camera.GrayFrame
import com.sunnygumber.cctvaivisionlab.core.BoundingBox
import com.sunnygumber.cctvaivisionlab.core.Detection
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.max
import kotlin.math.min

data class CpuTrackingMetrics(
    val processingMs: Double = 0.0,
    val trackedObjects: Int = 0,
    val trackedPoints: Int = 0,
    val updates: Long = 0,
)

data class CpuTrackingResult(
    val detections: List<Detection>,
    val metrics: CpuTrackingMetrics,
)

private data class FlowTrack(
    var detection: Detection,
    var points: List<Point>,
    var misses: Int = 0,
)

class CpuOpticalFlowTracker(
    private val maxPointsPerObject: Int = 18,
    private val minimumPoints: Int = 4,
    private val maxFlowError: Float = 24f,
    private val maxMisses: Int = 3,
) : AutoCloseable {
    private val tracks = linkedMapOf<Int, FlowTrack>()
    private var previousGray: Mat? = null
    private var previousTimestampNs: Long = 0L
    private var updateCount = 0L

    @Synchronized
    fun reset() {
        tracks.clear()
        previousGray?.release()
        previousGray = null
        previousTimestampNs = 0L
        updateCount = 0L
    }

    @Synchronized
    fun currentDetections(): List<Detection> =
        tracks.values.map { it.detection }

    @Synchronized
    fun latestTimestampNs(): Long = previousTimestampNs

    @Synchronized
    fun update(frame: GrayFrame): CpuTrackingResult {
        val started = SystemClock.elapsedRealtimeNanos()
        val previous = previousGray

        if (previous == null || tracks.isEmpty()) {
            replacePrevious(frame)
            return CpuTrackingResult(
                detections = currentDetections(),
                metrics = metrics(started),
            )
        }

        val pointOwners = ArrayList<Int>()
        val previousPoints = ArrayList<Point>()
        tracks.forEach { (trackId, track) ->
            track.points.forEach { point ->
                pointOwners += trackId
                previousPoints += point
            }
        }

        if (previousPoints.isEmpty()) {
            tracks.values.forEach { track ->
                track.points = seedPoints(frame.mat, track.detection.box)
            }
            replacePrevious(frame)
            return CpuTrackingResult(
                detections = currentDetections(),
                metrics = metrics(started),
            )
        }

        val prevPts = MatOfPoint2f(*previousPoints.toTypedArray())
        val nextPts = MatOfPoint2f()
        val status = MatOfByte()
        val error = MatOfFloat()

        try {
            Video.calcOpticalFlowPyrLK(
                previous,
                frame.mat,
                prevPts,
                nextPts,
                status,
                error,
                Size(21.0, 21.0),
                3,
                TermCriteria(TermCriteria.COUNT or TermCriteria.EPS, 20, 0.03),
                0,
                1e-4,
            )

            val nextArray = nextPts.toArray()
            val statusArray = status.toArray()
            val errorArray = error.toArray()

            val byTrack = linkedMapOf<Int, MutableList<Pair<Point, Point>>>()
            for (index in previousPoints.indices) {
                if (index >= nextArray.size || index >= statusArray.size) break
                if (statusArray[index].toInt() == 0) continue
                if (index < errorArray.size && errorArray[index] > maxFlowError) continue

                val previousPoint = previousPoints[index]
                val nextPoint = nextArray[index]
                if (
                    nextPoint.x !in 0.0..frame.width.toDouble() ||
                    nextPoint.y !in 0.0..frame.height.toDouble()
                ) {
                    continue
                }

                val owner = pointOwners[index]
                byTrack.getOrPut(owner) { ArrayList() } += previousPoint to nextPoint
            }

            val remove = ArrayList<Int>()
            tracks.forEach { (trackId, track) ->
                val pairs = byTrack[trackId].orEmpty()
                if (pairs.size >= minimumPoints) {
                    val dx = median(pairs.map { it.second.x - it.first.x }).toFloat()
                    val dy = median(pairs.map { it.second.y - it.first.y }).toFloat()
                    val shifted = shiftAndClamp(track.detection.box, dx, dy, frame.width, frame.height)
                    track.detection = track.detection.copy(box = shifted)
                    track.points = pairs.map { it.second }
                    track.misses = 0

                    if (track.points.size < minimumPoints * 2) {
                        track.points = seedPoints(frame.mat, shifted)
                    }
                } else {
                    track.misses += 1
                    if (track.misses > maxMisses) {
                        remove += trackId
                    } else {
                        track.points = seedPoints(frame.mat, track.detection.box)
                    }
                }
            }

            remove.forEach(tracks::remove)
        } finally {
            prevPts.release()
            nextPts.release()
            status.release()
            error.release()
        }

        replacePrevious(frame)
        updateCount += 1

        return CpuTrackingResult(
            detections = currentDetections(),
            metrics = metrics(started),
        )
    }

    @Synchronized
    fun correctFromDetector(
        detectorFrame: GrayFrame,
        detections: List<Detection>,
    ): CpuTrackingResult {
        val started = SystemClock.elapsedRealtimeNanos()
        val destination = previousGray

        val aligned = if (
            destination != null &&
            previousTimestampNs > detectorFrame.timestampNs &&
            detections.isNotEmpty()
        ) {
            propagateDetections(
                sourceGray = detectorFrame.mat,
                destinationGray = destination,
                detections = detections,
                width = destination.cols(),
                height = destination.rows(),
            )
        } else {
            detections
        }

        val currentGray = destination ?: detectorFrame.mat
        tracks.clear()
        aligned.forEach { detection ->
            val id = detection.trackId ?: return@forEach
            tracks[id] = FlowTrack(
                detection = detection,
                points = seedPoints(currentGray, detection.box),
            )
        }

        if (destination == null) {
            replacePrevious(detectorFrame)
        }

        return CpuTrackingResult(
            detections = currentDetections(),
            metrics = metrics(started),
        )
    }

    private fun propagateDetections(
        sourceGray: Mat,
        destinationGray: Mat,
        detections: List<Detection>,
        width: Int,
        height: Int,
    ): List<Detection> {
        val sourcePoints = ArrayList<Point>()
        val owners = ArrayList<Int>()

        detections.forEachIndexed { detectionIndex, detection ->
            seedPoints(sourceGray, detection.box).forEach { point ->
                sourcePoints += point
                owners += detectionIndex
            }
        }

        if (sourcePoints.isEmpty()) return detections

        val prevPts = MatOfPoint2f(*sourcePoints.toTypedArray())
        val nextPts = MatOfPoint2f()
        val status = MatOfByte()
        val error = MatOfFloat()

        return try {
            Video.calcOpticalFlowPyrLK(
                sourceGray,
                destinationGray,
                prevPts,
                nextPts,
                status,
                error,
                Size(31.0, 31.0),
                4,
                TermCriteria(TermCriteria.COUNT or TermCriteria.EPS, 30, 0.03),
                0,
                1e-4,
            )

            val next = nextPts.toArray()
            val statusArray = status.toArray()
            val errorArray = error.toArray()
            val deltas = Array(detections.size) { ArrayList<Pair<Double, Double>>() }

            sourcePoints.indices.forEach { index ->
                if (index >= next.size || index >= statusArray.size) return@forEach
                if (statusArray[index].toInt() == 0) return@forEach
                if (index < errorArray.size && errorArray[index] > maxFlowError * 1.5f) return@forEach

                val owner = owners[index]
                deltas[owner] +=
                    (next[index].x - sourcePoints[index].x) to
                    (next[index].y - sourcePoints[index].y)
            }

            detections.mapIndexed { index, detection ->
                val moves = deltas[index]
                if (moves.size < minimumPoints) {
                    detection
                } else {
                    val dx = median(moves.map { it.first }).toFloat()
                    val dy = median(moves.map { it.second }).toFloat()
                    detection.copy(
                        box = shiftAndClamp(detection.box, dx, dy, width, height),
                    )
                }
            }
        } finally {
            prevPts.release()
            nextPts.release()
            status.release()
            error.release()
        }
    }

    private fun seedPoints(gray: Mat, box: BoundingBox): List<Point> {
        if (gray.empty()) return emptyList()

        val left = box.left.toInt().coerceIn(0, max(0, gray.cols() - 1))
        val top = box.top.toInt().coerceIn(0, max(0, gray.rows() - 1))
        val right = box.right.toInt().coerceIn(left + 1, gray.cols())
        val bottom = box.bottom.toInt().coerceIn(top + 1, gray.rows())

        if (right - left < 4 || bottom - top < 4) return emptyList()

        val mask = Mat.zeros(gray.rows(), gray.cols(), CvType.CV_8UC1)
        val corners = MatOfPoint()
        return try {
            Imgproc.rectangle(
                mask,
                Point(left.toDouble(), top.toDouble()),
                Point((right - 1).toDouble(), (bottom - 1).toDouble()),
                Scalar(255.0),
                -1,
            )
            Imgproc.goodFeaturesToTrack(
                gray,
                corners,
                maxPointsPerObject,
                0.01,
                5.0,
                mask,
                3,
                false,
                0.04,
            )
            corners.toArray().toList()
        } finally {
            corners.release()
            mask.release()
        }
    }

    private fun shiftAndClamp(
        box: BoundingBox,
        dx: Float,
        dy: Float,
        width: Int,
        height: Int,
    ): BoundingBox {
        val boxWidth = box.width.coerceAtMost(width.toFloat())
        val boxHeight = box.height.coerceAtMost(height.toFloat())
        val left = (box.left + dx).coerceIn(0f, max(0f, width - boxWidth))
        val top = (box.top + dy).coerceIn(0f, max(0f, height - boxHeight))

        return BoundingBox(
            left = left,
            top = top,
            right = min(width.toFloat(), left + boxWidth),
            bottom = min(height.toFloat(), top + boxHeight),
        )
    }

    private fun replacePrevious(frame: GrayFrame) {
        previousGray?.release()
        previousGray = frame.mat.clone()
        previousTimestampNs = frame.timestampNs
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }

    private fun metrics(startedNs: Long): CpuTrackingMetrics =
        CpuTrackingMetrics(
            processingMs = (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000.0,
            trackedObjects = tracks.size,
            trackedPoints = tracks.values.sumOf { it.points.size },
            updates = updateCount,
        )

    override fun close() {
        reset()
    }
}
