package com.sunnygumber.cctvaivisionlab.tracking

import com.sunnygumber.cctvaivisionlab.core.Detection
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot

data class FramePoint(
    val x: Float,
    val y: Float,
)

data class MetricTrackState(
    val trackId: Int,
    val label: String,
    val xMeters: Double,
    val yMeters: Double,
    val speedMetersPerSecond: Double,
    val distanceSincePreviousMeters: Double,
)

class GroundPlaneMetricTracker : AutoCloseable {
    private var homography: Mat? = null
    private var previous = mutableMapOf<Int, Pair<Long, Point>>()

    val isCalibrated: Boolean
        get() = homography != null

    fun configure(
        imagePoints: List<FramePoint>,
        widthMeters: Double,
        depthMeters: Double,
    ) {
        require(imagePoints.size == 4) { "Four ground-plane image points are required." }
        require(widthMeters > 0.0) { "Ground width must be > 0 meters." }
        require(depthMeters > 0.0) { "Ground depth must be > 0 meters." }

        val source = MatOfPoint2f(
            Point(imagePoints[0].x.toDouble(), imagePoints[0].y.toDouble()),
            Point(imagePoints[1].x.toDouble(), imagePoints[1].y.toDouble()),
            Point(imagePoints[2].x.toDouble(), imagePoints[2].y.toDouble()),
            Point(imagePoints[3].x.toDouble(), imagePoints[3].y.toDouble()),
        )
        val destination = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(widthMeters, 0.0),
            Point(widthMeters, depthMeters),
            Point(0.0, depthMeters),
        )

        val matrix = try {
            Imgproc.getPerspectiveTransform(source, destination)
        } finally {
            source.release()
            destination.release()
        }

        homography?.release()
        homography = matrix
        previous.clear()
    }

    fun clear() {
        homography?.release()
        homography = null
        previous.clear()
    }

    fun update(
        detections: List<Detection>,
        timestampNs: Long,
    ): List<MetricTrackState> {
        val matrix = homography ?: return emptyList()
        val output = ArrayList<MetricTrackState>()
        val nextPrevious = mutableMapOf<Int, Pair<Long, Point>>()

        detections.forEach { detection ->
            val trackId = detection.trackId ?: return@forEach
            val source = MatOfPoint2f(
                Point(
                    detection.box.centerX.toDouble(),
                    detection.box.bottom.toDouble(),
                ),
            )
            val destination = MatOfPoint2f()

            try {
                Core.perspectiveTransform(source, destination, matrix)
                val point = destination.toArray().firstOrNull() ?: return@forEach
                val prior = previous[trackId]
                val distance = if (prior == null) {
                    0.0
                } else {
                    hypot(point.x - prior.second.x, point.y - prior.second.y)
                }
                val elapsedSeconds = if (prior == null) {
                    0.0
                } else {
                    (timestampNs - prior.first).coerceAtLeast(0L) / 1_000_000_000.0
                }
                val speed = if (elapsedSeconds > 0.03) distance / elapsedSeconds else 0.0

                output += MetricTrackState(
                    trackId = trackId,
                    label = detection.label,
                    xMeters = point.x,
                    yMeters = point.y,
                    speedMetersPerSecond = speed,
                    distanceSincePreviousMeters = distance,
                )
                nextPrevious[trackId] = timestampNs to point
            } finally {
                source.release()
                destination.release()
            }
        }

        previous = nextPrevious
        return output
    }

    override fun close() {
        clear()
    }
}
