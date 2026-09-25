package com.sunnygumber.cctvaivisionlab.ui

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.nativeCanvas
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import com.sunnygumber.cctvaivisionlab.tracking.FramePoint
import com.sunnygumber.cctvaivisionlab.tracking.MetricTrackState
import java.util.Locale
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

private val vehicleClasses = setOf(1, 2, 3, 5, 7)

@Composable
fun VisionOverlay(
    frameWidth: Int,
    frameHeight: Int,
    detections: List<Detection>,
    relations: List<SceneRelation>,
    showDetections: Boolean,
    calibrationPoints: List<FramePoint> = emptyList(),
    metricTracks: List<MetricTrackState> = emptyList(),
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier) {
        if (frameWidth <= 0 || frameHeight <= 0) return@Canvas

        val scale = min(size.width / frameWidth, size.height / frameHeight)
        val offsetX = (size.width - frameWidth * scale) / 2f
        val offsetY = (size.height - frameHeight * scale) / 2f
        val metricByTrack = metricTracks.associateBy { it.trackId }

        fun point(x: Float, y: Float) = Offset(
            x = offsetX + x * scale,
            y = offsetY + y * scale,
        )

        if (showDetections) {
            detections.forEach { detection ->
                val topLeft = point(detection.box.left, detection.box.top)
                val boxSize = Size(
                    detection.box.width * scale,
                    detection.box.height * scale,
                )
                val color = when {
                    detection.classId == 0 -> Color(0xFF58E2D3)
                    detection.classId in vehicleClasses -> Color(0xFFF4CF52)
                    else -> Color(0xFF79BDF2)
                }
                drawRect(
                    color = color,
                    topLeft = topLeft,
                    size = boxSize,
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2.5f),
                )

                val metric = detection.trackId?.let(metricByTrack::get)
                if (metric != null) {
                    val anchor = point(detection.box.centerX, detection.box.bottom)
                    val text = String.format(
                        Locale.US,
                        "#%d %.1fm, %.1fm · %.2fm/s",
                        metric.trackId,
                        metric.xMeters,
                        metric.yMeters,
                        metric.speedMetersPerSecond,
                    )
                    drawContext.canvas.nativeCanvas.drawText(
                        text,
                        anchor.x + 6f,
                        anchor.y - 8f,
                        Paint(Paint.ANTI_ALIAS_FLAG).apply {
                            this.color = android.graphics.Color.WHITE
                            textSize = 22f
                            style = Paint.Style.FILL
                            setShadowLayer(4f, 1f, 1f, android.graphics.Color.BLACK)
                        },
                    )
                }
            }
        }

        relations.take(3).forEach { relation ->
            val start = point(relation.subject.box.centerX, relation.subject.box.centerY)
            val end = point(relation.objectDetection.box.centerX, relation.objectDetection.box.centerY)
            val relationColor = Color(0xFFFFAD66)
            drawLine(
                color = relationColor,
                start = start,
                end = end,
                strokeWidth = 3f,
            )

            val angle = atan2(end.y - start.y, end.x - start.x)
            val arrow = 12f
            drawLine(
                color = relationColor,
                start = end,
                end = Offset(
                    end.x - arrow * cos(angle - Math.PI.toFloat() / 6f),
                    end.y - arrow * sin(angle - Math.PI.toFloat() / 6f),
                ),
                strokeWidth = 3f,
            )
            drawLine(
                color = relationColor,
                start = end,
                end = Offset(
                    end.x - arrow * cos(angle + Math.PI.toFloat() / 6f),
                    end.y - arrow * sin(angle + Math.PI.toFloat() / 6f),
                ),
                strokeWidth = 3f,
            )

            val middle = Offset((start.x + end.x) / 2f, (start.y + end.y) / 2f)
            val text = "${relation.predicate} ${(relation.score * 100).toInt()}%"
            drawContext.canvas.nativeCanvas.drawText(
                text,
                middle.x,
                middle.y,
                Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    this.color = android.graphics.Color.rgb(255, 210, 173)
                    textSize = 28f
                    style = Paint.Style.FILL
                    setShadowLayer(5f, 1f, 1f, android.graphics.Color.BLACK)
                },
            )
        }

        val calibrationColor = Color(0xFFFFE082)
        val mappedCalibration = calibrationPoints.map { point(it.x, it.y) }
        mappedCalibration.forEachIndexed { index, location ->
            drawCircle(
                color = calibrationColor,
                radius = 8f,
                center = location,
            )
            drawContext.canvas.nativeCanvas.drawText(
                (index + 1).toString(),
                location.x + 10f,
                location.y - 10f,
                Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    this.color = android.graphics.Color.WHITE
                    textSize = 24f
                    style = Paint.Style.FILL
                    setShadowLayer(4f, 1f, 1f, android.graphics.Color.BLACK)
                },
            )
        }

        if (mappedCalibration.size >= 2) {
            for (index in 0 until mappedCalibration.size - 1) {
                drawLine(
                    color = calibrationColor,
                    start = mappedCalibration[index],
                    end = mappedCalibration[index + 1],
                    strokeWidth = 3f,
                )
            }
            if (mappedCalibration.size == 4) {
                drawLine(
                    color = calibrationColor,
                    start = mappedCalibration.last(),
                    end = mappedCalibration.first(),
                    strokeWidth = 3f,
                )
            }
        }
    }
}
