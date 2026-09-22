package com.sunnygumber.cctvaivisionlab.inference

import ai.onnxruntime.OnnxTensor
import com.sunnygumber.cctvaivisionlab.core.BoundingBox
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.DetectionResult
import com.sunnygumber.cctvaivisionlab.core.Detector
import com.sunnygumber.cctvaivisionlab.core.FrameData
import com.sunnygumber.cctvaivisionlab.core.InferenceMetrics
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class YoloXDetector(
    modelFile: File,
    preferNnapi: Boolean = true,
) : Detector {
    private val nativeSession = OrtSessionFactory.create(modelFile, preferNnapi)
    private val inputName = nativeSession.session.inputNames.first()
    private val inputSize = 640
    private val classCount = CocoLabels.names.size
    private val attributes = 5 + classCount
    private val strides = intArrayOf(8, 16, 32)

    override suspend fun detect(
        frame: FrameData,
        confidenceThreshold: Float,
    ): DetectionResult {
        val preprocessStart = System.nanoTime()
        val prepared = ImagePreprocess.yoloxLetterboxBgr(frame, inputSize)
        val preprocessMs = elapsedMs(preprocessStart)

        val inputTensor = OnnxTensor.createTensor(
            OrtSessionFactory.environment,
            FloatBuffer.wrap(prepared.tensor),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()),
        )

        val inferenceStart = System.nanoTime()
        val rawOutput: FloatArray
        nativeSession.session.run(mapOf(inputName to inputTensor)).use { result ->
            val outputTensor = result.get(0) as? OnnxTensor
                ?: error("YOLOX output 0 is not a tensor")
            val buffer = outputTensor.floatBuffer
                ?: error("YOLOX output is not float32-compatible")
            rawOutput = FloatArray(buffer.remaining())
            buffer.get(rawOutput)
        }
        inputTensor.close()
        val inferenceMs = elapsedMs(inferenceStart)

        val postStart = System.nanoTime()
        val decoded = decode(
            output = rawOutput,
            scale = prepared.scale,
            sourceWidth = frame.width,
            sourceHeight = frame.height,
            threshold = max(0.08f, confidenceThreshold),
        )
        val final = nonMaxSuppression(decoded, iouThreshold = 0.65f, maxDetections = 40)
        val postprocessingMs = elapsedMs(postStart)

        return DetectionResult(
            detections = final,
            metrics = InferenceMetrics(
                provider = nativeSession.provider,
                preprocessingMs = preprocessMs,
                inferenceMs = inferenceMs,
                postprocessingMs = postprocessingMs,
            ),
        )
    }

    private fun decode(
        output: FloatArray,
        scale: Float,
        sourceWidth: Int,
        sourceHeight: Int,
        threshold: Float,
    ): List<Detection> {
        require(output.size % attributes == 0) {
            "Unexpected YOLOX output length ${output.size}; expected a multiple of $attributes"
        }

        val expectedCells = strides.sumOf { stride ->
            val side = inputSize / stride
            side * side
        }
        val cellCount = output.size / attributes
        require(cellCount == expectedCells) {
            "Unexpected YOLOX cell count $cellCount; expected $expectedCells"
        }

        val detections = ArrayList<Detection>()
        var cellIndex = 0

        for (stride in strides) {
            val gridSize = inputSize / stride
            for (gridY in 0 until gridSize) {
                for (gridX in 0 until gridSize) {
                    val row = cellIndex * attributes
                    val centerX = (output[row] + gridX) * stride
                    val centerY = (output[row + 1] + gridY) * stride
                    val width = exp(output[row + 2].toDouble()).toFloat() * stride
                    val height = exp(output[row + 3].toDouble()).toFloat() * stride
                    val objectness = output[row + 4]

                    var bestClass = -1
                    var bestClassScore = 0f
                    for (classId in 0 until classCount) {
                        val score = output[row + 5 + classId]
                        if (score > bestClassScore) {
                            bestClassScore = score
                            bestClass = classId
                        }
                    }

                    val confidence = objectness * bestClassScore
                    if (bestClass >= 0 && confidence >= threshold) {
                        val x1 = ((centerX - width / 2f) / scale).coerceIn(0f, sourceWidth.toFloat())
                        val y1 = ((centerY - height / 2f) / scale).coerceIn(0f, sourceHeight.toFloat())
                        val x2 = ((centerX + width / 2f) / scale).coerceIn(0f, sourceWidth.toFloat())
                        val y2 = ((centerY + height / 2f) / scale).coerceIn(0f, sourceHeight.toFloat())

                        if (x2 > x1 && y2 > y1) {
                            detections += Detection(
                                box = BoundingBox(x1, y1, x2, y2),
                                confidence = confidence,
                                label = CocoLabels.names[bestClass],
                                classId = bestClass,
                            )
                        }
                    }
                    cellIndex += 1
                }
            }
        }

        return detections
    }

    private fun nonMaxSuppression(
        detections: List<Detection>,
        iouThreshold: Float,
        maxDetections: Int,
    ): List<Detection> {
        val kept = ArrayList<Detection>(min(maxDetections, detections.size))
        val candidates = detections.sortedByDescending { it.confidence }.toMutableList()

        while (candidates.isNotEmpty() && kept.size < maxDetections) {
            val best = candidates.removeAt(0)
            kept += best
            candidates.removeAll { candidate ->
                candidate.classId == best.classId && iou(best.box, candidate.box) >= iouThreshold
            }
        }

        return kept
    }

    private fun iou(a: BoundingBox, b: BoundingBox): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        if (intersection <= 0f) return 0f
        val union = a.area + b.area - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private fun elapsedMs(startNs: Long): Double =
        (System.nanoTime() - startNs) / 1_000_000.0

    override suspend fun close() {
        nativeSession.session.close()
    }
}
