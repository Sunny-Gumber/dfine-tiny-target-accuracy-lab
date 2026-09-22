package com.sunnygumber.cctvaivisionlab.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import android.os.SystemClock
import com.sunnygumber.cctvaivisionlab.core.BoundingBox
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.DetectionResult
import com.sunnygumber.cctvaivisionlab.core.Detector
import com.sunnygumber.cctvaivisionlab.core.FrameData
import com.sunnygumber.cctvaivisionlab.core.InferenceMetrics
import java.io.File
import kotlin.math.max
import kotlin.math.min

class YoloXDetector(
    modelFile: File,
    preferNnapi: Boolean = true,
) : Detector {
    private val runtime = OrtRuntimeFactory.createSession(modelFile, preferNnapi)
    private val environment = OrtRuntimeFactory.environment
    private val inputName = runtime.session.inputNames.first()

    override suspend fun detect(frame: FrameData, confidenceThreshold: Float): DetectionResult {
        val preprocessStart = SystemClock.elapsedRealtimeNanos()
        val preprocessed = ImagePreprocessor.yoloX(frame, INPUT_SIZE)
        val preprocessingMs = elapsedMs(preprocessStart)

        val inputTensor = OnnxTensor.createTensor(
            environment,
            preprocessed.data,
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
        )

        val inferenceStart = SystemClock.elapsedRealtimeNanos()
        val result = try {
            runtime.session.run(mapOf(inputName to inputTensor))
        } finally {
            inputTensor.close()
        }
        val inferenceMs = elapsedMs(inferenceStart)

        val postStart = SystemClock.elapsedRealtimeNanos()
        val detections = result.use { outputs ->
            val output = outputs.get(0) as OnnxTensor
            val info = output.info as TensorInfo
            val shape = info.shape
            require(shape.size == 3) { "YOLOX output must be rank 3, got ${shape.contentToString()}" }

            val candidates = shape[1].toInt()
            val stride = shape[2].toInt()
            require(stride >= 6) { "Unexpected YOLOX output stride: $stride" }
            val classes = stride - 5
            val data = output.floatBuffer ?: error("YOLOX output is not float32-compatible.")
            val values = FloatArray(data.remaining())
            data.get(values)

            val raw = ArrayList<Detection>()
            for (index in 0 until candidates) {
                val base = index * stride
                val objectness = values[base + 4]
                if (objectness < confidenceThreshold) continue

                var bestClass = -1
                var bestClassScore = Float.NEGATIVE_INFINITY
                for (classIndex in 0 until classes) {
                    val score = values[base + 5 + classIndex]
                    if (score > bestClassScore) {
                        bestClassScore = score
                        bestClass = classIndex
                    }
                }

                val confidence = objectness * bestClassScore
                if (confidence < confidenceThreshold || bestClass !in CocoLabels.names.indices) continue

                val cx = values[base]
                val cy = values[base + 1]
                val width = values[base + 2]
                val height = values[base + 3]
                val scale = preprocessed.scale

                val left = ((cx - width / 2f) / scale).coerceIn(0f, frame.width.toFloat())
                val top = ((cy - height / 2f) / scale).coerceIn(0f, frame.height.toFloat())
                val right = ((cx + width / 2f) / scale).coerceIn(0f, frame.width.toFloat())
                val bottom = ((cy + height / 2f) / scale).coerceIn(0f, frame.height.toFloat())
                if (right <= left || bottom <= top) continue

                raw += Detection(
                    box = BoundingBox(left, top, right, bottom),
                    confidence = confidence,
                    label = CocoLabels.names[bestClass],
                    classId = bestClass,
                )
            }

            nms(raw, IOU_THRESHOLD, MAX_DETECTIONS)
        }
        val postprocessingMs = elapsedMs(postStart)

        return DetectionResult(
            detections = detections,
            metrics = InferenceMetrics(
                provider = runtime.provider,
                preprocessingMs = preprocessingMs,
                inferenceMs = inferenceMs,
                postprocessingMs = postprocessingMs,
            ),
        )
    }

    override suspend fun close() {
        runtime.close()
    }

    private fun nms(input: List<Detection>, threshold: Float, maxDetections: Int): List<Detection> {
        val output = ArrayList<Detection>()

        input.groupBy { it.classId }.values.forEach { classDetections ->
            val active = classDetections
                .sortedByDescending { it.confidence }
                .take(maxDetections * 3)
                .toMutableList()

            while (active.isNotEmpty()) {
                val best = active.removeAt(0)
                output += best
                active.removeAll { iou(best.box, it.box) > threshold }
            }
        }

        return output.sortedByDescending { it.confidence }.take(maxDetections)
    }

    companion object {
        const val INPUT_SIZE = 640
        private const val IOU_THRESHOLD = 0.65f
        private const val MAX_DETECTIONS = 40

        fun iou(a: BoundingBox, b: BoundingBox): Float {
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
            (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0
    }
}
