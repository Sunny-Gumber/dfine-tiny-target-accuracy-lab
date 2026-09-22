package com.sunnygumber.cctvaivisionlab.core

data class BoundingBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    init {
        require(right >= left) { "right must be >= left" }
        require(bottom >= top) { "bottom must be >= top" }
    }

    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val area: Float get() = width * height
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f
}

data class FrameData(
    val width: Int,
    val height: Int,
    val rgb: ByteArray,
    val timestampNs: Long,
    val rotationDegrees: Int = 0,
)

data class Detection(
    val box: BoundingBox,
    val confidence: Float,
    val label: String,
    val classId: Int,
    val trackId: Int? = null,
    val trackAge: Int = 0,
)

data class SceneRelation(
    val subject: Detection,
    val predicate: String,
    val objectDetection: Detection,
    val score: Float,
    val rankScore: Float,
)

enum class InferenceProvider {
    NNAPI,
    CPU,
}

data class InferenceMetrics(
    val provider: InferenceProvider,
    val preprocessingMs: Double = 0.0,
    val inferenceMs: Double = 0.0,
    val postprocessingMs: Double = 0.0,
)

data class DetectionResult(
    val detections: List<Detection>,
    val metrics: InferenceMetrics,
)

data class RelationResult(
    val relations: List<SceneRelation>,
    val metrics: InferenceMetrics,
)
