package com.sunnygumber.cctvaivisionlab.core

import java.io.File

interface Detector {
    suspend fun detect(frame: FrameData, confidenceThreshold: Float): DetectionResult
    suspend fun close()
}

interface RelationEngine {
    suspend fun analyse(
        frame: FrameData,
        detections: List<Detection>,
        confidenceThreshold: Float,
    ): RelationResult

    suspend fun close()
}

interface Tracker {
    fun update(detections: List<Detection>): List<Detection>
    fun reset()
}

interface FrameScheduler {
    fun shouldRunDetector(timestampNs: Long): Boolean
    fun shouldRunRelation(timestampNs: Long, relationCadenceMs: Long): Boolean
    fun markDetectorStarted(timestampNs: Long)
    fun markDetectorFinished(timestampNs: Long)
    fun markRelationStarted(timestampNs: Long)
    fun markRelationFinished(timestampNs: Long)
    fun reset()
}

enum class ModelState {
    MISSING,
    DOWNLOADING,
    READY,
    FAILED,
}

data class ModelDescriptor(
    val id: String,
    val fileName: String,
    val url: String,
    val version: String,
    val sha256: String? = null,
)

data class ModelStatus(
    val descriptor: ModelDescriptor,
    val state: ModelState,
    val localFile: File? = null,
    val downloadedBytes: Long = 0,
    val totalBytes: Long? = null,
    val error: String? = null,
)

interface ModelManager {
    suspend fun status(descriptor: ModelDescriptor): ModelStatus
    suspend fun ensureAvailable(
        descriptor: ModelDescriptor,
        onProgress: (ModelStatus) -> Unit = {},
    ): ModelStatus

    suspend fun clear(descriptor: ModelDescriptor)
}
