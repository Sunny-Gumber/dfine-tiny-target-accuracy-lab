package com.sunnygumber.cctvaivisionlab.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.sunnygumber.cctvaivisionlab.camera.ImageFrameLoader
import com.sunnygumber.cctvaivisionlab.camera.LensFacing
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.FrameData
import com.sunnygumber.cctvaivisionlab.core.InferenceMetrics
import com.sunnygumber.cctvaivisionlab.core.ModelState
import com.sunnygumber.cctvaivisionlab.core.ModelStatus
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import com.sunnygumber.cctvaivisionlab.inference.YoloXDetector
import com.sunnygumber.cctvaivisionlab.models.AndroidModelManager
import com.sunnygumber.cctvaivisionlab.models.ModelCatalog
import com.sunnygumber.cctvaivisionlab.relations.RelateAnythingEngine
import com.sunnygumber.cctvaivisionlab.tracking.InferenceScheduler
import com.sunnygumber.cctvaivisionlab.tracking.IoUTracker
import com.sunnygumber.cctvaivisionlab.tracking.RelationSmoother
import com.sunnygumber.cctvaivisionlab.tracking.deduplicateDetections
import com.sunnygumber.cctvaivisionlab.tracking.resolveRelationBoxes
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

enum class SourceMode {
    CAMERA,
    IMAGE,
}

enum class DisplayMode {
    ALL,
    HUMAN_VEHICLE,
}

data class VisionUiState(
    val sourceMode: SourceMode = SourceMode.CAMERA,
    val lensFacing: LensFacing = LensFacing.BACK,
    val imageUri: String? = null,
    val models: Map<String, ModelStatus> = emptyMap(),
    val modelsReady: Boolean = false,
    val preparingModels: Boolean = false,
    val modelDownloadLabel: String = "",
    val modelDownloadProgress: Float = 0f,
    val running: Boolean = false,
    val liveSceneEnabled: Boolean = false,
    val debugOverlay: Boolean = false,
    val displayMode: DisplayMode = DisplayMode.ALL,
    val detectionThreshold: Float = 0.60f,
    val relationThreshold: Float = 0.56f,
    val relationCadenceMs: Long = 1500L,
    val detections: List<Detection> = emptyList(),
    val relations: List<SceneRelation> = emptyList(),
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val detectorMetrics: InferenceMetrics? = null,
    val relationMetrics: InferenceMetrics? = null,
    val relationUpdates: Int = 0,
    val activeTracks: Int = 0,
    val statusMessage: String = "Prepare the AI models to begin.",
    val error: String? = null,
)

class VisionViewModel(application: Application) : AndroidViewModel(application) {
    private val modelManager = AndroidModelManager(application)
    private val tracker = IoUTracker()
    private val smoother = RelationSmoother()
    private val scheduler = InferenceScheduler()
    private val processingFrame = AtomicBoolean(false)
    private val generation = AtomicLong(0)

    private var detector: YoloXDetector? = null
    private var relationEngine: RelateAnythingEngine? = null

    private val _state = MutableStateFlow(VisionUiState())
    val state: StateFlow<VisionUiState> = _state.asStateFlow()

    init {
        refreshModelState()
    }

    fun prepareModels() {
        if (_state.value.preparingModels) return

        viewModelScope.launch {
            _state.value = _state.value.copy(
                preparingModels = true,
                error = null,
                statusMessage = "Preparing native AI models…",
            )

            try {
                for (descriptor in ModelCatalog.required) {
                    val result = modelManager.ensureAvailable(descriptor) { update ->
                        val progress = update.totalBytes?.takeIf { it > 0L }?.let { total ->
                            (update.downloadedBytes.toDouble() / total.toDouble()).toFloat()
                        } ?: 0f

                        _state.value = _state.value.copy(
                            models = _state.value.models + (descriptor.id to update),
                            modelDownloadLabel = descriptor.id,
                            modelDownloadProgress = progress.coerceIn(0f, 1f),
                            statusMessage = when (update.state) {
                                ModelState.DOWNLOADING -> "Downloading ${descriptor.id}…"
                                ModelState.READY -> "${descriptor.id} stored locally."
                                ModelState.FAILED -> "Download failed: ${descriptor.id}"
                                ModelState.MISSING -> "Waiting for ${descriptor.id}"
                            },
                        )
                    }

                    if (result.state != ModelState.READY) {
                        throw IllegalStateException(
                            result.error ?: "Could not prepare ${descriptor.id}",
                        )
                    }
                }

                loadEngines()
                _state.value = _state.value.copy(
                    modelsReady = true,
                    preparingModels = false,
                    modelDownloadLabel = "",
                    modelDownloadProgress = 1f,
                    statusMessage = "Models are ready locally. Start the camera AI.",
                )
            } catch (error: Throwable) {
                _state.value = _state.value.copy(
                    modelsReady = false,
                    preparingModels = false,
                    error = error.message ?: "Model preparation failed.",
                    statusMessage = "AI models are not ready.",
                )
            }
        }
    }

    private fun refreshModelState() {
        viewModelScope.launch {
            val statuses = ModelCatalog.required.associate { descriptor ->
                descriptor.id to modelManager.status(descriptor)
            }
            val ready = statuses.values.all { it.state == ModelState.READY }
            _state.value = _state.value.copy(
                models = statuses,
                modelsReady = ready,
                statusMessage = if (ready) {
                    "Cached models found. Initializing native inference…"
                } else {
                    "First run: download the AI models once. Later launches reuse local files."
                },
            )

            if (ready) {
                try {
                    loadEngines()
                    _state.value = _state.value.copy(
                        modelsReady = true,
                        statusMessage = "Cached models loaded. Ready.",
                    )
                } catch (error: Throwable) {
                    _state.value = _state.value.copy(
                        modelsReady = false,
                        error = error.message,
                        statusMessage = "Cached models exist but native inference could not initialize.",
                    )
                }
            }
        }
    }

    private suspend fun loadEngines() = withContext(Dispatchers.Default) {
        detector?.close()
        relationEngine?.close()

        val detectorFile = modelManager.status(ModelCatalog.yoloxSmall).localFile
            ?: error("YOLOX-S model is not available locally")
        val relationFile = modelManager.status(ModelCatalog.relateAnything).localFile
            ?: error("RelateAnything model is not available locally")
        val bankFile = modelManager.status(ModelCatalog.predicateBank).localFile
            ?: error("Predicate bank is not available locally")

        detector = YoloXDetector(detectorFile, preferNnapi = true)
        relationEngine = RelateAnythingEngine(
            modelFile = relationFile,
            predicateBankFile = bankFile,
            preferNnapi = true,
        )
    }

    fun clearModels() {
        if (processingFrame.get()) {
            _state.value = _state.value.copy(
                running = false,
                statusMessage = "AI paused. Wait for the current inference to finish, then clear models.",
            )
            return
        }

        viewModelScope.launch {
            stopAndReset("Clearing locally stored models…")
            withContext(Dispatchers.Default) {
                detector?.close()
                relationEngine?.close()
                detector = null
                relationEngine = null
            }
            ModelCatalog.required.forEach { modelManager.clear(it) }
            _state.value = VisionUiState(
                statusMessage = "Local models cleared. Download them again when required.",
            )
            refreshModelState()
        }
    }

    fun setSourceMode(mode: SourceMode) {
        if (_state.value.sourceMode == mode) return
        stopAndReset(if (mode == SourceMode.CAMERA) "Camera mode." else "Image mode.")
        _state.value = _state.value.copy(sourceMode = mode)
    }

    fun setImage(uri: Uri) {
        _state.value = _state.value.copy(
            sourceMode = SourceMode.IMAGE,
            imageUri = uri.toString(),
            statusMessage = "Image selected. Tap Analyze Image.",
            error = null,
        )
        resetTracking()
    }

    fun analyseSelectedImage() {
        val uriText = _state.value.imageUri ?: return
        if (!_state.value.modelsReady) {
            _state.value = _state.value.copy(statusMessage = "Prepare the AI models first.")
            return
        }
        if (!processingFrame.compareAndSet(false, true)) return

        val requestGeneration = generation.get()
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val frame = ImageFrameLoader.load(
                    getApplication<Application>().contentResolver,
                    Uri.parse(uriText),
                )
                processFrame(
                    frame = frame,
                    forceRelation = _state.value.liveSceneEnabled,
                    isLive = false,
                    requestGeneration = requestGeneration,
                )
            } catch (error: Throwable) {
                _state.value = _state.value.copy(
                    error = error.message ?: "Image analysis failed.",
                    statusMessage = "Image analysis failed.",
                )
            } finally {
                processingFrame.set(false)
            }
        }
    }

    fun shouldCaptureCameraFrame(): Boolean {
        val snapshot = _state.value
        return snapshot.sourceMode == SourceMode.CAMERA &&
            snapshot.running &&
            snapshot.modelsReady &&
            detector != null &&
            !processingFrame.get()
    }

    fun onCameraFrame(frame: FrameData) {
        val snapshot = _state.value
        if (
            snapshot.sourceMode != SourceMode.CAMERA ||
            !snapshot.running ||
            !snapshot.modelsReady ||
            detector == null
        ) {
            return
        }

        if (!processingFrame.compareAndSet(false, true)) return
        val requestGeneration = generation.get()

        viewModelScope.launch(Dispatchers.Default) {
            try {
                processFrame(
                    frame = frame,
                    forceRelation = false,
                    isLive = true,
                    requestGeneration = requestGeneration,
                )
            } catch (error: Throwable) {
                _state.value = _state.value.copy(
                    running = false,
                    error = error.message ?: "Native inference failed.",
                    statusMessage = "AI paused after an inference error.",
                )
            } finally {
                processingFrame.set(false)
            }
        }
    }

    private suspend fun processFrame(
        frame: FrameData,
        forceRelation: Boolean,
        isLive: Boolean,
        requestGeneration: Long,
    ) {
        val currentDetector = detector ?: error("Detector is not initialized")
        val snapshot = _state.value

        scheduler.markDetectorStarted(frame.timestampNs)
        val detectionResult = try {
            currentDetector.detect(frame, snapshot.detectionThreshold)
        } finally {
            scheduler.markDetectorFinished(System.nanoTime())
        }

        if (requestGeneration != generation.get()) return

        val deduplicated = deduplicateDetections(detectionResult.detections, 0.45f, 0.72f)
        val tracked = if (isLive) tracker.update(deduplicated) else deduplicated

        var relations = if (snapshot.liveSceneEnabled) {
            resolveRelationBoxes(smoother.values(), tracked)
        } else {
            emptyList()
        }
        var relationMetrics = _state.value.relationMetrics
        var relationUpdates = _state.value.relationUpdates

        val stableCandidates = tracked.filter { it.trackAge >= 2 || !isLive }
        val relationDue = forceRelation ||
            (
                snapshot.liveSceneEnabled &&
                    stableCandidates.size >= 2 &&
                    scheduler.shouldRunRelation(System.nanoTime(), snapshot.relationCadenceMs)
                )

        if (relationDue && stableCandidates.size >= 2) {
            if (requestGeneration != generation.get()) return
            val currentRelationEngine = relationEngine
                ?: error("Relation engine is not initialized")
            val started = System.nanoTime()
            scheduler.markRelationStarted(started)
            try {
                val result = currentRelationEngine.analyse(
                    frame = frame,
                    detections = stableCandidates,
                    confidenceThreshold = snapshot.relationThreshold,
                )
                relations = if (isLive) {
                    smoother.update(result.relations)
                } else {
                    result.relations
                }
                relationMetrics = result.metrics
                relationUpdates += 1
            } finally {
                scheduler.markRelationFinished(System.nanoTime())
            }
        }

        if (requestGeneration != generation.get()) return

        _state.value = _state.value.copy(
            detections = tracked,
            relations = relations,
            frameWidth = frame.width,
            frameHeight = frame.height,
            detectorMetrics = detectionResult.metrics,
            relationMetrics = relationMetrics,
            relationUpdates = relationUpdates,
            activeTracks = if (isLive) tracker.activeTrackCount else tracked.size,
            error = null,
            statusMessage = when {
                snapshot.liveSceneEnabled && relations.isNotEmpty() ->
                    "Live Scene AI: ${relations.size} relationship(s)."
                snapshot.liveSceneEnabled ->
                    "Live Scene AI active; waiting for a relationship above threshold."
                isLive ->
                    "Object detection and tracking active."
                else ->
                    "Image analysis complete."
            },
        )
    }

    fun toggleRunning() {
        if (!_state.value.modelsReady) {
            _state.value = _state.value.copy(statusMessage = "Prepare the AI models first.")
            return
        }
        val running = !_state.value.running
        if (!running) resetTracking()
        _state.value = _state.value.copy(
            running = running,
            statusMessage = if (running) "Native camera AI started." else "Camera AI paused.",
        )
    }

    fun toggleLiveScene() {
        if (!_state.value.modelsReady) return
        val enabled = !_state.value.liveSceneEnabled
        smoother.reset()
        scheduler.reset()
        _state.value = _state.value.copy(
            liveSceneEnabled = enabled,
            relations = emptyList(),
            relationMetrics = null,
            relationUpdates = 0,
            statusMessage = if (enabled) {
                "Live Scene AI enabled. Detector boxes are hidden unless Debug Overlay is on."
            } else {
                "Live Scene AI disabled. Object detection continues."
            },
        )
    }

    fun toggleDebugOverlay() {
        _state.value = _state.value.copy(debugOverlay = !_state.value.debugOverlay)
    }

    fun switchCamera() {
        resetTracking()
        _state.value = _state.value.copy(
            lensFacing = if (_state.value.lensFacing == LensFacing.BACK) {
                LensFacing.FRONT
            } else {
                LensFacing.BACK
            },
            statusMessage = "Camera switched.",
        )
    }

    fun setDetectionThreshold(value: Float) {
        resetTracking()
        _state.value = _state.value.copy(
            detectionThreshold = value.coerceIn(0.08f, 0.95f),
            statusMessage = "Detection threshold updated.",
        )
    }

    fun setRelationThreshold(value: Float) {
        smoother.reset()
        scheduler.reset()
        _state.value = _state.value.copy(
            relationThreshold = value.coerceIn(0.20f, 0.95f),
            relations = emptyList(),
            relationUpdates = 0,
            statusMessage = "Relationship threshold updated.",
        )
    }

    fun setRelationCadence(valueMs: Long) {
        scheduler.reset()
        _state.value = _state.value.copy(
            relationCadenceMs = valueMs.coerceIn(1000L, 3000L),
            statusMessage = "Relation cadence updated.",
        )
    }

    fun setDisplayMode(mode: DisplayMode) {
        _state.value = _state.value.copy(displayMode = mode)
    }

    private fun stopAndReset(message: String) {
        resetTracking()
        _state.value = _state.value.copy(
            running = false,
            relations = emptyList(),
            relationUpdates = 0,
            statusMessage = message,
        )
    }

    private fun resetTracking() {
        generation.incrementAndGet()
        tracker.reset()
        smoother.reset()
        scheduler.reset()
        _state.value = _state.value.copy(
            detections = emptyList(),
            relations = emptyList(),
            activeTracks = 0,
            detectorMetrics = null,
            relationMetrics = null,
            relationUpdates = 0,
        )
    }

    override fun onCleared() {
        runBlocking(Dispatchers.Default) {
            detector?.close()
            relationEngine?.close()
        }
        super.onCleared()
    }
}
