package com.sunnygumber.cctvaivisionlab.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.sunnygumber.cctvaivisionlab.camera.FrameConverter
import com.sunnygumber.cctvaivisionlab.core.DefaultFrameScheduler
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.FrameData
import com.sunnygumber.cctvaivisionlab.core.InferenceMetrics
import com.sunnygumber.cctvaivisionlab.core.ModelState
import com.sunnygumber.cctvaivisionlab.core.ModelStatus
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import com.sunnygumber.cctvaivisionlab.inference.RelateAnythingEngine
import com.sunnygumber.cctvaivisionlab.inference.YoloXDetector
import com.sunnygumber.cctvaivisionlab.models.LocalModelManager
import com.sunnygumber.cctvaivisionlab.models.ModelCatalog
import com.sunnygumber.cctvaivisionlab.tracking.IoUTracker
import com.sunnygumber.cctvaivisionlab.tracking.RelationSmoother
import com.sunnygumber.cctvaivisionlab.tracking.deduplicateDetections
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
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

enum class DisplayFilter {
    ALL,
    HUMAN_VEHICLE,
}

data class CctvAiUiState(
    val sourceMode: SourceMode = SourceMode.CAMERA,
    val lensFacing: Int = CameraSelector.LENS_FACING_BACK,
    val aiEnabled: Boolean = true,
    val sceneEnabled: Boolean = false,
    val debugOverlay: Boolean = false,
    val displayFilter: DisplayFilter = DisplayFilter.ALL,
    val detectorThreshold: Float = 0.60f,
    val relationThreshold: Float = 0.56f,
    val relationCadenceMs: Long = 1_500L,
    val detections: List<Detection> = emptyList(),
    val relations: List<SceneRelation> = emptyList(),
    val activeTracks: Int = 0,
    val relationUpdates: Int = 0,
    val detectorMetrics: InferenceMetrics? = null,
    val relationMetrics: InferenceMetrics? = null,
    val detectorModel: ModelStatus? = null,
    val relationModel: ModelStatus? = null,
    val predicateBank: ModelStatus? = null,
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val selectedBitmap: Bitmap? = null,
    val status: String = "Preparing detector model…",
    val error: String? = null,
)

class CctvAiViewModel(application: Application) : AndroidViewModel(application) {
    private val modelManager = LocalModelManager(application)
    private val tracker = IoUTracker()
    private val smoother = RelationSmoother()
    private val scheduler = DefaultFrameScheduler()

    private var detector: YoloXDetector? = null
    private var relationEngine: RelateAnythingEngine? = null
    private var currentRelations: List<SceneRelation> = emptyList()

    private val _state = MutableStateFlow(CctvAiUiState())
    val state: StateFlow<CctvAiUiState> = _state.asStateFlow()

    init {
        prepareDetector()
        refreshModelStates()
    }

    fun setSourceMode(mode: SourceMode) {
        _state.value = _state.value.copy(sourceMode = mode, error = null)
    }

    fun toggleLens() {
        val next = if (_state.value.lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        resetLiveState()
        _state.value = _state.value.copy(lensFacing = next)
    }

    fun setAiEnabled(enabled: Boolean) {
        if (!enabled) {
            resetLiveState()
        }
        _state.value = _state.value.copy(
            aiEnabled = enabled,
            sceneEnabled = if (enabled) _state.value.sceneEnabled else false,
            status = if (enabled) "Object AI active." else "AI paused.",
        )
    }

    fun setSceneEnabled(enabled: Boolean) {
        if (!enabled) {
            smoother.reset()
            currentRelations = emptyList()
            _state.value = _state.value.copy(
                sceneEnabled = false,
                relations = emptyList(),
                relationUpdates = 0,
                status = "Object AI active.",
            )
            return
        }

        _state.value = _state.value.copy(
            aiEnabled = true,
            sceneEnabled = true,
            status = "Preparing RelateAnything…",
            error = null,
        )
        prepareRelationEngine()
    }

    fun setDebugOverlay(enabled: Boolean) {
        _state.value = _state.value.copy(debugOverlay = enabled)
    }

    fun setDisplayFilter(filter: DisplayFilter) {
        _state.value = _state.value.copy(displayFilter = filter)
    }

    fun setDetectorThreshold(value: Float) {
        _state.value = _state.value.copy(detectorThreshold = value)
    }

    fun setRelationThreshold(value: Float) {
        _state.value = _state.value.copy(relationThreshold = value)
    }

    fun setRelationCadence(valueMs: Long) {
        _state.value = _state.value.copy(relationCadenceMs = valueMs)
    }

    fun onCameraImage(image: ImageProxy, mirrored: Boolean) {
        val snapshot = _state.value
        if (!snapshot.aiEnabled || detector == null) return

        val timestamp = image.imageInfo.timestamp
        if (!scheduler.shouldRunDetector(timestamp)) return
        scheduler.markDetectorStarted(timestamp)

        val frame = try {
            FrameConverter.fromImageProxy(image, mirrorHorizontally = mirrored)
        } catch (error: Throwable) {
            scheduler.markDetectorFinished(timestamp)
            setError("Camera frame conversion failed: ${error.message}")
            return
        }

        viewModelScope.launch(Dispatchers.Default) {
            processFrame(frame, live = true)
        }
    }

    fun analyseImage(uri: Uri) {
        _state.value = _state.value.copy(
            sourceMode = SourceMode.IMAGE,
            status = "Loading image…",
            error = null,
        )

        viewModelScope.launch(Dispatchers.Default) {
            try {
                val source = ImageDecoder.createSource(getApplication<Application>().contentResolver, uri)
                val bitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }
                val frame = FrameConverter.fromBitmap(bitmap)
                _state.value = _state.value.copy(selectedBitmap = bitmap)
                processFrame(frame, live = false)
            } catch (error: Throwable) {
                setError("Could not analyse image: ${error.message}")
            }
        }
    }

    fun clearAndRedownloadModels() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { detector?.close() }
            runCatching { relationEngine?.close() }
            detector = null
            relationEngine = null
            ModelCatalog.all.forEach { modelManager.clear(it) }
            _state.value = _state.value.copy(
                detectorModel = null,
                relationModel = null,
                predicateBank = null,
                relationMetrics = null,
                detectorMetrics = null,
                status = "Local AI models cleared. Downloading detector again…",
                error = null,
            )
            prepareDetector()
            if (_state.value.sceneEnabled) prepareRelationEngine()
        }
    }

    fun refreshModelStates() {
        viewModelScope.launch(Dispatchers.IO) {
            val detectorStatus = modelManager.status(ModelCatalog.yoloXSmall)
            val relationStatus = modelManager.status(ModelCatalog.relateAnything)
            val bankStatus = modelManager.status(ModelCatalog.predicateBank)
            _state.value = _state.value.copy(
                detectorModel = detectorStatus,
                relationModel = relationStatus,
                predicateBank = bankStatus,
            )
        }
    }

    private fun prepareDetector() {
        if (detector != null) return

        viewModelScope.launch(Dispatchers.IO) {
            val status = modelManager.ensureAvailable(ModelCatalog.yoloXSmall) { progress ->
                _state.value = _state.value.copy(
                    detectorModel = progress,
                    status = modelMessage("YOLOX-S", progress),
                )
            }

            if (status.state != ModelState.READY || status.localFile == null) {
                setError(status.error ?: "YOLOX-S model could not be prepared.")
                return@launch
            }

            try {
                detector = YoloXDetector(status.localFile, preferNnapi = true)
                _state.value = _state.value.copy(
                    detectorModel = status,
                    status = "YOLOX-S ready. Camera AI can start.",
                    error = null,
                )
            } catch (error: Throwable) {
                setError("Could not initialize YOLOX-S: ${error.message}")
            }
        }
    }

    private fun prepareRelationEngine() {
        if (relationEngine != null) return

        viewModelScope.launch(Dispatchers.IO) {
            val modelDeferred = async {
                modelManager.ensureAvailable(ModelCatalog.relateAnything) { progress ->
                    _state.value = _state.value.copy(
                        relationModel = progress,
                        status = modelMessage("RelateAnything", progress),
                    )
                }
            }
            val bankDeferred = async {
                modelManager.ensureAvailable(ModelCatalog.predicateBank) { progress ->
                    _state.value = _state.value.copy(predicateBank = progress)
                }
            }

            val model = modelDeferred.await()
            val bank = bankDeferred.await()
            if (
                model.state != ModelState.READY ||
                bank.state != ModelState.READY ||
                model.localFile == null ||
                bank.localFile == null
            ) {
                setError(model.error ?: bank.error ?: "RelateAnything assets could not be prepared.")
                return@launch
            }

            try {
                relationEngine = RelateAnythingEngine(
                    modelFile = model.localFile,
                    predicateBankFile = bank.localFile,
                    preferNnapi = true,
                )
                _state.value = _state.value.copy(
                    relationModel = model,
                    predicateBank = bank,
                    status = "Live Scene AI ready.",
                    error = null,
                )
            } catch (error: Throwable) {
                setError("Could not initialize RelateAnything: ${error.message}")
            }
        }
    }

    private suspend fun processFrame(frame: FrameData, live: Boolean) {
        val localDetector = detector
        if (localDetector == null) {
            if (live) scheduler.markDetectorFinished(frame.timestampNs)
            return
        }

        var relationCandidates: List<Detection> = emptyList()
        var runRelation = false

        try {
            val snapshot = _state.value
            val detectionResult = localDetector.detect(frame, snapshot.detectorThreshold)
            val deduplicated = deduplicateDetections(detectionResult.detections)
            val tracked = if (live) tracker.update(deduplicated) else deduplicated

            if (live) {
                currentRelations = smoother.resolveBoxes(currentRelations, tracked)
            }

            _state.value = _state.value.copy(
                detections = tracked,
                relations = currentRelations,
                activeTracks = if (live) tracker.activeTrackCount else 0,
                detectorMetrics = detectionResult.metrics,
                frameWidth = frame.width,
                frameHeight = frame.height,
                status = if (snapshot.sceneEnabled) "Live Scene AI active." else "Object AI active.",
                error = null,
            )

            val relation = relationEngine
            if (snapshot.sceneEnabled && relation != null) {
                relationCandidates = if (live) {
                    tracked.filter { it.trackAge >= 2 }
                } else {
                    tracked
                }.sortedByDescending { it.confidence }.take(8)

                if (relationCandidates.size >= 2) {
                    if (!live || scheduler.shouldRunRelation(frame.timestampNs, snapshot.relationCadenceMs)) {
                        if (live) scheduler.markRelationStarted(frame.timestampNs)
                        runRelation = true
                    }
                }
            }
        } catch (error: Throwable) {
            setError("Detector inference failed: ${error.message}")
        } finally {
            if (live) scheduler.markDetectorFinished(frame.timestampNs)
        }

        if (!runRelation) return

        val relation = relationEngine ?: return
        try {
            val relationResult = relation.analyse(
                frame = frame,
                detections = relationCandidates,
                confidenceThreshold = _state.value.relationThreshold,
            )
            val stable = if (live) {
                smoother.update(relationResult.relations)
            } else {
                relationResult.relations
            }
            currentRelations = stable
            _state.value = _state.value.copy(
                relations = stable,
                relationMetrics = relationResult.metrics,
                relationUpdates = _state.value.relationUpdates + 1,
                status = if (stable.isEmpty()) {
                    "Scene AI active · no relationship crossed the threshold."
                } else {
                    "Scene AI active · ${stable.size} relationship(s)."
                },
                error = null,
            )
        } catch (error: Throwable) {
            setError("Relation inference failed: ${error.message}")
        } finally {
            if (live) scheduler.markRelationFinished(System.nanoTime())
        }
    }

    private fun resetLiveState() {
        scheduler.reset()
        tracker.reset()
        smoother.reset()
        currentRelations = emptyList()
        _state.value = _state.value.copy(
            detections = emptyList(),
            relations = emptyList(),
            activeTracks = 0,
            relationUpdates = 0,
            detectorMetrics = null,
            relationMetrics = null,
        )
    }

    private fun modelMessage(name: String, status: ModelStatus): String {
        if (status.state != ModelState.DOWNLOADING) return "$name: ${status.state.name.lowercase()}"
        val total = status.totalBytes
        return if (total != null && total > 0) {
            val percent = (status.downloadedBytes * 100 / total).coerceIn(0, 100)
            "Downloading $name… $percent%"
        } else {
            "Downloading $name… ${status.downloadedBytes / 1_048_576} MB"
        }
    }

    private fun setError(message: String) {
        _state.value = _state.value.copy(error = message, status = message)
    }

    override fun onCleared() {
        runBlocking(Dispatchers.IO) {
            runCatching { detector?.close() }
            runCatching { relationEngine?.close() }
        }
        super.onCleared()
    }
}
