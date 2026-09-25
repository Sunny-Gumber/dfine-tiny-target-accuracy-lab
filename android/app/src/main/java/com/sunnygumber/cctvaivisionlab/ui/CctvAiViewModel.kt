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
import com.sunnygumber.cctvaivisionlab.core.ModelDescriptor
import com.sunnygumber.cctvaivisionlab.core.ModelState
import com.sunnygumber.cctvaivisionlab.core.ModelStatus
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import com.sunnygumber.cctvaivisionlab.inference.RelateAnythingEngine
import com.sunnygumber.cctvaivisionlab.inference.YoloXDetector
import com.sunnygumber.cctvaivisionlab.models.LocalModelManager
import com.sunnygumber.cctvaivisionlab.models.ModelCatalog
import com.sunnygumber.cctvaivisionlab.tracking.IoUTracker
import com.sunnygumber.cctvaivisionlab.tracking.RelationPolicy
import com.sunnygumber.cctvaivisionlab.tracking.RelationSmoother
import com.sunnygumber.cctvaivisionlab.tracking.deduplicateDetections
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max

enum class SourceMode {
    CAMERA,
    IMAGE,
}

enum class DisplayFilter {
    ALL,
    HUMAN_VEHICLE,
}

enum class DetectorProfile(
    val displayName: String,
    val inputSize: Int,
) {
    FAST_NANO("YOLOX Nano · 416", 416),
    ACCURATE_SMALL("YOLOX Small · 640", 640),
}

data class CctvAiUiState(
    val sourceMode: SourceMode = SourceMode.CAMERA,
    val lensFacing: Int = CameraSelector.LENS_FACING_BACK,
    val aiEnabled: Boolean = true,
    val sceneEnabled: Boolean = false,
    val debugOverlay: Boolean = false,
    val displayFilter: DisplayFilter = DisplayFilter.ALL,
    val detectorProfile: DetectorProfile = DetectorProfile.ACCURATE_SMALL,
    val detectorThreshold: Float = 0.55f,
    val relationThreshold: Float = 0.55f,
    val relationCadenceMs: Long = 3_000L,
    val effectiveRelationCadenceMs: Long = 3_000L,
    val detections: List<Detection> = emptyList(),
    val relations: List<SceneRelation> = emptyList(),
    val relationCandidateCount: Int = 0,
    val activeTracks: Int = 0,
    val relationUpdates: Int = 0,
    val detectorMetrics: InferenceMetrics? = null,
    val relationMetrics: InferenceMetrics? = null,
    val nanoLastInferenceMs: Double? = null,
    val smallLastInferenceMs: Double? = null,
    val detectorModel: ModelStatus? = null,
    val nanoModel: ModelStatus? = null,
    val smallModel: ModelStatus? = null,
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
    private val detectorMutex = Mutex()
    private val detectorGeneration = AtomicInteger(0)

    @Volatile
    private var detector: YoloXDetector? = null
    private var relationEngine: RelateAnythingEngine? = null
    private var currentRelations: List<SceneRelation> = emptyList()

    private val _state = MutableStateFlow(CctvAiUiState())
    val state: StateFlow<CctvAiUiState> = _state.asStateFlow()

    init {
        prepareDetector(_state.value.detectorProfile)
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

    fun setDetectorProfile(profile: DetectorProfile) {
        if (_state.value.detectorProfile == profile) return
        resetLiveState()
        _state.value = _state.value.copy(
            detectorProfile = profile,
            detectorModel = null,
            status = "Preparing ${profile.displayName}…",
            error = null,
        )
        prepareDetector(profile)
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
                relationCandidateCount = 0,
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
        val snapshot = _state.value
        _state.value = snapshot.copy(
            relationCadenceMs = valueMs,
            effectiveRelationCadenceMs = calculateEffectiveCadence(
                requestedMs = valueMs,
                detectorMetrics = snapshot.detectorMetrics,
                relationMetrics = snapshot.relationMetrics,
            ),
        )
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
        resetLiveState()
        detectorGeneration.incrementAndGet()
        viewModelScope.launch(Dispatchers.IO) {
            detectorMutex.withLock {
                runCatching { detector?.close() }
                detector = null
            }
            runCatching { relationEngine?.close() }
            relationEngine = null
            ModelCatalog.all.forEach { modelManager.clear(it) }
            _state.value = _state.value.copy(
                detectorModel = null,
                nanoModel = null,
                smallModel = null,
                relationModel = null,
                predicateBank = null,
                relationMetrics = null,
                detectorMetrics = null,
                nanoLastInferenceMs = null,
                smallLastInferenceMs = null,
                status = "Local AI models cleared. Downloading active detector again…",
                error = null,
            )
            prepareDetector(_state.value.detectorProfile)
            if (_state.value.sceneEnabled) prepareRelationEngine()
        }
    }

    fun refreshModelStates() {
        viewModelScope.launch(Dispatchers.IO) {
            val nanoStatus = modelManager.status(ModelCatalog.yoloXNano)
            val smallStatus = modelManager.status(ModelCatalog.yoloXSmall)
            val relationStatus = modelManager.status(ModelCatalog.relateAnything)
            val bankStatus = modelManager.status(ModelCatalog.predicateBank)
            val active = if (_state.value.detectorProfile == DetectorProfile.FAST_NANO) {
                nanoStatus
            } else {
                smallStatus
            }
            _state.value = _state.value.copy(
                detectorModel = active,
                nanoModel = nanoStatus,
                smallModel = smallStatus,
                relationModel = relationStatus,
                predicateBank = bankStatus,
            )
        }
    }

    private fun prepareDetector(profile: DetectorProfile) {
        val generation = detectorGeneration.incrementAndGet()
        val descriptor = detectorDescriptor(profile)

        viewModelScope.launch(Dispatchers.IO) {
            detectorMutex.withLock {
                runCatching { detector?.close() }
                detector = null
            }

            val status = modelManager.ensureAvailable(descriptor) { progress ->
                updateDetectorModelStatus(profile, progress)
                _state.value = _state.value.copy(
                    status = modelMessage(profile.displayName, progress),
                )
            }

            if (generation != detectorGeneration.get()) return@launch

            if (status.state != ModelState.READY || status.localFile == null) {
                setError(status.error ?: "${profile.displayName} could not be prepared.")
                return@launch
            }

            try {
                val newDetector = YoloXDetector(
                    modelFile = status.localFile,
                    inputSize = profile.inputSize,
                    preferNnapi = true,
                )
                if (generation != detectorGeneration.get()) {
                    newDetector.close()
                    return@launch
                }

                detectorMutex.withLock {
                    detector = newDetector
                }
                updateDetectorModelStatus(profile, status)
                _state.value = _state.value.copy(
                    detectorModel = status,
                    status = "${profile.displayName} ready.",
                    error = null,
                )
            } catch (error: Throwable) {
                setError("Could not initialize ${profile.displayName}: ${error.message}")
            }
        }
    }

    private fun updateDetectorModelStatus(profile: DetectorProfile, status: ModelStatus) {
        _state.value = when (profile) {
            DetectorProfile.FAST_NANO -> _state.value.copy(
                detectorModel = if (_state.value.detectorProfile == profile) status else _state.value.detectorModel,
                nanoModel = status,
            )
            DetectorProfile.ACCURATE_SMALL -> _state.value.copy(
                detectorModel = if (_state.value.detectorProfile == profile) status else _state.value.detectorModel,
                smallModel = status,
            )
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
        var relationCandidates: List<Detection> = emptyList()
        var runRelation = false

        try {
            val snapshot = _state.value
            val detectionResult = detectorMutex.withLock {
                detector?.detect(frame, snapshot.detectorThreshold)
            } ?: return

            val deduplicated = deduplicateDetections(detectionResult.detections)
            val tracked = if (live) tracker.update(deduplicated) else deduplicated

            if (live) {
                currentRelations = smoother.resolveBoxes(currentRelations, tracked)
            }

            val effectiveCadence = calculateEffectiveCadence(
                requestedMs = snapshot.relationCadenceMs,
                detectorMetrics = detectionResult.metrics,
                relationMetrics = snapshot.relationMetrics,
            )
            val benchmarkState = when (snapshot.detectorProfile) {
                DetectorProfile.FAST_NANO -> snapshot.copy(
                    nanoLastInferenceMs = detectionResult.metrics.inferenceMs,
                )
                DetectorProfile.ACCURATE_SMALL -> snapshot.copy(
                    smallLastInferenceMs = detectionResult.metrics.inferenceMs,
                )
            }

            _state.value = benchmarkState.copy(
                detections = tracked,
                relations = currentRelations,
                activeTracks = if (live) tracker.activeTrackCount else 0,
                detectorMetrics = detectionResult.metrics,
                effectiveRelationCadenceMs = effectiveCadence,
                frameWidth = frame.width,
                frameHeight = frame.height,
                status = if (snapshot.sceneEnabled) "Live Scene AI active." else "Object AI active.",
                error = null,
            )

            val relation = relationEngine
            if (snapshot.sceneEnabled && relation != null) {
                relationCandidates = selectRelationCandidates(tracked, live)
                _state.value = _state.value.copy(relationCandidateCount = relationCandidates.size)

                if (relationCandidates.size >= 2) {
                    if (!live || scheduler.shouldRunRelation(frame.timestampNs, effectiveCadence)) {
                        if (live) scheduler.markRelationStarted(frame.timestampNs)
                        runRelation = true
                    }
                }
            } else {
                _state.value = _state.value.copy(relationCandidateCount = 0)
            }
        } catch (error: Throwable) {
            setError("Detector inference failed: ${error.message}")
        } finally {
            if (live) scheduler.markDetectorFinished(frame.timestampNs)
        }

        if (!runRelation) return

        val relation = relationEngine
        if (relation == null) {
            if (live) scheduler.markRelationFinished(System.nanoTime())
            return
        }

        try {
            val relationResult = relation.analyse(
                frame = frame,
                detections = relationCandidates,
                confidenceThreshold = _state.value.relationThreshold,
            )
            val plausible = RelationPolicy.filterAndRank(
                relations = relationResult.relations,
                maxRelations = 3,
            )
            val stable = if (live) {
                smoother.update(plausible)
            } else {
                plausible
            }

            if (!live || _state.value.sceneEnabled) {
                currentRelations = stable
                val snapshot = _state.value
                val effectiveCadence = calculateEffectiveCadence(
                    requestedMs = snapshot.relationCadenceMs,
                    detectorMetrics = snapshot.detectorMetrics,
                    relationMetrics = relationResult.metrics,
                )
                _state.value = snapshot.copy(
                    relations = stable,
                    relationMetrics = relationResult.metrics,
                    effectiveRelationCadenceMs = effectiveCadence,
                    relationUpdates = snapshot.relationUpdates + 1,
                    status = if (stable.isEmpty()) {
                        "Scene AI active · no plausible relationship crossed the threshold."
                    } else {
                        "Scene AI active · ${stable.size} validated relationship(s)."
                    },
                    error = null,
                )
            }
        } catch (error: Throwable) {
            setError("Relation inference failed: ${error.message}")
        } finally {
            if (live) scheduler.markRelationFinished(System.nanoTime())
        }
    }

    private fun selectRelationCandidates(
        detections: List<Detection>,
        live: Boolean,
    ): List<Detection> {
        val eligible = (if (live) detections.filter { it.trackAge >= 2 } else detections)
            .sortedByDescending { it.confidence }

        if (eligible.size <= MAX_RELATION_CANDIDATES) return eligible

        val persons = eligible.filter { it.classId == 0 }.take(3)
        if (persons.isEmpty()) return eligible.take(MAX_RELATION_CANDIDATES)

        val personIds = persons.mapNotNull { it.trackId }.toSet()
        val others = eligible
            .filter { candidate ->
                candidate.classId != 0 || candidate.trackId !in personIds
            }
            .sortedWith(
                compareBy<Detection> { candidate ->
                    persons.minOf { person ->
                        normalizedCenterDistance(person, candidate)
                    }
                }.thenByDescending { it.confidence },
            )

        return (persons + others)
            .distinctBy { it.trackId ?: ((it.classId + 1) * 100_000 + it.box.centerX.toInt()) }
            .take(MAX_RELATION_CANDIDATES)
    }

    private fun normalizedCenterDistance(a: Detection, b: Detection): Double {
        val distance = hypot(
            (a.box.centerX - b.box.centerX).toDouble(),
            (a.box.centerY - b.box.centerY).toDouble(),
        )
        val scale = max(
            max(a.box.width, a.box.height),
            max(b.box.width, b.box.height),
        ).coerceAtLeast(1f)
        return distance / scale
    }

    private fun calculateEffectiveCadence(
        requestedMs: Long,
        detectorMetrics: InferenceMetrics?,
        relationMetrics: InferenceMetrics?,
    ): Long {
        if (detectorMetrics == null || relationMetrics == null) return requestedMs

        val cycleMs = totalMs(detectorMetrics) + totalMs(relationMetrics) + SCHEDULER_HEADROOM_MS
        val rounded = ceil(cycleMs / 500.0).toLong() * 500L
        return max(requestedMs, rounded).coerceAtMost(10_000L)
    }

    private fun totalMs(metrics: InferenceMetrics): Double =
        metrics.preprocessingMs + metrics.inferenceMs + metrics.postprocessingMs

    private fun detectorDescriptor(profile: DetectorProfile): ModelDescriptor = when (profile) {
        DetectorProfile.FAST_NANO -> ModelCatalog.yoloXNano
        DetectorProfile.ACCURATE_SMALL -> ModelCatalog.yoloXSmall
    }

    private fun resetLiveState() {
        scheduler.reset()
        tracker.reset()
        smoother.reset()
        currentRelations = emptyList()
        val snapshot = _state.value
        _state.value = snapshot.copy(
            detections = emptyList(),
            relations = emptyList(),
            relationCandidateCount = 0,
            activeTracks = 0,
            relationUpdates = 0,
            detectorMetrics = null,
            relationMetrics = null,
            effectiveRelationCadenceMs = snapshot.relationCadenceMs,
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
        detectorGeneration.incrementAndGet()
        runBlocking(Dispatchers.IO) {
            detectorMutex.withLock {
                runCatching { detector?.close() }
                detector = null
            }
            runCatching { relationEngine?.close() }
        }
        super.onCleared()
    }

    companion object {
        private const val MAX_RELATION_CANDIDATES = 5
        private const val SCHEDULER_HEADROOM_MS = 250.0
    }
}
