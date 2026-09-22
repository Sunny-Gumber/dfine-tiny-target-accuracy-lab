package com.sunnygumber.cctvaivisionlab.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.sunnygumber.cctvaivisionlab.camera.CameraPreview
import com.sunnygumber.cctvaivisionlab.core.InferenceMetrics
import com.sunnygumber.cctvaivisionlab.core.ModelState
import com.sunnygumber.cctvaivisionlab.core.ModelStatus
import java.util.Locale

private val focusClasses = setOf(0, 1, 2, 3, 5, 7)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CctvAiApp(
    viewModel: CctvAiViewModel = viewModel(),
) {
    val context = LocalContext.current
    val state by viewModel.state.collectAsState()
    var cameraGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }

    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        cameraGranted = granted
    }

    val imagePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { uri ->
        if (uri != null) viewModel.analyseImage(uri)
    }

    val visibleDetections = remember(state.detections, state.displayFilter) {
        when (state.displayFilter) {
            DisplayFilter.ALL -> state.detections
            DisplayFilter.HUMAN_VEHICLE -> state.detections.filter { it.classId in focusClasses }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("CCTV AI Vision Lab") },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "Native Android AI",
                style = MaterialTheme.typography.headlineSmall,
            )
            Text(
                text = "CameraX + native ONNX Runtime · NNAPI first, CPU fallback",
                style = MaterialTheme.typography.bodyMedium,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                ModeButton(
                    label = "Live camera",
                    selected = state.sourceMode == SourceMode.CAMERA,
                    onClick = { viewModel.setSourceMode(SourceMode.CAMERA) },
                    modifier = Modifier.weight(1f),
                )
                ModeButton(
                    label = "Image",
                    selected = state.sourceMode == SourceMode.IMAGE,
                    onClick = {
                        viewModel.setSourceMode(SourceMode.IMAGE)
                        imagePicker.launch("image/*")
                    },
                    modifier = Modifier.weight(1f),
                )
            }

            VisionSurface(
                state = state,
                detections = visibleDetections,
                cameraGranted = cameraGranted,
                requestCamera = { cameraPermission.launch(Manifest.permission.CAMERA) },
                onImage = viewModel::onCameraImage,
            )

            if (state.sourceMode == SourceMode.IMAGE) {
                OutlinedButton(
                    onClick = { imagePicker.launch("image/*") },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Choose another image")
                }
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text("AI controls", style = MaterialTheme.typography.titleMedium)

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Button(
                            onClick = { viewModel.setAiEnabled(!state.aiEnabled) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text(if (state.aiEnabled) "Pause object AI" else "Start object AI")
                        }
                        Button(
                            onClick = { viewModel.setSceneEnabled(!state.sceneEnabled) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text(if (state.sceneEnabled) "Stop Scene AI" else "Start Scene AI")
                        }
                    }

                    if (state.sourceMode == SourceMode.CAMERA) {
                        OutlinedButton(
                            onClick = viewModel::toggleLens,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Text(
                                if (state.lensFacing == CameraSelector.LENS_FACING_BACK) {
                                    "Switch to front camera"
                                } else {
                                    "Switch to back camera"
                                },
                            )
                        }
                    }

                    LabeledSlider(
                        label = "Detector confidence",
                        value = state.detectorThreshold,
                        onValueChange = viewModel::setDetectorThreshold,
                        range = 0.20f..0.90f,
                    )
                    LabeledSlider(
                        label = "Relationship confidence",
                        value = state.relationThreshold,
                        onValueChange = viewModel::setRelationThreshold,
                        range = 0.30f..0.90f,
                    )

                    Text("Relation cadence", style = MaterialTheme.typography.labelLarge)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        listOf(1_000L to "1.0 s", 1_500L to "1.5 s", 3_000L to "3.0 s")
                            .forEach { (milliseconds, label) ->
                                ModeButton(
                                    label = label,
                                    selected = state.relationCadenceMs == milliseconds,
                                    onClick = { viewModel.setRelationCadence(milliseconds) },
                                    modifier = Modifier.weight(1f),
                                )
                            }
                    }

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("Debug detector boxes", style = MaterialTheme.typography.labelLarge)
                            Text(
                                "Scene AI hides raw detector boxes by default.",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        Switch(
                            checked = state.debugOverlay,
                            onCheckedChange = viewModel::setDebugOverlay,
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        ModeButton(
                            label = "All objects",
                            selected = state.displayFilter == DisplayFilter.ALL,
                            onClick = { viewModel.setDisplayFilter(DisplayFilter.ALL) },
                            modifier = Modifier.weight(1f),
                        )
                        ModeButton(
                            label = "Human + vehicle",
                            selected = state.displayFilter == DisplayFilter.HUMAN_VEHICLE,
                            onClick = { viewModel.setDisplayFilter(DisplayFilter.HUMAN_VEHICLE) },
                            modifier = Modifier.weight(1f),
                        )
                    }
                }
            }

            PerformanceCard(state)
            ModelCard(state, viewModel::clearAndRedownloadModels)
            RelationCard(state)

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text("Status", style = MaterialTheme.typography.titleMedium)
                    Text(state.status)
                    state.error?.let {
                        Text(
                            text = it,
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun VisionSurface(
    state: CctvAiUiState,
    detections: List<com.sunnygumber.cctvaivisionlab.core.Detection>,
    cameraGranted: Boolean,
    requestCamera: () -> Unit,
    onImage: (androidx.camera.core.ImageProxy, Boolean) -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(16f / 9f)
            .background(Color.Black),
        contentAlignment = Alignment.Center,
    ) {
        when (state.sourceMode) {
            SourceMode.CAMERA -> {
                if (cameraGranted) {
                    CameraPreview(
                        lensFacing = state.lensFacing,
                        onImage = onImage,
                        modifier = Modifier.fillMaxSize(),
                    )
                } else {
                    Button(onClick = requestCamera) {
                        Text("Allow camera")
                    }
                }
            }

            SourceMode.IMAGE -> {
                val bitmap = state.selectedBitmap
                if (bitmap == null) {
                    Text("Choose an image to analyse", color = Color.White)
                } else {
                    androidx.compose.foundation.Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Selected CCTV frame",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit,
                    )
                }
            }
        }

        VisionOverlay(
            frameWidth = state.frameWidth,
            frameHeight = state.frameHeight,
            detections = detections,
            relations = state.relations,
            showDetections = state.debugOverlay || !state.sceneEnabled,
            modifier = Modifier.fillMaxSize(),
        )
    }
}

@Composable
private fun PerformanceCard(state: CctvAiUiState) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(7.dp),
        ) {
            Text("Performance", style = MaterialTheme.typography.titleMedium)
            MetricLine("Detections", state.detections.size.toString())
            MetricLine("Active tracks", state.activeTracks.toString())
            MetricLine("Relationships", state.relations.size.toString())
            MetricLine("Relation updates", state.relationUpdates.toString())
            MetricsBlock("Detector", state.detectorMetrics)
            MetricsBlock("Relation", state.relationMetrics)
        }
    }
}

@Composable
private fun MetricsBlock(label: String, metrics: InferenceMetrics?) {
    HorizontalDivider()
    Text(label, style = MaterialTheme.typography.labelLarge)
    if (metrics == null) {
        Text("Waiting for inference", style = MaterialTheme.typography.bodySmall)
        return
    }
    MetricLine("Runtime", metrics.provider.name)
    MetricLine("Preprocess", formatMs(metrics.preprocessingMs))
    MetricLine("Inference", formatMs(metrics.inferenceMs))
    MetricLine("Postprocess", formatMs(metrics.postprocessingMs))
}

@Composable
private fun ModelCard(
    state: CctvAiUiState,
    clearAndRedownload: () -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(7.dp),
        ) {
            Text("Local AI models", style = MaterialTheme.typography.titleMedium)
            ModelLine("YOLOX-S", state.detectorModel)
            ModelLine("RelateAnything", state.relationModel)
            ModelLine("Predicate bank", state.predicateBank)
            Text(
                "READY models are loaded from app-private storage on the next launch; they are not downloaded again.",
                style = MaterialTheme.typography.bodySmall,
            )
            OutlinedButton(
                onClick = clearAndRedownload,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Clear and re-download models")
            }
        }
    }
}

@Composable
private fun ModelLine(label: String, status: ModelStatus?) {
    val detail = when {
        status == null -> "checking"
        status.state == ModelState.DOWNLOADING && status.totalBytes != null && status.totalBytes > 0 ->
            "downloading ${status.downloadedBytes * 100 / status.totalBytes}%"
        status.state == ModelState.DOWNLOADING ->
            "downloading ${status.downloadedBytes / 1_048_576} MB"
        status.state == ModelState.READY ->
            "READY · ${status.downloadedBytes / 1_048_576} MB local"
        status.state == ModelState.FAILED ->
            "FAILED · ${status.error ?: "unknown error"}"
        else -> status.state.name
    }
    MetricLine(label, detail)
}

@Composable
private fun RelationCard(state: CctvAiUiState) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .padding(12.dp)
                .heightIn(min = 70.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text("Scene relationships", style = MaterialTheme.typography.titleMedium)
            if (state.relations.isEmpty()) {
                Text("No relationship above the current threshold.")
            } else {
                state.relations.take(8).forEach { relation ->
                    val subject = relation.subject.trackId?.let { "#$it ${relation.subject.label}" }
                        ?: relation.subject.label
                    val objectName = relation.objectDetection.trackId?.let {
                        "#$it ${relation.objectDetection.label}"
                    } ?: relation.objectDetection.label
                    Text(
                        "$subject → ${relation.predicate} → $objectName · ${(relation.score * 100).toInt()}%",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
        }
    }
}

@Composable
private fun LabeledSlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    range: ClosedFloatingPointRange<Float>,
) {
    Column {
        Row(modifier = Modifier.fillMaxWidth()) {
            Text(label, modifier = Modifier.weight(1f), style = MaterialTheme.typography.labelLarge)
            Text("${(value * 100).toInt()}%")
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = range,
        )
    }
}

@Composable
private fun ModeButton(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    if (selected) {
        Button(onClick = onClick, modifier = modifier) { Text(label) }
    } else {
        OutlinedButton(onClick = onClick, modifier = modifier) { Text(label) }
    }
}

@Composable
private fun MetricLine(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Text(label, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
        Text(value, style = MaterialTheme.typography.bodySmall)
    }
}

private fun formatMs(value: Double): String =
    if (value <= 0.0) {
        "—"
    } else if (value < 10) {
        String.format(Locale.US, "%.1f ms", value)
    } else {
        "${value.toInt()} ms"
    }
