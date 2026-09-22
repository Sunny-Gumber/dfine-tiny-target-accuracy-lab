package com.sunnygumber.cctvaivisionlab.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Paint
import android.net.Uri
import android.widget.ImageView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sunnygumber.cctvaivisionlab.camera.CameraPreview
import com.sunnygumber.cctvaivisionlab.camera.LensFacing
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.InferenceMetrics
import com.sunnygumber.cctvaivisionlab.core.ModelState
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import kotlin.math.min

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CctvAiApp(viewModel: VisionViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    var cameraPermissionGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA,
            ) == PackageManager.PERMISSION_GRANTED,
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        cameraPermissionGranted = granted
    }

    val imageLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { uri ->
        if (uri != null) viewModel.setImage(uri)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("CCTV AI Vision Lab")
                        Text(
                            "Native Android · ONNX Runtime",
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                },
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
            ModelPanel(
                state = state,
                onPrepare = viewModel::prepareModels,
                onClear = viewModel::clearModels,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                ModeButton(
                    selected = state.sourceMode == SourceMode.CAMERA,
                    label = "Live Camera",
                    onClick = { viewModel.setSourceMode(SourceMode.CAMERA) },
                    modifier = Modifier.weight(1f),
                )
                ModeButton(
                    selected = state.sourceMode == SourceMode.IMAGE,
                    label = "Image",
                    onClick = { viewModel.setSourceMode(SourceMode.IMAGE) },
                    modifier = Modifier.weight(1f),
                )
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(16f / 9f)
                        .background(Color.Black),
                ) {
                    if (state.sourceMode == SourceMode.CAMERA) {
                        if (cameraPermissionGranted) {
                            CameraPreview(
                                facing = state.lensFacing,
                                enabled = true,
                                shouldCaptureFrame = viewModel::shouldCaptureCameraFrame,
                                onFrame = viewModel::onCameraFrame,
                                modifier = Modifier.fillMaxSize(),
                            )
                        } else {
                            Column(
                                modifier = Modifier
                                    .align(Alignment.Center)
                                    .padding(20.dp),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(10.dp),
                            ) {
                                Text(
                                    "Camera permission is required for live analysis.",
                                    color = Color.White,
                                )
                                Button(
                                    onClick = {
                                        permissionLauncher.launch(Manifest.permission.CAMERA)
                                    },
                                ) {
                                    Text("Allow Camera")
                                }
                            }
                        }
                    } else {
                        val uriText = state.imageUri
                        if (uriText == null) {
                            Text(
                                "Choose an image to analyse",
                                color = Color.White,
                                modifier = Modifier.align(Alignment.Center),
                            )
                        } else {
                            AndroidView(
                                factory = { imageContext ->
                                    ImageView(imageContext).apply {
                                        scaleType = ImageView.ScaleType.FIT_CENTER
                                    }
                                },
                                update = { imageView ->
                                    imageView.setImageURI(Uri.parse(uriText))
                                },
                                modifier = Modifier.fillMaxSize(),
                            )
                        }
                    }

                    VisionOverlay(
                        state = state,
                        mirrorHorizontally =
                            state.sourceMode == SourceMode.CAMERA &&
                                state.lensFacing == LensFacing.FRONT,
                        modifier = Modifier.fillMaxSize(),
                    )
                }
            }

            if (state.sourceMode == SourceMode.CAMERA) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Button(
                        onClick = viewModel::toggleRunning,
                        enabled = state.modelsReady,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text(if (state.running) "Pause AI" else "Start AI")
                    }
                    OutlinedButton(
                        onClick = viewModel::switchCamera,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text(
                            if (state.lensFacing == LensFacing.BACK) {
                                "Front Camera"
                            } else {
                                "Back Camera"
                            },
                        )
                    }
                }
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Button(
                        onClick = { imageLauncher.launch("image/*") },
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Choose Image")
                    }
                    Button(
                        onClick = viewModel::analyseSelectedImage,
                        enabled = state.imageUri != null && state.modelsReady,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Analyze Image")
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                ModeButton(
                    selected = state.liveSceneEnabled,
                    label = if (state.liveSceneEnabled) "Scene AI ON" else "Scene AI OFF",
                    onClick = viewModel::toggleLiveScene,
                    enabled = state.modelsReady,
                    modifier = Modifier.weight(1f),
                )
                ModeButton(
                    selected = state.debugOverlay,
                    label = "Debug Boxes",
                    onClick = viewModel::toggleDebugOverlay,
                    modifier = Modifier.weight(1f),
                )
            }

            ControlsPanel(state, viewModel)
            MetricsPanel(state)

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text("Status", style = MaterialTheme.typography.titleMedium)
                    Text(state.statusMessage)
                    state.error?.let {
                        Text(
                            text = it,
                            color = MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }

            RelationList(state.relations)
        }
    }
}

@Composable
private fun ModelPanel(
    state: VisionUiState,
    onPrepare: () -> Unit,
    onClear: () -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("AI Models", style = MaterialTheme.typography.titleMedium)
            state.models.values.sortedBy { it.descriptor.id }.forEach { model ->
                val label = when (model.state) {
                    ModelState.READY -> "Local"
                    ModelState.DOWNLOADING -> "Downloading"
                    ModelState.MISSING -> "Not downloaded"
                    ModelState.FAILED -> "Failed"
                }
                Text("${model.descriptor.id}: $label")
            }

            if (state.preparingModels) {
                LinearProgressIndicator(
                    progress = { state.modelDownloadProgress },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    if (state.modelDownloadLabel.isBlank()) {
                        "Preparing models…"
                    } else {
                        "Preparing ${state.modelDownloadLabel}"
                    },
                    style = MaterialTheme.typography.labelMedium,
                )
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = onPrepare,
                    enabled = !state.preparingModels && !state.modelsReady,
                ) {
                    Text(if (state.modelsReady) "Models Ready" else "Prepare Models")
                }
                OutlinedButton(
                    onClick = onClear,
                    enabled = !state.preparingModels && state.models.isNotEmpty(),
                ) {
                    Text("Clear Local Models")
                }
            }
            Text(
                "Models are downloaded once to app-private storage and reused on later launches.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun ControlsPanel(
    state: VisionUiState,
    viewModel: VisionViewModel,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("AI Controls", style = MaterialTheme.typography.titleMedium)

            Text("Detection confidence · ${(state.detectionThreshold * 100).toInt()}%")
            Slider(
                value = state.detectionThreshold,
                onValueChange = viewModel::setDetectionThreshold,
                valueRange = 0.08f..0.95f,
            )

            Text("Relationship confidence · ${(state.relationThreshold * 100).toInt()}%")
            Slider(
                value = state.relationThreshold,
                onValueChange = viewModel::setRelationThreshold,
                valueRange = 0.20f..0.95f,
            )

            Text("Relation cadence")
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(1000L to "1.0s", 1500L to "1.5s", 3000L to "3.0s")
                    .forEach { (value, label) ->
                        ModeButton(
                            selected = state.relationCadenceMs == value,
                            label = label,
                            onClick = { viewModel.setRelationCadence(value) },
                        )
                    }
            }

            Text("Object display")
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                ModeButton(
                    selected = state.displayMode == DisplayMode.ALL,
                    label = "All COCO",
                    onClick = { viewModel.setDisplayMode(DisplayMode.ALL) },
                )
                ModeButton(
                    selected = state.displayMode == DisplayMode.HUMAN_VEHICLE,
                    label = "Human + Vehicle",
                    onClick = { viewModel.setDisplayMode(DisplayMode.HUMAN_VEHICLE) },
                )
            }
        }
    }
}

@Composable
private fun MetricsPanel(state: VisionUiState) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text("Performance", style = MaterialTheme.typography.titleMedium)
            MetricLine("Detections", state.detections.size.toString())
            MetricLine("Active tracks", state.activeTracks.toString())
            MetricLine("Relationships", state.relations.size.toString())
            MetricLine("Relation updates", state.relationUpdates.toString())

            state.detectorMetrics?.let { metrics ->
                HorizontalDivider()
                MetricLine("Detector provider", metrics.provider.name)
                MetricLine("Detector inference", formatMs(metrics.inferenceMs))
                MetricLine(
                    "Detector total",
                    formatMs(
                        metrics.preprocessingMs +
                            metrics.inferenceMs +
                            metrics.postprocessingMs,
                    ),
                )
            }

            state.relationMetrics?.let { metrics ->
                HorizontalDivider()
                MetricLine("Relation provider", metrics.provider.name)
                MetricLine("Relation inference", formatMs(metrics.inferenceMs))
                MetricLine(
                    "Relation total",
                    formatMs(
                        metrics.preprocessingMs +
                            metrics.inferenceMs +
                            metrics.postprocessingMs,
                    ),
                )
            }
        }
    }
}

@Composable
private fun RelationList(relations: List<SceneRelation>) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text("Relationships", style = MaterialTheme.typography.titleMedium)
            if (relations.isEmpty()) {
                Text(
                    "No active relationship result.",
                    style = MaterialTheme.typography.bodyMedium,
                )
            } else {
                relations.take(8).forEach { relation ->
                    val subject = relation.subject.trackId?.let {
                        "#$it ${relation.subject.label}"
                    } ?: relation.subject.label
                    val objectName = relation.objectDetection.trackId?.let {
                        "#$it ${relation.objectDetection.label}"
                    } ?: relation.objectDetection.label
                    Text(
                        "$subject → ${relation.predicate} → $objectName  " +
                            "${(relation.score * 100).toInt()}%",
                    )
                }
            }
        }
    }
}

@Composable
private fun MetricLine(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Text(value, style = MaterialTheme.typography.labelLarge)
    }
}

@Composable
private fun ModeButton(
    selected: Boolean,
    label: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    if (selected) {
        Button(
            onClick = onClick,
            enabled = enabled,
            modifier = modifier,
        ) {
            Text(label)
        }
    } else {
        OutlinedButton(
            onClick = onClick,
            enabled = enabled,
            modifier = modifier,
        ) {
            Text(label)
        }
    }
}

@Composable
private fun VisionOverlay(
    state: VisionUiState,
    mirrorHorizontally: Boolean,
    modifier: Modifier = Modifier,
) {
    val filteredDetections = if (state.displayMode == DisplayMode.ALL) {
        state.detections
    } else {
        state.detections.filter { it.classId in setOf(0, 1, 2, 3, 5, 7) }
    }

    Canvas(modifier = modifier) {
        if (state.frameWidth <= 0 || state.frameHeight <= 0) return@Canvas

        val scale = min(
            size.width / state.frameWidth.toFloat(),
            size.height / state.frameHeight.toFloat(),
        )
        val contentWidth = state.frameWidth * scale
        val contentHeight = state.frameHeight * scale
        val offsetX = (size.width - contentWidth) / 2f
        val offsetY = (size.height - contentHeight) / 2f

        fun mapX(x: Float): Float {
            val sourceX = if (mirrorHorizontally) {
                state.frameWidth - x
            } else {
                x
            }
            return offsetX + sourceX * scale
        }

        fun mapY(y: Float): Float = offsetY + y * scale

        val showDetectorBoxes = !state.liveSceneEnabled || state.debugOverlay
        if (showDetectorBoxes) {
            filteredDetections.forEach { detection ->
                val left = mapX(
                    if (mirrorHorizontally) detection.box.right else detection.box.left,
                )
                val right = mapX(
                    if (mirrorHorizontally) detection.box.left else detection.box.right,
                )
                val top = mapY(detection.box.top)
                val bottom = mapY(detection.box.bottom)
                val color = when {
                    detection.classId == 0 -> Color(0xff58e2d3)
                    detection.classId in setOf(1, 2, 3, 5, 7) -> Color(0xfff4cf52)
                    else -> Color(0xff79bdf2)
                }

                drawRect(
                    color = color,
                    topLeft = Offset(minOf(left, right), top),
                    size = Size(kotlin.math.abs(right - left), bottom - top),
                    style = Stroke(width = 2.5f),
                )
            }
        }

        val nativePaint = Paint().apply {
            isAntiAlias = true
            textSize = 28f
            color = android.graphics.Color.WHITE
            setShadowLayer(4f, 0f, 0f, android.graphics.Color.BLACK)
        }
        val lineColor = Color(0xffffad66)

        state.relations.take(6).forEach { relation ->
            val sx = mapX(relation.subject.box.centerX)
            val sy = mapY(relation.subject.box.centerY)
            val ox = mapX(relation.objectDetection.box.centerX)
            val oy = mapY(relation.objectDetection.box.centerY)

            drawLine(
                color = lineColor,
                start = Offset(sx, sy),
                end = Offset(ox, oy),
                strokeWidth = 3f,
            )

            val midX = (sx + ox) / 2f
            val midY = (sy + oy) / 2f
            drawContext.canvas.nativeCanvas.drawText(
                "${relation.predicate} ${(relation.score * 100).toInt()}%",
                midX,
                midY,
                nativePaint,
            )
        }
    }
}

private fun formatMs(value: Double): String =
    if (value < 10.0) {
        "${"%.1f".format(value)} ms"
    } else {
        "${value.toInt()} ms"
    }
