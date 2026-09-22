package com.sunnygumber.cctvaivisionlab.camera

import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.sunnygumber.cctvaivisionlab.core.FrameData
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

enum class LensFacing {
    BACK,
    FRONT,
}

@Composable
fun CameraPreview(
    facing: LensFacing,
    enabled: Boolean,
    onFrame: (FrameData) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
    }
    val analysisExecutor: ExecutorService = remember {
        Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "cctv-ai-camera-analysis").apply {
                priority = Thread.NORM_PRIORITY
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            analysisExecutor.shutdown()
        }
    }

    DisposableEffect(facing, enabled, lifecycleOwner) {
        if (!enabled) {
            onDispose { }
        } else {
            val providerFuture = ProcessCameraProvider.getInstance(context)
            var provider: ProcessCameraProvider? = null
            var preview: Preview? = null
            var analysis: ImageAnalysis? = null

            val listener = Runnable {
                try {
                    provider = providerFuture.get()

                    preview = Preview.Builder()
                        .setTargetResolution(Size(640, 360))
                        .build()
                        .also { it.surfaceProvider = previewView.surfaceProvider }

                    analysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 360))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                        .also { useCase ->
                            useCase.setAnalyzer(analysisExecutor) { image ->
                                try {
                                    onFrame(CameraFrameConverter.fromRgba(image))
                                } catch (_: Throwable) {
                                    // Drop malformed frames. The following frame remains usable.
                                } finally {
                                    image.close()
                                }
                            }
                        }

                    val selector = if (facing == LensFacing.BACK) {
                        CameraSelector.DEFAULT_BACK_CAMERA
                    } else {
                        CameraSelector.DEFAULT_FRONT_CAMERA
                    }

                    provider?.unbindAll()
                    provider?.bindToLifecycle(
                        lifecycleOwner,
                        selector,
                        preview,
                        analysis,
                    )
                } catch (_: Throwable) {
                    // Permission/device errors are surfaced by the screen state.
                }
            }

            providerFuture.addListener(listener, ContextCompat.getMainExecutor(context))

            onDispose {
                try {
                    analysis?.clearAnalyzer()
                    val useCases = listOfNotNull(preview, analysis).toTypedArray()
                    if (useCases.isNotEmpty()) provider?.unbind(*useCases)
                } catch (_: Throwable) {
                    // Lifecycle teardown must remain best-effort.
                }
            }
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier,
    )
}
