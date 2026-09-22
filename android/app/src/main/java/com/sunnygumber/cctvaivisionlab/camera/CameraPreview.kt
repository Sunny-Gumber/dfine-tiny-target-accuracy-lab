package com.sunnygumber.cctvaivisionlab.camera

import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

@Composable
fun CameraPreview(
    lensFacing: Int,
    onImage: (ImageProxy, Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val latestOnImage by rememberUpdatedState(onImage)
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }

    DisposableEffect(lensFacing, lifecycleOwner) {
        val analyzerExecutor = Executors.newSingleThreadExecutor()
        val future = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null

        val listener = Runnable {
            val cameraProvider = future.get()
            provider = cameraProvider

            val preview = Preview.Builder()
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 360))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT
            analysis.setAnalyzer(analyzerExecutor) { image ->
                try {
                    latestOnImage(image, isFront)
                } finally {
                    image.close()
                }
            }

            val selector = CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()

            runCatching {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    selector,
                    preview,
                    analysis,
                )
            }
        }

        future.addListener(listener, ContextCompat.getMainExecutor(context))

        onDispose {
            provider?.unbindAll()
            analyzerExecutor.shutdownNow()
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier,
    )
}
