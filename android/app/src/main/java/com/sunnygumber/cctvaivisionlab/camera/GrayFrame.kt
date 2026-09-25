package com.sunnygumber.cctvaivisionlab.camera

import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat

data class GrayFrame(
    val mat: Mat,
    val timestampNs: Long,
) : AutoCloseable {
    val width: Int get() = mat.cols()
    val height: Int get() = mat.rows()

    fun copy(): GrayFrame = GrayFrame(mat.clone(), timestampNs)

    override fun close() {
        mat.release()
    }
}

object GrayFrameConverter {
    @Volatile
    private var loaded = false

    @Synchronized
    private fun ensureLoaded() {
        if (loaded) return
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
        loaded = true
    }

    fun fromImageProxy(
        image: ImageProxy,
        mirrorHorizontally: Boolean,
    ): GrayFrame {
        ensureLoaded()

        val width = image.width
        val height = image.height
        val plane = image.planes[0]
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val source = plane.buffer.duplicate()
        val luma = ByteArray(width * height)

        if (pixelStride == 1 && rowStride == width) {
            source.position(0)
            source.get(luma, 0, luma.size)
        } else {
            var destination = 0
            for (row in 0 until height) {
                val rowStart = row * rowStride
                for (column in 0 until width) {
                    luma[destination++] = source.get(rowStart + column * pixelStride)
                }
            }
        }

        var current = Mat(height, width, CvType.CV_8UC1)
        current.put(0, 0, luma)

        val rotation = image.imageInfo.rotationDegrees
        if (rotation != 0) {
            val rotated = Mat()
            val code = when (rotation) {
                90 -> Core.ROTATE_90_CLOCKWISE
                180 -> Core.ROTATE_180
                270 -> Core.ROTATE_90_COUNTERCLOCKWISE
                else -> null
            }
            if (code != null) {
                Core.rotate(current, rotated, code)
                current.release()
                current = rotated
            } else {
                rotated.release()
            }
        }

        if (mirrorHorizontally) {
            val mirrored = Mat()
            Core.flip(current, mirrored, 1)
            current.release()
            current = mirrored
        }

        return GrayFrame(
            mat = current,
            timestampNs = image.imageInfo.timestamp,
        )
    }
}
