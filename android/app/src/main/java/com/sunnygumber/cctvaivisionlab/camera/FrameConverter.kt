package com.sunnygumber.cctvaivisionlab.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import com.sunnygumber.cctvaivisionlab.core.FrameData

object FrameConverter {
    fun fromImageProxy(image: ImageProxy, mirrorHorizontally: Boolean): FrameData {
        val source = image.toBitmap()
        val rotation = image.imageInfo.rotationDegrees
        val oriented = orient(source, rotation, mirrorHorizontally)
        return fromBitmap(
            bitmap = oriented,
            timestampNs = image.imageInfo.timestamp,
        )
    }

    fun fromBitmap(
        bitmap: Bitmap,
        timestampNs: Long = System.nanoTime(),
    ): FrameData {
        val software = if (bitmap.config == Bitmap.Config.ARGB_8888) {
            bitmap
        } else {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }

        val width = software.width
        val height = software.height
        val pixels = IntArray(width * height)
        software.getPixels(pixels, 0, width, 0, 0, width, height)

        val rgb = ByteArray(width * height * 3)
        var destination = 0
        pixels.forEach { pixel ->
            rgb[destination++] = ((pixel shr 16) and 0xff).toByte()
            rgb[destination++] = ((pixel shr 8) and 0xff).toByte()
            rgb[destination++] = (pixel and 0xff).toByte()
        }

        if (software !== bitmap) software.recycle()

        return FrameData(
            width = width,
            height = height,
            rgb = rgb,
            timestampNs = timestampNs,
            rotationDegrees = 0,
        )
    }

    private fun orient(
        bitmap: Bitmap,
        rotationDegrees: Int,
        mirrorHorizontally: Boolean,
    ): Bitmap {
        var current = bitmap

        if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            val rotated = Bitmap.createBitmap(
                current,
                0,
                0,
                current.width,
                current.height,
                matrix,
                true,
            )
            if (rotated !== current) current.recycle()
            current = rotated
        }

        if (mirrorHorizontally) {
            val matrix = Matrix().apply { setScale(-1f, 1f) }
            val mirrored = Bitmap.createBitmap(
                current,
                0,
                0,
                current.width,
                current.height,
                matrix,
                true,
            )
            if (mirrored !== current) current.recycle()
            current = mirrored
        }

        return current
    }
}
