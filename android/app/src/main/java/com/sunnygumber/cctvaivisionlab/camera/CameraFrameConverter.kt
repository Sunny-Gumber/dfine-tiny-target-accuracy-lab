package com.sunnygumber.cctvaivisionlab.camera

import androidx.camera.core.ImageProxy
import com.sunnygumber.cctvaivisionlab.core.FrameData

object CameraFrameConverter {
    fun fromRgba(image: ImageProxy): FrameData {
        val plane = image.planes.firstOrNull()
            ?: error("Camera frame has no RGBA plane")
        val width = image.width
        val height = image.height
        val rotation = ((image.imageInfo.rotationDegrees % 360) + 360) % 360
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val buffer = plane.buffer.duplicate()

        require(pixelStride >= 4) {
            "Expected CameraX RGBA pixel stride >= 4, got $pixelStride"
        }

        val sourceRgb = ByteArray(width * height * 3)
        for (y in 0 until height) {
            val rowBase = y * rowStride
            for (x in 0 until width) {
                val offset = rowBase + x * pixelStride
                val dest = (y * width + x) * 3

                // CameraX OUTPUT_IMAGE_FORMAT_RGBA_8888 exposes A,R,G,B bytes
                // in the first plane.
                sourceRgb[dest] = buffer.get(offset + 1)
                sourceRgb[dest + 1] = buffer.get(offset + 2)
                sourceRgb[dest + 2] = buffer.get(offset + 3)
            }
        }

        if (rotation == 0) {
            return FrameData(
                width = width,
                height = height,
                rgb = sourceRgb,
                timestampNs = image.imageInfo.timestamp,
                rotationDegrees = 0,
            )
        }

        val rotatedWidth = if (rotation == 90 || rotation == 270) height else width
        val rotatedHeight = if (rotation == 90 || rotation == 270) width else height
        val rotated = ByteArray(rotatedWidth * rotatedHeight * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val (dx, dy) = when (rotation) {
                    90 -> (height - 1 - y) to x
                    180 -> (width - 1 - x) to (height - 1 - y)
                    270 -> y to (width - 1 - x)
                    else -> x to y
                }
                val src = (y * width + x) * 3
                val dst = (dy * rotatedWidth + dx) * 3
                rotated[dst] = sourceRgb[src]
                rotated[dst + 1] = sourceRgb[src + 1]
                rotated[dst + 2] = sourceRgb[src + 2]
            }
        }

        return FrameData(
            width = rotatedWidth,
            height = rotatedHeight,
            rgb = rotated,
            timestampNs = image.imageInfo.timestamp,
            rotationDegrees = 0,
        )
    }
}
