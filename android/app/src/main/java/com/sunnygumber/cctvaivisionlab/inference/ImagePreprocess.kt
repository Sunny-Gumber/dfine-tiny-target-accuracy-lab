package com.sunnygumber.cctvaivisionlab.inference

import com.sunnygumber.cctvaivisionlab.core.FrameData
import kotlin.math.floor
import kotlin.math.min

data class LetterboxResult(
    val tensor: FloatArray,
    val scale: Float,
)

object ImagePreprocess {
    fun yoloxLetterboxBgr(
        frame: FrameData,
        size: Int = 640,
    ): LetterboxResult {
        require(frame.width > 0 && frame.height > 0)
        require(frame.rgb.size >= frame.width * frame.height * 3)

        val scale = min(size.toFloat() / frame.width, size.toFloat() / frame.height)
        val resizedWidth = maxOf(1, floor(frame.width * scale).toInt())
        val resizedHeight = maxOf(1, floor(frame.height * scale).toInt())
        val plane = size * size
        val tensor = FloatArray(plane * 3) { 114f }

        for (dstY in 0 until resizedHeight) {
            val srcY = minOf(frame.height - 1, floor(dstY / scale).toInt())
            for (dstX in 0 until resizedWidth) {
                val srcX = minOf(frame.width - 1, floor(dstX / scale).toInt())
                val src = (srcY * frame.width + srcX) * 3
                val dst = dstY * size + dstX
                val red = frame.rgb[src].toInt() and 0xff
                val green = frame.rgb[src + 1].toInt() and 0xff
                val blue = frame.rgb[src + 2].toInt() and 0xff

                tensor[dst] = blue.toFloat()
                tensor[plane + dst] = green.toFloat()
                tensor[plane * 2 + dst] = red.toFloat()
            }
        }

        return LetterboxResult(tensor = tensor, scale = scale)
    }

    fun relationRgbNormalized(
        frame: FrameData,
        size: Int = 448,
    ): FloatArray {
        require(frame.width > 0 && frame.height > 0)
        require(frame.rgb.size >= frame.width * frame.height * 3)

        val plane = size * size
        val tensor = FloatArray(plane * 3)

        for (dstY in 0 until size) {
            val srcY = minOf(frame.height - 1, floor(dstY * frame.height.toFloat() / size).toInt())
            for (dstX in 0 until size) {
                val srcX = minOf(frame.width - 1, floor(dstX * frame.width.toFloat() / size).toInt())
                val src = (srcY * frame.width + srcX) * 3
                val dst = dstY * size + dstX

                tensor[dst] = (frame.rgb[src].toInt() and 0xff) / 255f
                tensor[plane + dst] = (frame.rgb[src + 1].toInt() and 0xff) / 255f
                tensor[plane * 2 + dst] = (frame.rgb[src + 2].toInt() and 0xff) / 255f
            }
        }

        return tensor
    }
}
