package com.sunnygumber.cctvaivisionlab.inference

import com.sunnygumber.cctvaivisionlab.core.FrameData
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

data class YoloXInput(
    val data: FloatBuffer,
    val scale: Float,
)

object ImagePreprocessor {
    fun yoloX(frame: FrameData, size: Int = 640): YoloXInput {
        val scale = min(size.toFloat() / frame.width, size.toFloat() / frame.height)
        val scaledWidth = max(1, floor(frame.width * scale).toInt())
        val scaledHeight = max(1, floor(frame.height * scale).toInt())
        val plane = size * size
        val values = FloatArray(plane * 3) { 114f }

        for (y in 0 until scaledHeight) {
            val sourceY = min(frame.height - 1, floor(y / scale).toInt())
            for (x in 0 until scaledWidth) {
                val sourceX = min(frame.width - 1, floor(x / scale).toInt())
                val source = (sourceY * frame.width + sourceX) * 3
                val destination = y * size + x

                val r = frame.rgb[source].toInt() and 0xff
                val g = frame.rgb[source + 1].toInt() and 0xff
                val b = frame.rgb[source + 2].toInt() and 0xff

                values[destination] = b.toFloat()
                values[plane + destination] = g.toFloat()
                values[plane * 2 + destination] = r.toFloat()
            }
        }

        return YoloXInput(floatBuffer(values), scale)
    }

    fun relationRgb(frame: FrameData, size: Int = 448): FloatBuffer {
        val plane = size * size
        val values = FloatArray(plane * 3)

        for (y in 0 until size) {
            val sourceY = min(frame.height - 1, (y.toLong() * frame.height / size).toInt())
            for (x in 0 until size) {
                val sourceX = min(frame.width - 1, (x.toLong() * frame.width / size).toInt())
                val source = (sourceY * frame.width + sourceX) * 3
                val destination = y * size + x

                values[destination] = (frame.rgb[source].toInt() and 0xff) / 255f
                values[plane + destination] = (frame.rgb[source + 1].toInt() and 0xff) / 255f
                values[plane * 2 + destination] = (frame.rgb[source + 2].toInt() and 0xff) / 255f
            }
        }

        return floatBuffer(values)
    }

    fun floatBuffer(values: FloatArray): FloatBuffer {
        val buffer = ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(values)
        buffer.rewind()
        return buffer
    }
}
