package com.sunnygumber.cctvaivisionlab.camera

import android.content.ContentResolver
import android.graphics.ImageDecoder
import android.net.Uri
import com.sunnygumber.cctvaivisionlab.core.FrameData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

object ImageFrameLoader {
    suspend fun load(
        resolver: ContentResolver,
        uri: Uri,
    ): FrameData = withContext(Dispatchers.IO) {
        val source = ImageDecoder.createSource(resolver, uri)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.isMutableRequired = false
        }

        try {
            val width = bitmap.width
            val height = bitmap.height
            require(width > 0 && height > 0)

            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            val rgb = ByteArray(width * height * 3)

            pixels.forEachIndexed { index, argb ->
                val offset = index * 3
                rgb[offset] = ((argb shr 16) and 0xff).toByte()
                rgb[offset + 1] = ((argb shr 8) and 0xff).toByte()
                rgb[offset + 2] = (argb and 0xff).toByte()
            }

            FrameData(
                width = width,
                height = height,
                rgb = rgb,
                timestampNs = System.nanoTime(),
            )
        } finally {
            bitmap.recycle()
        }
    }
}
