package com.sunnygumber.cctvaivisionlab.relations

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.Charset
import java.util.zip.ZipFile

data class PredicateBank(
    val predicates: List<String>,
    val embeddings: FloatArray,
    val alpha: FloatArray,
)

object PredicateBankLoader {
    private const val TEXT_DIM = 512

    private val defaultPredicates = listOf(
        "wearing",
        "riding",
        "playing",
        "sitting on",
        "sitting at",
        "holding",
        "sitting in",
        "looking at",
        "using",
        "watching",
        "standing on",
        "carrying",
        "talking to",
        "smiling at",
        "standing beside",
        "walking past",
        "posing with",
        "leaning against",
        "part of",
        "resting on",
        "on",
        "covering",
        "inside",
        "on top of",
        "contained in",
        "hanging from",
        "surrounding",
        "attached to",
        "in front of",
        "beside",
        "to the left of",
        "to the right of",
        "behind",
        "above",
        "below",
    )

    val cctvPredicates = listOf(
        "wearing",
        "riding",
        "sitting on",
        "holding",
        "looking at",
        "using",
        "standing on",
        "carrying",
        "standing beside",
        "walking past",
        "leaning against",
        "resting on",
        "on",
        "covering",
        "inside",
        "attached to",
        "in front of",
        "beside",
        "behind",
        "above",
        "below",
    )

    fun load(file: File): PredicateBank {
        require(file.isFile) { "Predicate bank is missing" }

        ZipFile(file).use { zip ->
            val wBytes = readEntry(zip, "W.npy")
            val alphaBytes = readEntry(zip, "alpha.npy")
            val w = parseFloat32Npy(wBytes)
            val alpha = parseFloat32Npy(alphaBytes)

            require(w.shape.size == 2 && w.shape[1] == TEXT_DIM) {
                "Unexpected predicate embedding shape: ${w.shape.joinToString(" x ")}"
            }
            require(w.shape[0] >= defaultPredicates.size)
            require(alpha.data.size >= defaultPredicates.size)

            val indices = cctvPredicates.map { predicate ->
                defaultPredicates.indexOf(predicate).also { index ->
                    require(index >= 0) { "Predicate not found: $predicate" }
                }
            }

            val selectedW = FloatArray(indices.size * TEXT_DIM)
            val selectedAlpha = FloatArray(indices.size)

            indices.forEachIndexed { targetIndex, sourceIndex ->
                System.arraycopy(
                    w.data,
                    sourceIndex * TEXT_DIM,
                    selectedW,
                    targetIndex * TEXT_DIM,
                    TEXT_DIM,
                )
                selectedAlpha[targetIndex] = alpha.data[sourceIndex]
            }

            return PredicateBank(
                predicates = cctvPredicates,
                embeddings = selectedW,
                alpha = selectedAlpha,
            )
        }
    }

    private fun readEntry(zip: ZipFile, name: String): ByteArray {
        val entry = zip.getEntry(name) ?: error("Predicate bank entry $name was not found")
        return zip.getInputStream(entry).use { it.readBytes() }
    }

    private data class NpyArray(
        val shape: IntArray,
        val data: FloatArray,
    )

    private fun parseFloat32Npy(bytes: ByteArray): NpyArray {
        require(bytes.size >= 12)
        require((bytes[0].toInt() and 0xff) == 0x93)
        require(bytes.copyOfRange(1, 6).toString(Charsets.US_ASCII) == "NUMPY")

        val major = bytes[6].toInt() and 0xff
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val headerLength: Int
        val headerStart: Int

        if (major == 1) {
            headerLength = buffer.getShort(8).toInt() and 0xffff
            headerStart = 10
        } else {
            headerLength = buffer.getInt(8)
            headerStart = 12
        }

        require(headerStart + headerLength <= bytes.size)
        val header = bytes.copyOfRange(
            headerStart,
            headerStart + headerLength,
        ).toString(Charset.forName("ISO-8859-1"))

        val descr = Regex("'descr':\\s*'([^']+)'").find(header)?.groupValues?.get(1)
        val fortran = Regex("'fortran_order':\\s*(True|False)").find(header)?.groupValues?.get(1)
        val shapeText = Regex("'shape':\\s*\\(([^)]*)\\)").find(header)?.groupValues?.get(1)

        require(descr in setOf("<f4", "|f4", "=f4")) {
            "Expected float32 NPY data, got $descr"
        }
        require(fortran != "True") { "Fortran-order NPY arrays are not supported" }
        require(shapeText != null) { "NPY shape is missing" }

        val shape = shapeText
            .split(",")
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { it.toInt() }
            .toIntArray()
        val count = shape.fold(1) { product, value -> product * value }
        val dataStart = headerStart + headerLength
        val dataEnd = dataStart + count * 4
        require(dataEnd <= bytes.size) { "NPY payload is truncated" }

        val values = FloatArray(count)
        val dataBuffer = ByteBuffer.wrap(bytes, dataStart, count * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (index in values.indices) values[index] = dataBuffer.float

        return NpyArray(shape = shape, data = values)
    }
}
