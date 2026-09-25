package com.sunnygumber.cctvaivisionlab.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import android.os.SystemClock
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.FrameData
import com.sunnygumber.cctvaivisionlab.core.InferenceMetrics
import com.sunnygumber.cctvaivisionlab.core.RelationEngine
import com.sunnygumber.cctvaivisionlab.core.RelationResult
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipFile
import kotlin.math.exp

class RelateAnythingEngine(
    modelFile: File,
    predicateBankFile: File,
    preferNnapi: Boolean = true,
) : RelationEngine {
    private val runtime = OrtRuntimeFactory.createSession(modelFile, preferNnapi)
    private val environment = OrtRuntimeFactory.environment
    private val bank = loadPredicateBank(predicateBankFile)

    override suspend fun analyse(
        frame: FrameData,
        detections: List<Detection>,
        confidenceThreshold: Float,
    ): RelationResult {
        val selected = detections
            .sortedByDescending { it.confidence }
            .take(MAX_RELATION_OBJECTS)

        if (selected.size < 2) {
            return RelationResult(
                relations = emptyList(),
                metrics = InferenceMetrics(provider = runtime.provider),
            )
        }

        val preprocessStart = SystemClock.elapsedRealtimeNanos()
        val imageBuffer = ImagePreprocessor.relationRgb(frame, MODEL_SIZE)
        val boxesBuffer = makeBoxes(frame, selected)
        val boxCountBuffer = ByteBuffer.allocateDirect(Long.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asLongBuffer()
            .apply {
                put(selected.size.toLong())
                rewind()
            }
        val preprocessingMs = elapsedMs(preprocessStart)

        OnnxTensor.createTensor(
            environment,
            imageBuffer,
            longArrayOf(1, 3, MODEL_SIZE.toLong(), MODEL_SIZE.toLong()),
        ).use { image ->
            OnnxTensor.createTensor(
                environment,
                boxesBuffer,
                longArrayOf(1, MAX_BOXES.toLong(), 4),
            ).use { boxes ->
                OnnxTensor.createTensor(environment, boxCountBuffer, longArrayOf(1)).use { boxCounts ->
                    OnnxTensor.createTensor(
                        environment,
                        ImagePreprocessor.floatBuffer(bank.w),
                        longArrayOf(bank.predicates.size.toLong(), TEXT_DIM.toLong()),
                    ).use { w ->
                        OnnxTensor.createTensor(
                            environment,
                            ImagePreprocessor.floatBuffer(bank.alpha),
                            longArrayOf(bank.predicates.size.toLong()),
                        ).use { alpha ->
                            val inferenceStart = SystemClock.elapsedRealtimeNanos()
                            val output = runtime.session.run(
                                mapOf(
                                    "image" to image,
                                    "boxes" to boxes,
                                    "box_counts" to boxCounts,
                                    "W" to w,
                                    "alpha" to alpha,
                                ),
                            )
                            val inferenceMs = elapsedMs(inferenceStart)

                            val postStart = SystemClock.elapsedRealtimeNanos()
                            val relations = output.use { result ->
                                decodeRelations(result, selected, confidenceThreshold)
                            }
                            val postprocessingMs = elapsedMs(postStart)

                            return RelationResult(
                                relations = relations,
                                metrics = InferenceMetrics(
                                    provider = runtime.provider,
                                    preprocessingMs = preprocessingMs,
                                    inferenceMs = inferenceMs,
                                    postprocessingMs = postprocessingMs,
                                ),
                            )
                        }
                    }
                }
            }
        }
    }

    private fun decodeRelations(
        result: ai.onnxruntime.OrtSession.Result,
        selected: List<Detection>,
        threshold: Float,
    ): List<SceneRelation> {
        val predTensor = result.get("pred_logits").orElseThrow {
            IllegalStateException("RelateAnything output pred_logits is missing.")
        } as OnnxTensor
        val pairTensor = result.get("pair_logits").orElseThrow {
            IllegalStateException("RelateAnything output pair_logits is missing.")
        } as OnnxTensor
        val subTensor = result.get("sub_idx").orElseThrow {
            IllegalStateException("RelateAnything output sub_idx is missing.")
        } as OnnxTensor
        val objTensor = result.get("obj_idx").orElseThrow {
            IllegalStateException("RelateAnything output obj_idx is missing.")
        } as OnnxTensor
        val validTensor = result.get("valid_mask").orElseThrow {
            IllegalStateException("RelateAnything output valid_mask is missing.")
        } as OnnxTensor

        val predBuffer = predTensor.floatBuffer ?: error("pred_logits is not float32.")
        val pairBuffer = pairTensor.floatBuffer ?: error("pair_logits is not float32.")
        val subBuffer = subTensor.longBuffer ?: error("sub_idx is not int64.")
        val objBuffer = objTensor.longBuffer ?: error("obj_idx is not int64.")
        val validBuffer = validTensor.byteBuffer ?: error("valid_mask is unavailable.")

        val pred = FloatArray(predBuffer.remaining()).also { predBuffer.get(it) }
        val pair = FloatArray(pairBuffer.remaining()).also { pairBuffer.get(it) }
        val sub = LongArray(subBuffer.remaining()).also { subBuffer.get(it) }
        val obj = LongArray(objBuffer.remaining()).also { objBuffer.get(it) }
        val valid = ByteArray(validBuffer.remaining()).also { validBuffer.get(it) }

        val pairShape = (pairTensor.info as TensorInfo).shape
        val pairCount = pairShape.last().toInt()
        val predicateCount = bank.predicates.size
        val candidates = ArrayList<SceneRelation>()

        for (pairIndex in 0 until pairCount) {
            if (pairIndex >= valid.size || valid[pairIndex].toInt() == 0) continue
            val subjectIndex = sub.getOrNull(pairIndex)?.toInt() ?: continue
            val objectIndex = obj.getOrNull(pairIndex)?.toInt() ?: continue
            if (
                subjectIndex !in selected.indices ||
                objectIndex !in selected.indices ||
                subjectIndex == objectIndex
            ) {
                continue
            }

            var bestPredicate = -1
            var bestScore = Float.NEGATIVE_INFINITY
            val rowOffset = pairIndex * predicateCount

            for (predicateIndex in 0 until predicateCount) {
                val predIndex = rowOffset + predicateIndex
                if (predIndex >= pred.size || pairIndex >= pair.size) break
                val score = sigmoid(
                    CALIBRATION_A *
                        (pred[predIndex] + PAIR_WEIGHT * pair[pairIndex]) +
                        CALIBRATION_B,
                )
                if (score > bestScore) {
                    bestScore = score
                    bestPredicate = predicateIndex
                }
            }

            if (bestPredicate < 0 || bestScore < threshold) continue
            val subject = selected[subjectIndex]
            val objectDetection = selected[objectIndex]
            candidates += SceneRelation(
                subject = subject,
                predicate = bank.predicates[bestPredicate],
                objectDetection = objectDetection,
                score = bestScore,
                rankScore = bestScore * subject.confidence * objectDetection.confidence,
            )
        }

        return candidates
            .sortedByDescending { it.rankScore }
            .take(MAX_RELATIONS)
    }

    override suspend fun close() {
        runtime.close()
    }

    private fun makeBoxes(frame: FrameData, detections: List<Detection>) =
        ImagePreprocessor.floatBuffer(
            FloatArray(MAX_BOXES * 4).also { padded ->
                detections.forEachIndexed { index, detection ->
                    val box = detection.box
                    val x1 = box.left / frame.width.coerceAtLeast(1)
                    val y1 = box.top / frame.height.coerceAtLeast(1)
                    val x2 = box.right / frame.width.coerceAtLeast(1)
                    val y2 = box.bottom / frame.height.coerceAtLeast(1)
                    padded[index * 4] = (x1 + x2) / 2f
                    padded[index * 4 + 1] = (y1 + y2) / 2f
                    padded[index * 4 + 2] = x2 - x1
                    padded[index * 4 + 3] = y2 - y1
                }
            },
        )

    private data class PredicateBank(
        val predicates: List<String>,
        val w: FloatArray,
        val alpha: FloatArray,
    )

    private data class NpyArray(
        val shape: IntArray,
        val data: FloatArray,
    )

    private fun loadPredicateBank(file: File): PredicateBank {
        val wBytes: ByteArray
        val alphaBytes: ByteArray
        ZipFile(file).use { zip ->
            val wEntry = zip.getEntry("W.npy") ?: error("W.npy is missing from predicate bank.")
            val alphaEntry = zip.getEntry("alpha.npy") ?: error("alpha.npy is missing from predicate bank.")
            wBytes = zip.getInputStream(wEntry).use { it.readBytes() }
            alphaBytes = zip.getInputStream(alphaEntry).use { it.readBytes() }
        }

        val fullW = parseFloat32Npy(wBytes)
        val fullAlpha = parseFloat32Npy(alphaBytes)
        require(fullW.shape.size == 2 && fullW.shape[1] == TEXT_DIM) {
            "Unexpected predicate W shape ${fullW.shape.contentToString()}"
        }

        val indices = CCTV_PREDICATES.map { name ->
            DEFAULT_BANK_PREDICATES.indexOf(name).also { index ->
                require(index >= 0) { "Predicate $name is missing from default bank." }
            }
        }

        val selectedW = FloatArray(indices.size * TEXT_DIM)
        val selectedAlpha = FloatArray(indices.size)
        indices.forEachIndexed { targetIndex, sourceIndex ->
            fullW.data.copyInto(
                selectedW,
                destinationOffset = targetIndex * TEXT_DIM,
                startIndex = sourceIndex * TEXT_DIM,
                endIndex = (sourceIndex + 1) * TEXT_DIM,
            )
            selectedAlpha[targetIndex] = fullAlpha.data[sourceIndex]
        }

        return PredicateBank(CCTV_PREDICATES, selectedW, selectedAlpha)
    }

    private fun parseFloat32Npy(bytes: ByteArray): NpyArray {
        require(bytes.size >= 12) { "NPY payload is too small." }
        require((bytes[0].toInt() and 0xff) == 0x93) { "Invalid NPY magic." }
        require(String(bytes, 1, 5, Charsets.US_ASCII) == "NUMPY") { "Invalid NPY signature." }

        val major = bytes[6].toInt() and 0xff
        val headerLength: Int
        val headerStart: Int

        if (major == 1) {
            headerLength = (bytes[8].toInt() and 0xff) or ((bytes[9].toInt() and 0xff) shl 8)
            headerStart = 10
        } else {
            headerLength = ByteBuffer.wrap(bytes, 8, 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .int
            headerStart = 12
        }

        require(headerStart + headerLength <= bytes.size) { "NPY header is truncated." }
        val header = String(bytes, headerStart, headerLength, Charsets.US_ASCII)
        require(header.contains("'fortran_order': False")) { "Fortran-order NPY arrays are unsupported." }
        val descr = Regex("'descr':\\s*'([^']+)'").find(header)?.groupValues?.get(1)
        require(descr == "<f4" || descr == "|f4" || descr == "=f4") {
            "Expected float32 NPY data, got $descr"
        }

        val shapeText = Regex("'shape':\\s*\\(([^)]*)\\)").find(header)?.groupValues?.get(1)
            ?: error("NPY shape is missing.")
        val shape = shapeText.split(",")
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { it.toInt() }
            .toIntArray()
        val count = shape.fold(1) { product, value -> product * value }
        val dataStart = headerStart + headerLength
        require(dataStart + count * Float.SIZE_BYTES <= bytes.size) { "NPY data is truncated." }

        val floatBuffer = ByteBuffer.wrap(bytes, dataStart, count * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
        val data = FloatArray(count)
        floatBuffer.get(data)
        return NpyArray(shape, data)
    }

    private fun sigmoid(value: Float): Float =
        if (value >= 0f) {
            (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
        } else {
            val exponential = exp(value.toDouble())
            (exponential / (1.0 + exponential)).toFloat()
        }

    private fun elapsedMs(startNs: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0

    companion object {
        private const val MODEL_SIZE = 448
        private const val MAX_BOXES = 32
        private const val MAX_RELATION_OBJECTS = 5
        private const val TEXT_DIM = 512
        private const val CALIBRATION_A = 0.5651f
        private const val CALIBRATION_B = -1.9623f
        private const val PAIR_WEIGHT = 1f
        private const val MAX_RELATIONS = 12

        private val DEFAULT_BANK_PREDICATES = listOf(
            "wearing", "riding", "playing", "sitting on", "sitting at", "holding",
            "sitting in", "looking at", "using", "watching", "standing on", "carrying",
            "talking to", "smiling at", "standing beside", "walking past", "posing with",
            "leaning against", "part of", "resting on", "on", "covering", "inside",
            "on top of", "contained in", "hanging from", "surrounding", "attached to",
            "in front of", "beside", "to the left of", "to the right of", "behind",
            "above", "below",
        )

        private val CCTV_PREDICATES = listOf(
            "wearing", "riding", "sitting on", "holding", "looking at", "using",
            "standing on", "carrying", "standing beside", "walking past",
            "leaning against", "resting on", "on", "covering", "inside",
            "attached to", "in front of", "beside", "behind", "above", "below",
        )
    }
}
