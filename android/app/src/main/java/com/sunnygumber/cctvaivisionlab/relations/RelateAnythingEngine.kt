package com.sunnygumber.cctvaivisionlab.relations

import ai.onnxruntime.OnnxTensor
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.FrameData
import com.sunnygumber.cctvaivisionlab.core.InferenceMetrics
import com.sunnygumber.cctvaivisionlab.core.RelationEngine
import com.sunnygumber.cctvaivisionlab.core.RelationResult
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import com.sunnygumber.cctvaivisionlab.inference.ImagePreprocess
import com.sunnygumber.cctvaivisionlab.inference.OrtSessionFactory
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.exp

class RelateAnythingEngine(
    modelFile: File,
    predicateBankFile: File,
    preferNnapi: Boolean = true,
) : RelationEngine {
    private val nativeSession = OrtSessionFactory.create(modelFile, preferNnapi)
    private val bank = PredicateBankLoader.load(predicateBankFile)

    private val modelSize = 448
    private val maxBoxes = 32
    private val maxRelationObjects = 8
    private val textDim = 512
    private val calibrationA = 0.5651f
    private val calibrationB = -1.9623f
    private val pairWeight = 1f
    private val maxRelations = 12

    override suspend fun analyse(
        frame: FrameData,
        detections: List<Detection>,
        confidenceThreshold: Float,
    ): RelationResult {
        val selected = detections
            .sortedByDescending { it.confidence }
            .take(maxRelationObjects)

        if (selected.size < 2) {
            return RelationResult(
                relations = emptyList(),
                metrics = InferenceMetrics(provider = nativeSession.provider),
            )
        }

        val preprocessStarted = System.nanoTime()
        val imageData = ImagePreprocess.relationRgbNormalized(frame, modelSize)
        val boxData = makeBoxes(frame, selected)
        val preprocessingMs = elapsedMs(preprocessStarted)

        val environment = OrtSessionFactory.environment
        val imageTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(imageData),
            longArrayOf(1, 3, modelSize.toLong(), modelSize.toLong()),
        )
        val boxesTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(boxData),
            longArrayOf(1, maxBoxes.toLong(), 4),
        )
        val countsTensor = OnnxTensor.createTensor(
            environment,
            LongBuffer.wrap(longArrayOf(selected.size.toLong())),
            longArrayOf(1),
        )
        val wTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(bank.embeddings),
            longArrayOf(bank.predicates.size.toLong(), textDim.toLong()),
        )
        val alphaTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(bank.alpha),
            longArrayOf(bank.predicates.size.toLong()),
        )

        val inferenceStarted = System.nanoTime()
        val candidateRelations: List<SceneRelation>
        try {
            nativeSession.session.run(
                mapOf(
                    "image" to imageTensor,
                    "boxes" to boxesTensor,
                    "box_counts" to countsTensor,
                    "W" to wTensor,
                    "alpha" to alphaTensor,
                ),
            ).use { result ->
                val inferenceMs = elapsedMs(inferenceStarted)
                val postStarted = System.nanoTime()

                val predTensor = result.get("pred_logits").orElseThrow {
                    IllegalStateException("RelateAnything output pred_logits is missing")
                } as OnnxTensor
                val pairTensor = result.get("pair_logits").orElseThrow {
                    IllegalStateException("RelateAnything output pair_logits is missing")
                } as OnnxTensor
                val subTensor = result.get("sub_idx").orElseThrow {
                    IllegalStateException("RelateAnything output sub_idx is missing")
                } as OnnxTensor
                val objTensor = result.get("obj_idx").orElseThrow {
                    IllegalStateException("RelateAnything output obj_idx is missing")
                } as OnnxTensor
                val validTensor = result.get("valid_mask").orElseThrow {
                    IllegalStateException("RelateAnything output valid_mask is missing")
                } as OnnxTensor

                val pred = predTensor.floatBuffer.toFloatArray()
                val pair = pairTensor.floatBuffer.toFloatArray()
                val sub = subTensor.longBuffer.toLongArray()
                val obj = objTensor.longBuffer.toLongArray()
                val valid = validTensor.byteBuffer.toByteArray()

                val pairCount = pair.size
                val predicateCount = bank.predicates.size
                val candidates = ArrayList<SceneRelation>()

                for (pairIndex in 0 until pairCount) {
                    if (pairIndex >= valid.size || valid[pairIndex].toInt() == 0) continue
                    if (pairIndex >= sub.size || pairIndex >= obj.size) continue

                    val subjectIndex = sub[pairIndex].toInt()
                    val objectIndex = obj[pairIndex].toInt()
                    if (
                        subjectIndex !in selected.indices ||
                        objectIndex !in selected.indices ||
                        subjectIndex == objectIndex
                    ) {
                        continue
                    }

                    val rowOffset = pairIndex * predicateCount
                    if (rowOffset + predicateCount > pred.size) break

                    var bestPredicate = -1
                    var bestScore = Float.NEGATIVE_INFINITY

                    for (predicateIndex in 0 until predicateCount) {
                        val score = sigmoid(
                            calibrationA * (
                                pred[rowOffset + predicateIndex] +
                                    pairWeight * pair[pairIndex]
                                ) + calibrationB,
                        )
                        if (score > bestScore) {
                            bestScore = score
                            bestPredicate = predicateIndex
                        }
                    }

                    if (bestPredicate < 0 || bestScore < confidenceThreshold) continue

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

                candidateRelations = candidates
                    .sortedByDescending { it.rankScore }
                    .take(maxRelations)

                return RelationResult(
                    relations = candidateRelations,
                    metrics = InferenceMetrics(
                        provider = nativeSession.provider,
                        preprocessingMs = preprocessingMs,
                        inferenceMs = inferenceMs,
                        postprocessingMs = elapsedMs(postStarted),
                    ),
                )
            }
        } finally {
            imageTensor.close()
            boxesTensor.close()
            countsTensor.close()
            wTensor.close()
            alphaTensor.close()
        }
    }

    private fun makeBoxes(
        frame: FrameData,
        detections: List<Detection>,
    ): FloatArray {
        val padded = FloatArray(maxBoxes * 4)
        val width = frame.width.coerceAtLeast(1).toFloat()
        val height = frame.height.coerceAtLeast(1).toFloat()

        detections.forEachIndexed { index, detection ->
            val x1 = detection.box.left / width
            val y1 = detection.box.top / height
            val x2 = detection.box.right / width
            val y2 = detection.box.bottom / height
            padded[index * 4] = (x1 + x2) / 2f
            padded[index * 4 + 1] = (y1 + y2) / 2f
            padded[index * 4 + 2] = x2 - x1
            padded[index * 4 + 3] = y2 - y1
        }

        return padded
    }

    private fun sigmoid(value: Float): Float =
        if (value >= 0f) {
            1f / (1f + exp(-value))
        } else {
            val e = exp(value)
            e / (1f + e)
        }

    private fun elapsedMs(startNs: Long): Double =
        (System.nanoTime() - startNs) / 1_000_000.0

    override suspend fun close() {
        nativeSession.session.close()
    }

    private fun FloatBuffer?.toFloatArray(): FloatArray {
        val buffer = this ?: error("Expected float tensor output")
        val copy = FloatArray(buffer.remaining())
        buffer.get(copy)
        return copy
    }

    private fun LongBuffer?.toLongArray(): LongArray {
        val buffer = this ?: error("Expected int64 tensor output")
        val copy = LongArray(buffer.remaining())
        buffer.get(copy)
        return copy
    }

    private fun java.nio.ByteBuffer?.toByteArray(): ByteArray {
        val buffer = this ?: error("Expected byte/bool tensor output")
        val copy = ByteArray(buffer.remaining())
        buffer.get(copy)
        return copy
    }
}
