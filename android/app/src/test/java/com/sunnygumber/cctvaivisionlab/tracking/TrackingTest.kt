package com.sunnygumber.cctvaivisionlab.tracking

import com.sunnygumber.cctvaivisionlab.core.BoundingBox
import com.sunnygumber.cctvaivisionlab.core.Detection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TrackingTest {
    private fun detection(
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
        confidence: Float = 0.9f,
        label: String = "Car",
    ) = Detection(
        box = BoundingBox(left, top, right, bottom),
        confidence = confidence,
        label = label,
        classId = 2,
    )

    @Test
    fun duplicateSuppressionKeepsStrongerSameLabelBox() {
        val strong = detection(0f, 0f, 100f, 100f, 0.95f)
        val duplicate = detection(5f, 5f, 95f, 95f, 0.65f)
        val result = deduplicateDetections(listOf(duplicate, strong))

        assertEquals(1, result.size)
        assertEquals(0.95f, result.single().confidence)
    }

    @Test
    fun duplicateSuppressionDoesNotMergeDifferentLabels() {
        val car = detection(0f, 0f, 100f, 100f, label = "Car")
        val person = detection(0f, 0f, 100f, 100f, label = "Person")
        assertEquals(2, deduplicateDetections(listOf(car, person)).size)
    }

    @Test
    fun trackerKeepsIdAcrossOverlappingFrames() {
        val tracker = IoUTracker()
        val first = tracker.update(listOf(detection(0f, 0f, 100f, 100f))).single()
        val second = tracker.update(listOf(detection(5f, 5f, 105f, 105f))).single()

        assertEquals(first.trackId, second.trackId)
        assertEquals(2, second.trackAge)
        assertTrue(second.trackId != null)
    }
}
