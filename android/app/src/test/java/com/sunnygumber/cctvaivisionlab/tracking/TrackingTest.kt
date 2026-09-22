package com.sunnygumber.cctvaivisionlab.tracking

import com.sunnygumber.cctvaivisionlab.core.BoundingBox
import com.sunnygumber.cctvaivisionlab.core.Detection
import org.junit.Assert.assertEquals
import org.junit.Test

class TrackingTest {
    private fun detection(
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
        confidence: Float,
    ) = Detection(
        box = BoundingBox(left, top, right, bottom),
        confidence = confidence,
        label = "Car",
        classId = 2,
    )

    @Test
    fun removesSameLabelContainedDuplicate() {
        val detections = listOf(
            detection(0f, 0f, 100f, 100f, 0.94f),
            detection(10f, 10f, 90f, 90f, 0.64f),
        )

        val result = deduplicateDetections(detections)

        assertEquals(1, result.size)
        assertEquals(0.94f, result.first().confidence)
    }

    @Test
    fun keepsTrackIdAcrossOverlappingFrames() {
        val tracker = IoUTracker()
        val first = tracker.update(listOf(detection(0f, 0f, 100f, 100f, 0.9f))).single()
        val second = tracker.update(listOf(detection(5f, 5f, 105f, 105f, 0.88f))).single()

        assertEquals(first.trackId, second.trackId)
        assertEquals(2, second.trackAge)
    }
}
