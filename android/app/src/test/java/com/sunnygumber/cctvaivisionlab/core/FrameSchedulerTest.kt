package com.sunnygumber.cctvaivisionlab.core

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameSchedulerTest {
    @Test
    fun detectorCadenceSkipsIntermediateAiRefreshes() {
        val scheduler = DefaultFrameScheduler()
        val start = 1_000_000_000L

        assertTrue(scheduler.shouldRunDetector(start, 1_500))
        scheduler.markDetectorStarted(start)
        scheduler.markDetectorFinished(start + 300_000_000L)

        assertFalse(scheduler.shouldRunDetector(start + 1_000_000_000L, 1_500))
        assertTrue(scheduler.shouldRunDetector(start + 1_600_000_000L, 1_500))
    }

    @Test
    fun relationCadenceSelfThrottles() {
        val scheduler = DefaultFrameScheduler()
        val start = 1_000_000_000L

        assertTrue(scheduler.shouldRunRelation(start, 1_500))
        scheduler.markRelationStarted(start)
        assertFalse(scheduler.shouldRunDetector(start + 100))
        scheduler.markRelationFinished(start + 2_000_000_000L)

        assertTrue(scheduler.shouldRunRelation(start + 2_000_000_000L, 1_500))
    }
}
