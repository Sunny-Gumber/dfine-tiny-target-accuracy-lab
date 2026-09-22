package com.sunnygumber.cctvaivisionlab.tracking

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class InferenceSchedulerTest {
    @Test
    fun relationSelfThrottlesFromCompletionTime() {
        val scheduler = InferenceScheduler()
        val start = 1_000_000_000L

        assertTrue(scheduler.shouldRunRelation(start, 1500))
        scheduler.markRelationStarted(start)
        assertFalse(scheduler.shouldRunDetector(start + 1))
        scheduler.markRelationFinished(start + 2_000_000_000L)

        assertFalse(scheduler.shouldRunRelation(start + 3_000_000_000L, 1500))
        assertTrue(scheduler.shouldRunRelation(start + 3_600_000_000L, 1500))
    }

    @Test
    fun detectorAndRelationDoNotOverlap() {
        val scheduler = InferenceScheduler()
        val now = 5_000_000_000L

        scheduler.markDetectorStarted(now)
        assertFalse(scheduler.shouldRunRelation(now, 1000))
        scheduler.markDetectorFinished(now + 10)

        scheduler.markRelationStarted(now + 20)
        assertFalse(scheduler.shouldRunDetector(now + 30))
    }
}
