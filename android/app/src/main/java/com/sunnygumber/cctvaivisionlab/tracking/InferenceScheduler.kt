package com.sunnygumber.cctvaivisionlab.tracking

import com.sunnygumber.cctvaivisionlab.core.FrameScheduler

class InferenceScheduler : FrameScheduler {
    private var detectorBusy = false
    private var relationBusy = false
    private var lastRelationFinishedNs = Long.MIN_VALUE

    @Synchronized
    override fun shouldRunDetector(timestampNs: Long): Boolean = !detectorBusy && !relationBusy

    @Synchronized
    override fun shouldRunRelation(timestampNs: Long, relationCadenceMs: Long): Boolean {
        if (detectorBusy || relationBusy) return false
        if (lastRelationFinishedNs == Long.MIN_VALUE) return true
        val cadenceNs = relationCadenceMs * 1_000_000L
        return timestampNs - lastRelationFinishedNs >= cadenceNs
    }

    @Synchronized
    override fun markDetectorStarted(timestampNs: Long) {
        detectorBusy = true
    }

    @Synchronized
    override fun markDetectorFinished(timestampNs: Long) {
        detectorBusy = false
    }

    @Synchronized
    override fun markRelationStarted(timestampNs: Long) {
        relationBusy = true
    }

    @Synchronized
    override fun markRelationFinished(timestampNs: Long) {
        relationBusy = false
        lastRelationFinishedNs = timestampNs
    }

    @Synchronized
    override fun reset() {
        detectorBusy = false
        relationBusy = false
        lastRelationFinishedNs = Long.MIN_VALUE
    }
}
