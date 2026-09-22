package com.sunnygumber.cctvaivisionlab.core

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class DefaultFrameScheduler : FrameScheduler {
    private val detectorBusy = AtomicBoolean(false)
    private val relationBusy = AtomicBoolean(false)
    private val lastRelationNs = AtomicLong(0L)

    override fun shouldRunDetector(timestampNs: Long): Boolean =
        !detectorBusy.get() && !relationBusy.get()

    override fun shouldRunRelation(timestampNs: Long, relationCadenceMs: Long): Boolean {
        if (relationBusy.get()) return false
        val previous = lastRelationNs.get()
        return previous == 0L || timestampNs - previous >= relationCadenceMs * 1_000_000L
    }

    override fun markDetectorStarted(timestampNs: Long) {
        detectorBusy.set(true)
    }

    override fun markDetectorFinished(timestampNs: Long) {
        detectorBusy.set(false)
    }

    override fun markRelationStarted(timestampNs: Long) {
        relationBusy.set(true)
        lastRelationNs.set(timestampNs)
    }

    override fun markRelationFinished(timestampNs: Long) {
        relationBusy.set(false)
    }

    override fun reset() {
        detectorBusy.set(false)
        relationBusy.set(false)
        lastRelationNs.set(0L)
    }
}
