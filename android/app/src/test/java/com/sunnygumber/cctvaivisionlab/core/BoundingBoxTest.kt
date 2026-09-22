package com.sunnygumber.cctvaivisionlab.core

import org.junit.Assert.assertEquals
import org.junit.Test

class BoundingBoxTest {
    @Test
    fun derivesGeometry() {
        val box = BoundingBox(
            left = 10f,
            top = 20f,
            right = 50f,
            bottom = 80f,
        )

        assertEquals(40f, box.width)
        assertEquals(60f, box.height)
        assertEquals(2400f, box.area)
        assertEquals(30f, box.centerX)
        assertEquals(50f, box.centerY)
    }
}
