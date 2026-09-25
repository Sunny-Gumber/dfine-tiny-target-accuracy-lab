package com.sunnygumber.cctvaivisionlab.tracking

import com.sunnygumber.cctvaivisionlab.core.BoundingBox
import com.sunnygumber.cctvaivisionlab.core.Detection
import com.sunnygumber.cctvaivisionlab.core.SceneRelation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RelationPolicyTest {
    private fun detection(
        label: String,
        classId: Int,
        trackId: Int,
    ) = Detection(
        box = BoundingBox(0f, 0f, 100f, 100f),
        confidence = 0.9f,
        label = label,
        classId = classId,
        trackId = trackId,
        trackAge = 5,
    )

    private fun relation(
        subject: Detection,
        predicate: String,
        objectDetection: Detection,
        score: Float = 0.7f,
    ) = SceneRelation(
        subject = subject,
        predicate = predicate,
        objectDetection = objectDetection,
        score = score,
        rankScore = score,
    )

    @Test
    fun rejectsPersonWearingChair() {
        val person = detection("Person", 0, 1)
        val chair = detection("Chair", 56, 2)

        assertFalse(RelationPolicy.isPlausible(relation(person, "wearing", chair)))
    }

    @Test
    fun keepsChairBesideDiningTable() {
        val chair = detection("Chair", 56, 2)
        val table = detection("Dining Table", 60, 3)

        assertTrue(RelationPolicy.isPlausible(relation(chair, "beside", table)))
    }

    @Test
    fun keepsOnlyBestRelationPerObjectPair() {
        val person = detection("Person", 0, 1)
        val laptop = detection("Laptop", 63, 2)
        val output = RelationPolicy.filterAndRank(
            listOf(
                relation(person, "looking at", laptop, 0.75f),
                relation(person, "using", laptop, 0.72f),
            ),
        )

        assertEquals(1, output.size)
        assertEquals("using", output.single().predicate)
    }
}
