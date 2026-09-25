package com.sunnygumber.cctvaivisionlab.tracking

import com.sunnygumber.cctvaivisionlab.core.SceneRelation

object RelationPolicy {
    private val wearableObjects = setOf("Backpack", "Tie", "Handbag")
    private val rideableObjects = setOf("Bicycle", "Motorcycle", "Horse")
    private val sittingObjects = setOf("Chair", "Couch", "Bench", "Bed")
    private val usableObjects = setOf(
        "Laptop", "Cell Phone", "Keyboard", "Mouse", "Remote", "TV",
    )
    private val carryableObjects = setOf(
        "Backpack", "Handbag", "Suitcase", "Umbrella",
    )
    private val bulkyObjects = setOf(
        "Person", "Car", "Bus", "Truck", "Train", "Airplane", "Boat",
        "Chair", "Couch", "Bed", "Dining Table", "Toilet",
        "Refrigerator", "Oven", "Microwave", "Sink",
    )
    private val personLedPredicates = setOf(
        "wearing", "riding", "sitting on", "holding", "looking at", "using",
        "standing on", "carrying", "walking past", "leaning against",
    )
    private val actionPredicates = setOf(
        "wearing", "riding", "sitting on", "holding", "using", "carrying",
    )
    private val spatialPredicates = setOf(
        "in front of", "beside", "behind", "above", "below", "inside", "attached to",
    )

    fun filterAndRank(
        relations: List<SceneRelation>,
        maxRelations: Int = 3,
    ): List<SceneRelation> {
        val valid = relations
            .filter(::isPlausible)
            .map { relation ->
                relation.copy(rankScore = relation.rankScore * predicateWeight(relation.predicate))
            }

        return valid
            .groupBy(::pairKey)
            .values
            .mapNotNull { pair -> pair.maxByOrNull { it.rankScore } }
            .sortedByDescending { it.rankScore }
            .take(maxRelations)
    }

    fun isPlausible(relation: SceneRelation): Boolean {
        val subject = relation.subject.label
        val objectLabel = relation.objectDetection.label
        val predicate = relation.predicate

        if (predicate in personLedPredicates && subject != "Person") return false

        return when (predicate) {
            "wearing" -> objectLabel in wearableObjects
            "riding" -> objectLabel in rideableObjects
            "sitting on" -> objectLabel in sittingObjects
            "holding" -> objectLabel !in bulkyObjects
            "using" -> objectLabel in usableObjects
            "carrying" -> objectLabel in carryableObjects
            "standing on" -> objectLabel != "Person" && objectLabel !in setOf("Car", "Bus", "Truck")
            else -> true
        }
    }

    private fun predicateWeight(predicate: String): Float = when {
        predicate in actionPredicates -> 1.15f
        predicate == "looking at" || predicate == "walking past" || predicate == "leaning against" -> 1.05f
        predicate in spatialPredicates -> 0.90f
        else -> 1.0f
    }

    private fun pairKey(relation: SceneRelation): String {
        val subject = relation.subject.trackId?.toString()
            ?: "${relation.subject.classId}:${relation.subject.box.centerX.toInt()}:${relation.subject.box.centerY.toInt()}"
        val objectId = relation.objectDetection.trackId?.toString()
            ?: "${relation.objectDetection.classId}:${relation.objectDetection.box.centerX.toInt()}:${relation.objectDetection.box.centerY.toInt()}"
        return "$subject>$objectId"
    }
}
