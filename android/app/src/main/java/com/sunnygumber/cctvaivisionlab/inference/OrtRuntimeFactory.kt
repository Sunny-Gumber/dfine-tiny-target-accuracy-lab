package com.sunnygumber.cctvaivisionlab.inference

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.sunnygumber.cctvaivisionlab.core.InferenceProvider
import java.io.File

class OrtSessionBundle(
    val session: OrtSession,
    val options: OrtSession.SessionOptions,
    val provider: InferenceProvider,
) : AutoCloseable {
    override fun close() {
        session.close()
        options.close()
    }
}

object OrtRuntimeFactory {
    val environment: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    fun createSession(model: File, preferNnapi: Boolean = true): OrtSessionBundle {
        if (preferNnapi) {
            val options = baseOptions()
            try {
                options.addNnapi()
                val session = environment.createSession(model.absolutePath, options)
                return OrtSessionBundle(session, options, InferenceProvider.NNAPI)
            } catch (_: Throwable) {
                options.close()
            }
        }

        val options = baseOptions()
        val session = environment.createSession(model.absolutePath, options)
        return OrtSessionBundle(session, options, InferenceProvider.CPU)
    }

    private fun baseOptions() = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
    }
}
