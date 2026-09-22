package com.sunnygumber.cctvaivisionlab.inference

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.sunnygumber.cctvaivisionlab.core.InferenceProvider
import java.io.File

data class NativeSession(
    val session: OrtSession,
    val provider: InferenceProvider,
)

object OrtSessionFactory {
    val environment: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    fun create(
        modelFile: File,
        preferNnapi: Boolean = true,
    ): NativeSession {
        require(modelFile.isFile && modelFile.length() > 0L) {
            "Model file is missing or empty: ${modelFile.absolutePath}"
        }

        if (preferNnapi) {
            try {
                val options = OrtSession.SessionOptions()
                options.addNnapi()
                val session = environment.createSession(modelFile.absolutePath, options)
                return NativeSession(session, InferenceProvider.NNAPI)
            } catch (_: Throwable) {
                // Some graphs or devices cannot create a complete NNAPI session.
                // A clean CPU session is the reliability fallback for V1.
            }
        }

        val cpuOptions = OrtSession.SessionOptions()
        return NativeSession(
            session = environment.createSession(modelFile.absolutePath, cpuOptions),
            provider = InferenceProvider.CPU,
        )
    }
}
