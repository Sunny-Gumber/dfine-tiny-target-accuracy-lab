package com.sunnygumber.cctvaivisionlab.models

import android.content.Context
import com.sunnygumber.cctvaivisionlab.core.ModelDescriptor
import com.sunnygumber.cctvaivisionlab.core.ModelManager
import com.sunnygumber.cctvaivisionlab.core.ModelState
import com.sunnygumber.cctvaivisionlab.core.ModelStatus
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class AndroidModelManager(
    context: Context,
) : ModelManager {
    private val appContext = context.applicationContext
    private val modelDir = File(appContext.filesDir, "models").apply { mkdirs() }
    private val prefs = appContext.getSharedPreferences("model_manifest", Context.MODE_PRIVATE)
    private val mutex = Mutex()

    override suspend fun status(descriptor: ModelDescriptor): ModelStatus = withContext(Dispatchers.IO) {
        val file = File(modelDir, descriptor.fileName)
        val recordedVersion = prefs.getString("${descriptor.id}.version", null)
        val recordedSha = prefs.getString("${descriptor.id}.sha256", null)
        val ready = file.isFile &&
            file.length() > 0L &&
            recordedVersion == descriptor.version &&
            (descriptor.sha256 == null || recordedSha.equals(descriptor.sha256, ignoreCase = true))

        ModelStatus(
            descriptor = descriptor,
            state = if (ready) ModelState.READY else ModelState.MISSING,
            localFile = file.takeIf { ready },
            downloadedBytes = if (file.exists()) file.length() else 0L,
            totalBytes = if (ready) file.length() else null,
        )
    }

    override suspend fun ensureAvailable(
        descriptor: ModelDescriptor,
        onProgress: (ModelStatus) -> Unit,
    ): ModelStatus = mutex.withLock {
        status(descriptor).takeIf { it.state == ModelState.READY }?.let { return@withLock it }

        withContext(Dispatchers.IO) {
            modelDir.mkdirs()
            val finalFile = File(modelDir, descriptor.fileName)
            val tempFile = File(modelDir, "${descriptor.fileName}.part")
            if (tempFile.exists()) tempFile.delete()

            try {
                onProgress(
                    ModelStatus(
                        descriptor = descriptor,
                        state = ModelState.DOWNLOADING,
                        downloadedBytes = 0,
                    ),
                )

                val connection = (URL(descriptor.url).openConnection() as HttpURLConnection).apply {
                    instanceFollowRedirects = true
                    connectTimeout = 20_000
                    readTimeout = 60_000
                    requestMethod = "GET"
                    setRequestProperty("User-Agent", "CctvAiVisionLab/0.1 Android")
                }

                connection.connect()
                if (connection.responseCode !in 200..299) {
                    throw IllegalStateException(
                        "Model download failed for ${descriptor.id}: HTTP ${connection.responseCode}",
                    )
                }

                val total = connection.contentLengthLong.takeIf { it > 0 }
                connection.inputStream.use { input ->
                    FileOutputStream(tempFile).use { output ->
                        val buffer = ByteArray(DEFAULT_BUFFER_SIZE * 8)
                        var downloaded = 0L
                        while (true) {
                            val read = input.read(buffer)
                            if (read < 0) break
                            output.write(buffer, 0, read)
                            downloaded += read

                            onProgress(
                                ModelStatus(
                                    descriptor = descriptor,
                                    state = ModelState.DOWNLOADING,
                                    downloadedBytes = downloaded,
                                    totalBytes = total,
                                ),
                            )
                        }
                        output.fd.sync()
                    }
                }
                connection.disconnect()

                if (!tempFile.isFile || tempFile.length() == 0L) {
                    throw IllegalStateException("Downloaded model is empty: ${descriptor.id}")
                }

                descriptor.sha256?.let { expected ->
                    val actual = sha256(tempFile)
                    if (!actual.equals(expected, ignoreCase = true)) {
                        throw IllegalStateException(
                            "Checksum mismatch for ${descriptor.id}. Expected $expected, got $actual",
                        )
                    }
                }

                if (finalFile.exists() && !finalFile.delete()) {
                    throw IllegalStateException("Could not replace old model file ${finalFile.name}")
                }
                if (!tempFile.renameTo(finalFile)) {
                    tempFile.copyTo(finalFile, overwrite = true)
                    tempFile.delete()
                }

                prefs.edit()
                    .putString("${descriptor.id}.version", descriptor.version)
                    .putString("${descriptor.id}.sha256", descriptor.sha256)
                    .putLong("${descriptor.id}.bytes", finalFile.length())
                    .apply()

                ModelStatus(
                    descriptor = descriptor,
                    state = ModelState.READY,
                    localFile = finalFile,
                    downloadedBytes = finalFile.length(),
                    totalBytes = finalFile.length(),
                ).also(onProgress)
            } catch (error: Throwable) {
                val partialBytes = tempFile.length()
                tempFile.delete()
                ModelStatus(
                    descriptor = descriptor,
                    state = ModelState.FAILED,
                    localFile = finalFile.takeIf { it.exists() },
                    downloadedBytes = partialBytes,
                    error = error.message ?: error::class.java.simpleName,
                ).also(onProgress)
            }
        }
    }

    override suspend fun clear(descriptor: ModelDescriptor) = withContext(Dispatchers.IO) {
        File(modelDir, descriptor.fileName).delete()
        File(modelDir, "${descriptor.fileName}.part").delete()
        prefs.edit()
            .remove("${descriptor.id}.version")
            .remove("${descriptor.id}.sha256")
            .remove("${descriptor.id}.bytes")
            .apply()
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE * 8)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
