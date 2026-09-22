package com.sunnygumber.cctvaivisionlab.models

import android.content.Context
import com.sunnygumber.cctvaivisionlab.core.ModelDescriptor
import com.sunnygumber.cctvaivisionlab.core.ModelManager
import com.sunnygumber.cctvaivisionlab.core.ModelState
import com.sunnygumber.cctvaivisionlab.core.ModelStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

class LocalModelManager(context: Context) : ModelManager {
    private val appContext = context.applicationContext
    private val directory = File(appContext.filesDir, "models").apply { mkdirs() }
    private val prefs = appContext.getSharedPreferences("model_manifest", Context.MODE_PRIVATE)

    override suspend fun status(descriptor: ModelDescriptor): ModelStatus = withContext(Dispatchers.IO) {
        val file = File(directory, descriptor.fileName)
        val storedVersion = prefs.getString("${descriptor.id}.version", null)
        val storedUrl = prefs.getString("${descriptor.id}.url", null)
        val ready = file.isFile &&
            file.length() > 0 &&
            storedVersion == descriptor.version &&
            storedUrl == descriptor.url &&
            verifyIfRequired(file, descriptor.sha256)

        if (ready) {
            ModelStatus(
                descriptor = descriptor,
                state = ModelState.READY,
                localFile = file,
                downloadedBytes = file.length(),
                totalBytes = file.length(),
            )
        } else {
            ModelStatus(descriptor = descriptor, state = ModelState.MISSING)
        }
    }

    override suspend fun ensureAvailable(
        descriptor: ModelDescriptor,
        onProgress: (ModelStatus) -> Unit,
    ): ModelStatus = withContext(Dispatchers.IO) {
        val current = status(descriptor)
        if (current.state == ModelState.READY) return@withContext current

        val target = File(directory, descriptor.fileName)
        val temp = File(directory, "${descriptor.fileName}.part")
        temp.delete()

        try {
            onProgress(ModelStatus(descriptor, ModelState.DOWNLOADING))
            val connection = (URL(descriptor.url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 60_000
                instanceFollowRedirects = true
                requestMethod = "GET"
                setRequestProperty("User-Agent", "CctvAiVisionLab-Android/0.2")
            }

            connection.connect()
            if (connection.responseCode !in 200..299) {
                throw IllegalStateException("HTTP ${connection.responseCode} while downloading ${descriptor.id}")
            }

            val total = connection.contentLengthLong.takeIf { it > 0 }
            var downloaded = 0L
            connection.inputStream.use { input ->
                temp.outputStream().buffered(1024 * 1024).use { output ->
                    val buffer = ByteArray(1024 * 1024)
                    var lastReported = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        downloaded += read
                        if (downloaded - lastReported >= 2L * 1024 * 1024 || downloaded == total) {
                            lastReported = downloaded
                            onProgress(
                                ModelStatus(
                                    descriptor = descriptor,
                                    state = ModelState.DOWNLOADING,
                                    downloadedBytes = downloaded,
                                    totalBytes = total,
                                ),
                            )
                        }
                    }
                }
            }
            connection.disconnect()

            require(temp.length() > 0L) { "Downloaded model is empty." }
            require(verifyIfRequired(temp, descriptor.sha256)) { "SHA-256 verification failed." }

            if (target.exists() && !target.delete()) {
                throw IllegalStateException("Could not replace old model file.")
            }
            if (!temp.renameTo(target)) {
                temp.copyTo(target, overwrite = true)
                temp.delete()
            }

            prefs.edit()
                .putString("${descriptor.id}.version", descriptor.version)
                .putString("${descriptor.id}.url", descriptor.url)
                .putLong("${descriptor.id}.bytes", target.length())
                .apply()

            ModelStatus(
                descriptor = descriptor,
                state = ModelState.READY,
                localFile = target,
                downloadedBytes = target.length(),
                totalBytes = target.length(),
            ).also(onProgress)
        } catch (error: Throwable) {
            temp.delete()
            ModelStatus(
                descriptor = descriptor,
                state = ModelState.FAILED,
                error = error.message ?: error.javaClass.simpleName,
            ).also(onProgress)
        }
    }

    override suspend fun clear(descriptor: ModelDescriptor) = withContext(Dispatchers.IO) {
        File(directory, descriptor.fileName).delete()
        File(directory, "${descriptor.fileName}.part").delete()
        prefs.edit()
            .remove("${descriptor.id}.version")
            .remove("${descriptor.id}.url")
            .remove("${descriptor.id}.bytes")
            .apply()
    }

    private fun verifyIfRequired(file: File, expected: String?): Boolean {
        if (expected.isNullOrBlank()) return true
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        return actual.equals(expected, ignoreCase = true)
    }
}
