package com.sllm.localization

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class Uploader(private val baseUrl: String, private val endpointPath: String) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    data class Result(val code: Int, val body: String)

    fun postAudio(wav: ByteArray): Result {
        val url = baseUrl.trimEnd('/') + ensureLeadingSlash(endpointPath)
        val req = Request.Builder()
            .url(url)
            .post(wav.toRequestBody(WAV_MEDIA))
            .build()
        client.newCall(req).execute().use { res ->
            return Result(res.code, res.body?.string().orEmpty())
        }
    }

    private fun ensureLeadingSlash(path: String): String =
        if (path.startsWith("/")) path else "/$path"

    companion object {
        private val WAV_MEDIA = "audio/wav".toMediaType()

        private val healthClient = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(5, TimeUnit.SECONDS)
            .build()

        /** GET {baseUrl}/health and return the architecture tags of the
         *  server's default model (driven by run.sh's AUDIO_LLM_DEFAULT).
         *  Returns an empty list if the server is unreachable or has no tags. */
        fun fetchTags(baseUrl: String): List<String> {
            val url = baseUrl.trimEnd('/') + "/health"
            val req = Request.Builder().url(url).get().build()
            healthClient.newCall(req).execute().use { res ->
                val body = res.body?.string().orEmpty()
                if (!res.isSuccessful || body.isBlank()) return emptyList()
                val root = JSONObject(body)
                val defaultModel = root.optString("default_model", "")
                val models = root.optJSONObject("models") ?: return emptyList()
                val entry = models.optJSONObject(defaultModel) ?: return emptyList()
                val arr = entry.optJSONArray("tags") ?: return emptyList()
                return (0 until arr.length())
                    .map { arr.optString(it, "") }
                    .filter { it.isNotBlank() }
            }
        }
    }
}
