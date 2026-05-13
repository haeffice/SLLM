package com.sllm.localization

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
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
    }
}
