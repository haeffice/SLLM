package com.sllm.localization

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

/**
 * Captures stereo PCM16 audio using AudioSource.CAMCORDER (which on Galaxy
 * devices typically exposes a 2-mic stereo stream). Sample rate is resolved
 * once at construction from AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE — the
 * standard Android proxy for the device's native HAL clock — and falls
 * back to [DEFAULT_SAMPLE_RATE] when the property is unavailable.
 *
 * Each emitted chunk is a complete WAV byte array (44-byte header +
 * interleaved stereo PCM16 data) ready to POST as `audio/wav`.
 */
class AudioRecorder(
    context: Context,
    private val onChunk: (ByteArray) -> Unit,
    private val onError: (Throwable) -> Unit,
) {
    companion object {
        const val DEFAULT_SAMPLE_RATE = 48_000
        const val CHANNELS = 2
        const val BITS_PER_SAMPLE = 16
        const val CHUNK_SECONDS = 2
    }

    val sampleRate: Int = resolveSampleRate(context)
    private val shortsPerChunk: Int = sampleRate * CHANNELS * CHUNK_SECONDS
    private val bytesPerChunk: Int = shortsPerChunk * 2

    @Volatile private var running = false
    private var record: AudioRecord? = null
    private var worker: Thread? = null

    private fun resolveSampleRate(context: Context): Int {
        val am = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
        val hinted = am?.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE)?.toIntOrNull()
        return hinted ?: DEFAULT_SAMPLE_RATE
    }

    @SuppressLint("MissingPermission")
    fun start() {
        if (running) return
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_STEREO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBuf <= 0) {
            onError(IllegalStateException("AudioRecord.getMinBufferSize failed: $minBuf"))
            return
        }
        val bufSize = maxOf(minBuf, bytesPerChunk / 4)

        val rec = AudioRecord(
            MediaRecorder.AudioSource.CAMCORDER,
            sampleRate,
            AudioFormat.CHANNEL_IN_STEREO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize,
        )
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            onError(IllegalStateException("AudioRecord init failed (state=${rec.state}, sr=$sampleRate)"))
            return
        }
        record = rec
        running = true
        rec.startRecording()

        worker = thread(name = "audio-recorder", isDaemon = true) {
            val pcm = ShortArray(shortsPerChunk)
            try {
                while (running) {
                    var filled = 0
                    while (filled < shortsPerChunk && running) {
                        val read = rec.read(pcm, filled, shortsPerChunk - filled)
                        if (read < 0) {
                            onError(IllegalStateException("AudioRecord.read error: $read"))
                            running = false
                            break
                        }
                        filled += read
                    }
                    if (filled == shortsPerChunk) {
                        onChunk(encodeWav(pcm))
                    }
                }
            } catch (t: Throwable) {
                onError(t)
            } finally {
                try {
                    rec.stop()
                } catch (_: Throwable) {}
                rec.release()
                record = null
            }
        }
    }

    fun stop() {
        running = false
        worker?.join(1_000)
        worker = null
    }

    private fun encodeWav(pcm: ShortArray): ByteArray {
        val dataSize = pcm.size * 2
        val out = ByteArrayOutputStream(44 + dataSize)
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(36 + dataSize)
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)
        header.putShort(1)
        header.putShort(CHANNELS.toShort())
        header.putInt(sampleRate)
        header.putInt(sampleRate * CHANNELS * BITS_PER_SAMPLE / 8)
        header.putShort((CHANNELS * BITS_PER_SAMPLE / 8).toShort())
        header.putShort(BITS_PER_SAMPLE.toShort())
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(dataSize)
        out.write(header.array())

        val body = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
        for (s in pcm) body.putShort(s)
        out.write(body.array())
        return out.toByteArray()
    }
}
