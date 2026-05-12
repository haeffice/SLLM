package com.sllm.localization

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

/**
 * Captures stereo PCM16 audio at 48 kHz using AudioSource.CAMCORDER (which on
 * Galaxy devices typically exposes a 2-mic stereo stream). Emits 2-second WAV
 * chunks via [onChunk].
 *
 * Each emitted chunk is a complete WAV byte array (44-byte header + interleaved
 * stereo PCM16 data) ready to POST to the backend as `audio/wav`.
 */
class AudioRecorder(
    private val onChunk: (ByteArray) -> Unit,
    private val onError: (Throwable) -> Unit,
) {
    companion object {
        const val SAMPLE_RATE = 48_000
        const val CHANNELS = 2
        const val BITS_PER_SAMPLE = 16
        const val CHUNK_SECONDS = 2
        const val SHORTS_PER_CHUNK = SAMPLE_RATE * CHANNELS * CHUNK_SECONDS
        const val BYTES_PER_CHUNK = SHORTS_PER_CHUNK * 2
    }

    @Volatile private var running = false
    private var record: AudioRecord? = null
    private var worker: Thread? = null

    @SuppressLint("MissingPermission")
    fun start() {
        if (running) return
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_STEREO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBuf <= 0) {
            onError(IllegalStateException("AudioRecord.getMinBufferSize failed: $minBuf"))
            return
        }
        val bufSize = maxOf(minBuf, BYTES_PER_CHUNK / 4)

        val rec = AudioRecord(
            MediaRecorder.AudioSource.CAMCORDER,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_STEREO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize,
        )
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            onError(IllegalStateException("AudioRecord init failed (state=${rec.state})"))
            return
        }
        record = rec
        running = true
        rec.startRecording()

        worker = thread(name = "audio-recorder", isDaemon = true) {
            val pcm = ShortArray(SHORTS_PER_CHUNK)
            try {
                while (running) {
                    var filled = 0
                    while (filled < SHORTS_PER_CHUNK && running) {
                        val read = rec.read(pcm, filled, SHORTS_PER_CHUNK - filled)
                        if (read < 0) {
                            onError(IllegalStateException("AudioRecord.read error: $read"))
                            running = false
                            break
                        }
                        filled += read
                    }
                    if (filled == SHORTS_PER_CHUNK) {
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
        header.putInt(SAMPLE_RATE)
        header.putInt(SAMPLE_RATE * CHANNELS * BITS_PER_SAMPLE / 8)
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
