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
 * Captures audio using AudioSource.CAMCORDER. Hardware capture is always
 * 2-channel (stereo) regardless of [outputChannels], because Galaxy devices
 * are known to expose CAMCORDER reliably as stereo; if [outputChannels] is
 * 1, the captured PCM is mixed down to mono in-app (`(L + R) / 2`) before
 * the WAV is built. This avoids relying on OEM/HAL behavior for native
 * mono capture.
 *
 * Each emitted chunk is a complete WAV byte array (44-byte header +
 * interleaved PCM16 data) ready to POST as `audio/wav`.
 */
class AudioRecorder(
    val sampleRate: Int,
    val outputChannels: Int,
    val chunkSeconds: Int,
    private val onChunk: (ByteArray) -> Unit,
    private val onError: (Throwable) -> Unit,
) {
    companion object {
        const val BITS_PER_SAMPLE = 16
        private const val CAPTURE_CHANNELS = 2
    }

    init {
        require(outputChannels == 1 || outputChannels == 2) {
            "outputChannels must be 1 (mono) or 2 (stereo), got $outputChannels"
        }
        require(sampleRate > 0) { "sampleRate must be positive, got $sampleRate" }
        require(chunkSeconds > 0) { "chunkSeconds must be positive, got $chunkSeconds" }
    }

    private val captureShortsPerChunk: Int = sampleRate * CAPTURE_CHANNELS * chunkSeconds
    private val outputShortsPerChunk: Int = sampleRate * outputChannels * chunkSeconds
    private val outputBytesPerChunk: Int = outputShortsPerChunk * 2

    @Volatile private var running = false
    private var record: AudioRecord? = null
    private var worker: Thread? = null

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
        val bufSize = maxOf(minBuf, captureShortsPerChunk * 2 / 4)

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
            val stereo = ShortArray(captureShortsPerChunk)
            try {
                while (running) {
                    var filled = 0
                    while (filled < captureShortsPerChunk && running) {
                        val read = rec.read(stereo, filled, captureShortsPerChunk - filled)
                        if (read < 0) {
                            onError(IllegalStateException("AudioRecord.read error: $read"))
                            running = false
                            break
                        }
                        filled += read
                    }
                    if (filled == captureShortsPerChunk) {
                        val out = downmixIfNeeded(stereo)
                        onChunk(encodeWav(out))
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

    private fun downmixIfNeeded(stereo: ShortArray): ShortArray {
        if (outputChannels == 2) return stereo
        val mono = ShortArray(stereo.size / 2)
        var j = 0
        var i = 0
        while (i < stereo.size) {
            val avg = (stereo[i].toInt() + stereo[i + 1].toInt()) / 2
            mono[j++] = avg.toShort()
            i += 2
        }
        return mono
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
        header.putShort(outputChannels.toShort())
        header.putInt(sampleRate)
        header.putInt(sampleRate * outputChannels * BITS_PER_SAMPLE / 8)
        header.putShort((outputChannels * BITS_PER_SAMPLE / 8).toShort())
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
