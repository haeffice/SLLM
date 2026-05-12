package com.sllm.localization

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.text.format.DateFormat
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.sllm.localization.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.Date

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var recorder: AudioRecorder? = null

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startRecording()
        else setStatus(getString(R.string.status_no_permission))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.startBtn.setOnClickListener {
            if (hasMicPermission()) startRecording()
            else requestPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
        binding.stopBtn.setOnClickListener { stopRecording() }
    }

    override fun onDestroy() {
        recorder?.stop()
        recorder = null
        super.onDestroy()
    }

    private fun hasMicPermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.RECORD_AUDIO
    ) == PackageManager.PERMISSION_GRANTED

    private fun startRecording() {
        if (recorder != null) return
        val relayUrl = binding.relayUrlInput.text?.toString()?.trim().orEmpty()
        if (relayUrl.isEmpty()) {
            setStatus("Relay URL을 입력하세요")
            return
        }
        val uploader = Uploader(relayUrl)

        val rec = AudioRecorder(
            context = this,
            onChunk = { wav ->
                lifecycleScope.launch { uploadAndShow(uploader, wav) }
            },
            onError = { e ->
                runOnUiThread {
                    setStatus(getString(R.string.status_error))
                    appendLog("recorder error: ${e.message}", error = true)
                    cleanupRecorder()
                }
            },
        )
        recorder = rec.also { it.start() }

        binding.startBtn.isEnabled = false
        binding.stopBtn.isEnabled = true
        binding.relayUrlInput.isEnabled = false
        setStatus(getString(R.string.status_recording))
        appendLog("녹음 시작: ${rec.sampleRate}Hz stereo, ${AudioRecorder.CHUNK_SECONDS}s chunks → $relayUrl")
    }

    private fun stopRecording() {
        cleanupRecorder()
        setStatus(getString(R.string.status_stopped))
        appendLog("녹음 종료")
    }

    private fun cleanupRecorder() {
        recorder?.stop()
        recorder = null
        binding.startBtn.isEnabled = true
        binding.stopBtn.isEnabled = false
        binding.relayUrlInput.isEnabled = true
    }

    private suspend fun uploadAndShow(uploader: Uploader, wav: ByteArray) {
        val result = runCatching {
            withContext(Dispatchers.IO) { uploader.postLocalize(wav) }
        }
        result.onSuccess { r ->
            val az = parseAzimuth(r.body)
            if (az != null) binding.azimuthView.text = "azimuth: ${"%.1f".format(az)}°"
            appendLog("[${r.code}] ${r.body}", error = r.code !in 200..299)
        }.onFailure { e ->
            appendLog("upload error: ${e.message}", error = true)
        }
    }

    private fun parseAzimuth(body: String): Double? = try {
        val obj = JSONObject(body)
        when {
            obj.has("azimuth_degrees") -> obj.getDouble("azimuth_degrees")
            obj.has("azimuth") -> obj.getDouble("azimuth")
            else -> null
        }
    } catch (_: Throwable) { null }

    private fun setStatus(text: String) {
        binding.statusView.text = text
    }

    private fun appendLog(line: String, error: Boolean = false) {
        val ts = DateFormat.format("HH:mm:ss", Date())
        val prefix = if (error) "! " else "  "
        binding.logView.append("$ts $prefix$line\n\n")
        binding.logScroll.post {
            binding.logScroll.fullScroll(android.view.View.FOCUS_DOWN)
        }
    }
}
