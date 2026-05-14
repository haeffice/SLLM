package com.sllm.localization

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.radiobutton.MaterialRadioButton
import com.google.android.material.textfield.TextInputEditText
import com.sllm.localization.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var recorder: AudioRecorder? = null

    // --- Settings state ------------------------------------------------------
    private var sampleRate = 16_000
    private var lengthSeconds = 10
    private var outputChannels = 2          // 1 = mono, 2 = stereo
    private var endpointPath = "/localize"  // "/localize" or "/inference"

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startRecording()
        else showError("마이크 권한 필요")
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
        binding.settingsBtn.setOnClickListener { showSettingsDialog() }
    }

    override fun onDestroy() {
        recorder?.stop()
        recorder = null
        super.onDestroy()
    }

    private fun hasMicPermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.RECORD_AUDIO
    ) == PackageManager.PERMISSION_GRANTED

    // -------------------------------------------------------------------------
    // Settings dialog
    // -------------------------------------------------------------------------

    private fun showSettingsDialog() {
        val view = layoutInflater.inflate(R.layout.dialog_settings, null)
        val srInput = view.findViewById<TextInputEditText>(R.id.sampleRateInput)
        val lenInput = view.findViewById<TextInputEditText>(R.id.lengthSecondsInput)
        val radioMono = view.findViewById<MaterialRadioButton>(R.id.radioMono)
        val radioStereo = view.findViewById<MaterialRadioButton>(R.id.radioStereo)
        val radioLocalize = view.findViewById<MaterialRadioButton>(R.id.radioLocalize)
        val radioInference = view.findViewById<MaterialRadioButton>(R.id.radioInference)

        srInput.setText(sampleRate.toString())
        lenInput.setText(lengthSeconds.toString())
        radioMono.isChecked = outputChannels == 1
        radioStereo.isChecked = outputChannels == 2
        radioLocalize.isChecked = endpointPath == "/localize"
        radioInference.isChecked = endpointPath == "/inference"

        AlertDialog.Builder(this)
            .setTitle(R.string.dialog_settings_title)
            .setView(view)
            .setPositiveButton(R.string.btn_save) { _, _ ->
                val sr = srInput.text?.toString()?.trim()?.toIntOrNull()
                val len = lenInput.text?.toString()?.trim()?.toIntOrNull()
                if (sr == null || sr <= 0) {
                    showError("Sample rate는 양의 정수여야 합니다")
                    return@setPositiveButton
                }
                if (len == null || len <= 0) {
                    showError("Audio length는 양의 정수여야 합니다")
                    return@setPositiveButton
                }
                sampleRate = sr
                lengthSeconds = len
                outputChannels = if (radioMono.isChecked) 1 else 2
                endpointPath = if (radioInference.isChecked) "/inference" else "/localize"
            }
            .setNegativeButton(R.string.btn_cancel, null)
            .show()
    }

    // -------------------------------------------------------------------------
    // Recording lifecycle
    // -------------------------------------------------------------------------

    private fun startRecording() {
        if (recorder != null) return

        val serverUrl = binding.serverUrlInput.text?.toString()?.trim().orEmpty()
        if (serverUrl.isEmpty()) {
            showError("Server URL을 입력하세요")
            return
        }

        val uploader = Uploader(serverUrl, endpointPath)

        val rec = AudioRecorder(
            sampleRate = sampleRate,
            outputChannels = outputChannels,
            chunkSeconds = lengthSeconds,
            onChunk = { wav ->
                lifecycleScope.launch { uploadAndShow(uploader, wav) }
            },
            onError = { e ->
                runOnUiThread {
                    showError("recorder error: ${e.message}")
                    cleanupRecorder()
                }
            },
        )
        recorder = rec.also { it.start() }

        setControlsEnabled(recording = true)
    }

    private fun stopRecording() {
        cleanupRecorder()
    }

    private fun cleanupRecorder() {
        recorder?.stop()
        recorder = null
        setControlsEnabled(recording = false)
    }

    private fun setControlsEnabled(recording: Boolean) {
        binding.startBtn.isEnabled = !recording
        binding.stopBtn.isEnabled = recording
        binding.serverUrlInput.isEnabled = !recording
        binding.settingsBtn.isEnabled = !recording
        binding.settingsBtn.alpha = if (recording) 0.4f else 1f
    }

    // -------------------------------------------------------------------------
    // Network + response display
    // -------------------------------------------------------------------------

    private suspend fun uploadAndShow(uploader: Uploader, wav: ByteArray) {
        val result = runCatching {
            withContext(Dispatchers.IO) { uploader.postAudio(wav) }
        }
        result.onSuccess { r ->
            showResponse(r.code, extractDisplayText(r.body))
        }.onFailure { e ->
            showError("upload error: ${e.message}")
        }
    }

    /** JSON 응답에서 화면에 보일 한 줄을 골라냄.
     *  - 정상 응답: "response" 필드
     *  - 에러 응답: "detail" 필드
     *  - 둘 다 없으면 본문 원문 */
    private fun extractDisplayText(body: String): String {
        if (body.isBlank()) return ""
        return try {
            val obj = JSONObject(body)
            when {
                obj.has("response") -> obj.optString("response", body)
                obj.has("detail") -> obj.optString("detail", body)
                else -> body
            }
        } catch (_: Throwable) {
            body
        }
    }

    private fun showResponse(code: Int, text: String) {
        val colorRes = when (code) {
            in 200..299 -> R.color.status_2xx
            in 400..499 -> R.color.status_4xx
            in 500..599 -> R.color.status_5xx
            else -> null
        }
        binding.statusCodeView.text = code.toString()
        if (colorRes != null) {
            binding.statusCodeView.setTextColor(ContextCompat.getColor(this, colorRes))
        } else {
            binding.statusCodeView.setTextColor(
                com.google.android.material.color.MaterialColors.getColor(
                    binding.statusCodeView,
                    com.google.android.material.R.attr.colorOnSurface,
                )
            )
        }
        binding.responseView.text = text
    }

    private fun showError(message: String) {
        binding.statusCodeView.text = getString(R.string.status_code_default)
        binding.statusCodeView.setTextColor(
            ContextCompat.getColor(this, R.color.status_4xx)
        )
        binding.responseView.text = message
    }
}
