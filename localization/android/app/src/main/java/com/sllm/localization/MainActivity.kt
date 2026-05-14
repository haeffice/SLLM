package com.sllm.localization

import android.Manifest
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
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
import org.json.JSONArray
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var recorder: AudioRecorder? = null

    // --- Settings state ------------------------------------------------------
    private var serverUrl = "http://192.168.0.42:9001"
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
        val urlInput = view.findViewById<TextInputEditText>(R.id.serverUrlInput)
        val srInput = view.findViewById<TextInputEditText>(R.id.sampleRateInput)
        val lenInput = view.findViewById<TextInputEditText>(R.id.lengthSecondsInput)
        val radioMono = view.findViewById<MaterialRadioButton>(R.id.radioMono)
        val radioStereo = view.findViewById<MaterialRadioButton>(R.id.radioStereo)
        val radioLocalize = view.findViewById<MaterialRadioButton>(R.id.radioLocalize)
        val radioInference = view.findViewById<MaterialRadioButton>(R.id.radioInference)

        urlInput.setText(serverUrl)
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
                val url = urlInput.text?.toString()?.trim().orEmpty()
                if (url.isEmpty()) {
                    showError("Server URL을 입력하세요")
                    return@setPositiveButton
                }
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
                serverUrl = url
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

        if (serverUrl.isBlank()) {
            showError("Server URL을 설정에서 입력하세요")
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
        playWaveCue()
    }

    /** Start 직후 좌/우 채널 label을 잠시 띄웠다가 fade out시키는 효과.
     *  pulse-like scale + alpha 변화를 동시에 줘 음향 파동 느낌. */
    private fun playWaveCue() {
        listOf(binding.leftWaveLabel, binding.rightWaveLabel).forEach { v ->
            v.animate().cancel()
            v.alpha = 0f
            v.scaleX = 0.8f
            v.scaleY = 0.8f
            v.visibility = View.VISIBLE

            val fadeIn = AnimatorSet().apply {
                playTogether(
                    ObjectAnimator.ofFloat(v, "alpha", 0f, 1f),
                    ObjectAnimator.ofFloat(v, "scaleX", 0.8f, 1.2f),
                    ObjectAnimator.ofFloat(v, "scaleY", 0.8f, 1.2f),
                )
                duration = 300
                interpolator = AccelerateDecelerateInterpolator()
            }
            val pulse = AnimatorSet().apply {
                playTogether(
                    ObjectAnimator.ofFloat(v, "scaleX", 1.2f, 1.0f),
                    ObjectAnimator.ofFloat(v, "scaleY", 1.2f, 1.0f),
                )
                duration = 250
                startDelay = 300
                interpolator = AccelerateDecelerateInterpolator()
            }
            val fadeOut = AnimatorSet().apply {
                playTogether(
                    ObjectAnimator.ofFloat(v, "alpha", 1f, 0f),
                )
                duration = 700
                startDelay = 800
                interpolator = AccelerateDecelerateInterpolator()
            }
            AnimatorSet().apply {
                playTogether(fadeIn, pulse, fadeOut)
                start()
            }
        }
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
            showResponse(r.code, extractDisplayLines(r.body))
        }.onFailure { e ->
            showError("upload error: ${e.message}")
        }
    }

    /** JSON 응답에서 화면에 띄울 lines 목록 추출.
     *  - 우선순위: `responses`(array) → `response`(string) → `detail`(string) → 원문
     */
    private fun extractDisplayLines(body: String): List<String> {
        if (body.isBlank()) return listOf("")
        return try {
            val obj = JSONObject(body)
            when {
                obj.has("responses") && obj.get("responses") is JSONArray -> {
                    val arr = obj.getJSONArray("responses")
                    List(arr.length()) { i -> arr.optString(i, "") }
                }
                obj.has("response") -> listOf(obj.optString("response", body))
                obj.has("detail") -> listOf(obj.optString("detail", body))
                else -> listOf(body)
            }
        } catch (_: Throwable) {
            listOf(body)
        }
    }

    private fun showResponse(code: Int, lines: List<String>) {
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
        // 응답이 여러 개면 한 줄씩 띄워서 표시 (XML의 lineSpacingExtra=6dp 적용됨)
        binding.responseView.text = lines.joinToString(separator = "\n")
    }

    private fun showError(message: String) {
        binding.statusCodeView.text = getString(R.string.status_code_default)
        binding.statusCodeView.setTextColor(
            ContextCompat.getColor(this, R.color.status_4xx)
        )
        binding.responseView.text = message
    }
}
