package com.sllm.localization

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.sllm.localization.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var recorder: AudioRecorder? = null

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

        val serverUrl = binding.serverUrlInput.text?.toString()?.trim().orEmpty()
        if (serverUrl.isEmpty()) {
            showError("Server URL을 입력하세요")
            return
        }

        val sampleRate = binding.sampleRateInput.text?.toString()?.trim()?.toIntOrNull()
        if (sampleRate == null || sampleRate <= 0) {
            showError("Sample rate는 양의 정수여야 합니다")
            return
        }

        val lengthSec = binding.lengthSecondsInput.text?.toString()?.trim()?.toIntOrNull()
        if (lengthSec == null || lengthSec <= 0) {
            showError("Audio length는 양의 정수여야 합니다")
            return
        }

        val outputChannels = if (binding.radioMono.isChecked) 1 else 2
        val endpointPath = if (binding.radioInference.isChecked) "/inference" else "/localize"

        val uploader = Uploader(serverUrl, endpointPath)

        val rec = AudioRecorder(
            sampleRate = sampleRate,
            outputChannels = outputChannels,
            chunkSeconds = lengthSec,
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

        setSettingsEnabled(false)
        binding.startBtn.isEnabled = false
        binding.stopBtn.isEnabled = true
    }

    private fun stopRecording() {
        cleanupRecorder()
    }

    private fun cleanupRecorder() {
        recorder?.stop()
        recorder = null
        setSettingsEnabled(true)
        binding.startBtn.isEnabled = true
        binding.stopBtn.isEnabled = false
    }

    private fun setSettingsEnabled(enabled: Boolean) {
        setGroupEnabled(binding.settingsPanel, enabled)
    }

    private fun setGroupEnabled(group: ViewGroup, enabled: Boolean) {
        group.isEnabled = enabled
        for (i in 0 until group.childCount) {
            val child = group.getChildAt(i)
            if (child is ViewGroup) setGroupEnabled(child, enabled)
            else child.isEnabled = enabled
        }
    }

    private suspend fun uploadAndShow(uploader: Uploader, wav: ByteArray) {
        val result = runCatching {
            withContext(Dispatchers.IO) { uploader.postAudio(wav) }
        }
        result.onSuccess { r ->
            showResponse(r.code, r.body)
        }.onFailure { e ->
            showError("upload error: ${e.message}")
        }
    }

    private fun showResponse(code: Int, body: String) {
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
        binding.responseBodyView.text = body
    }

    private fun showError(message: String) {
        binding.statusCodeView.text = "—"
        binding.statusCodeView.setTextColor(
            ContextCompat.getColor(this, R.color.status_4xx)
        )
        binding.responseBodyView.text = message
    }
}
