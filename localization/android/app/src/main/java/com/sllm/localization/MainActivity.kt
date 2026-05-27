package com.sllm.localization

import android.Manifest
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.view.animation.LinearInterpolator
import android.widget.LinearLayout
import android.widget.TextView
import com.google.android.material.color.MaterialColors
import kotlin.math.ceil
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
    private var chunkCountdown: ValueAnimator? = null

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

        setStatusIndicator(null)
        renderTags(fallbackTags()) // offline placeholder until /health responds
        refreshTags()
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
                val urlChanged = url != serverUrl
                serverUrl = url
                sampleRate = sr
                lengthSeconds = len
                outputChannels = if (radioMono.isChecked) 1 else 2
                endpointPath = if (radioInference.isChecked) "/inference" else "/localize"
                if (urlChanged) refreshTags() // re-pull tags from the new server
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
        setStatusIndicator(null) // start fresh — wait for first response
        binding.responseView.text = getString(R.string.response_body_default) // clear previous result
        playWaveCue()
        startChunkCountdown(lengthSeconds)
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
        cancelChunkCountdown()
        setStatusIndicator(null)
    }

    /** 현재 녹음 중인 chunk가 서버로 떠날 때까지 남은 시간을 progress bar +
     *  텍스트로 표시. `lengthSeconds`마다 자동으로 다시 시작되도록 무한 반복. */
    private fun startChunkCountdown(seconds: Int) {
        cancelChunkCountdown()
        val total = seconds * 100
        binding.chunkProgress.max = total
        binding.chunkProgress.progress = total
        binding.countdownText.text = getString(R.string.countdown_format, seconds)

        chunkCountdown = ValueAnimator.ofInt(total, 0).apply {
            duration = seconds * 1000L
            interpolator = LinearInterpolator()
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
            addUpdateListener { anim ->
                val current = anim.animatedValue as Int
                binding.chunkProgress.progress = current
                val secsLeft = ceil(current / 100.0).toInt().coerceAtLeast(0)
                binding.countdownText.text = getString(R.string.countdown_format, secsLeft)
            }
            start()
        }
    }

    private fun cancelChunkCountdown() {
        chunkCountdown?.cancel()
        chunkCountdown = null
        binding.chunkProgress.progress = 0
        binding.countdownText.text = getString(R.string.countdown_default)
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

    /** 응답 표시. 글씨는 layout에서 bold 고정.
     *  batch inference(2개 이상)일 때만 색을 입힌다:
     *   - 1번째 response  : 전체 푸른색
     *   - 2번째 response  : ','로 split해 마지막 조각은 보라색, 나머지는 초록색
     *   - 3번째 이후       : 색 없이 기본색(bold만)
     *  단일 응답(/localize·에러 등)은 색 없이 기본색. */
    private fun showResponse(code: Int, lines: List<String>) {
        setStatusIndicator(code)
        if (lines.size < 2) {
            binding.responseView.text = lines.joinToString(separator = "\n")
            return
        }

        val blue = ContextCompat.getColor(this, R.color.resp_first)
        val green = ContextCompat.getColor(this, R.color.resp_rest)
        val purple = ContextCompat.getColor(this, R.color.resp_last)

        val sb = SpannableStringBuilder()
        lines.forEachIndexed { i, line ->
            val start = sb.length
            sb.append(line)
            val end = sb.length
            when (i) {
                0 -> sb.color(blue, start, end)
                1 -> {
                    val lastComma = line.lastIndexOf(',')
                    if (lastComma >= 0) {
                        sb.color(green, start, start + lastComma + 1) // 콤마 포함 앞부분
                        sb.color(purple, start + lastComma + 1, end)  // 마지막 조각
                    } else {
                        sb.color(purple, start, end) // 콤마 없으면 전체가 '마지막 조각'
                    }
                }
                // i >= 2: 색 없음 (기본색 + bold)
            }
            if (i != lines.lastIndex) sb.append("\n")
        }
        binding.responseView.text = sb
    }

    private fun SpannableStringBuilder.color(color: Int, start: Int, end: Int) {
        if (start >= end) return
        setSpan(ForegroundColorSpan(color), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    }

    // -------------------------------------------------------------------------
    // Model 태그 (BE /health 기반)
    // -------------------------------------------------------------------------

    /** 서버에서 default 모델의 아키텍처 태그를 받아 우측 상단 chip row에 표시.
     *  실패하면 직전 태그(또는 fallback)를 그대로 둔다. */
    private fun refreshTags() {
        val url = serverUrl
        if (url.isBlank()) return
        lifecycleScope.launch {
            val tags = runCatching {
                withContext(Dispatchers.IO) { Uploader.fetchTags(url) }
            }.getOrDefault(emptyList())
            if (tags.isNotEmpty()) renderTags(tags)
        }
    }

    /** chip row를 비우고 태그 TextView들을 동적으로 채운다(layout의 정적 chip 대체). */
    private fun renderTags(tags: List<String>) {
        val row = binding.tagRow
        row.removeAllViews()
        val padH = dpToPx(10f)
        val padV = dpToPx(3f)
        val gap = dpToPx(6f)
        tags.forEachIndexed { i, tag ->
            val chip = TextView(this).apply {
                text = tag
                setBackgroundResource(R.drawable.bg_tag)
                setPadding(padH, padV, padH, padV)
                textSize = 10f
            }
            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { if (i > 0) marginStart = gap }
            row.addView(chip, lp)
        }
    }

    /** 서버 응답 전/오프라인 시 보여줄 기본 태그(BE의 bat 모델 값과 동일). */
    private fun fallbackTags(): List<String> = listOf(
        getString(R.string.tag_spatial_ast),
        getString(R.string.tag_llama_adaptor),
        getString(R.string.tag_llama2_7b),
        getString(R.string.tag_lora),
    )

    private fun showError(message: String) {
        // 네트워크/업로드 오류엔 HTTP code가 없지만 4xx와 같은 의미로 빨간 채움.
        setStatusIndicator(code = 400)
        binding.responseView.text = message
    }

    /** 우측 상단의 작은 status indicator. null이면 빈 동그라미(테두리만),
     *  값이 있으면 2xx 초록 / 4xx 빨강 / 5xx 보라로 채운 원. */
    private fun setStatusIndicator(code: Int?) {
        val view = binding.statusIndicator
        val drawable = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            when {
                code == null -> {
                    setColor(Color.TRANSPARENT)
                    val strokeColor = MaterialColors.getColor(
                        view,
                        com.google.android.material.R.attr.colorOutline,
                    )
                    setStroke(dpToPx(1.5f), strokeColor)
                }
                code in 200..299 ->
                    setColor(ContextCompat.getColor(this@MainActivity, R.color.status_2xx))
                code in 400..499 ->
                    setColor(ContextCompat.getColor(this@MainActivity, R.color.status_4xx))
                code in 500..599 ->
                    setColor(ContextCompat.getColor(this@MainActivity, R.color.status_5xx))
                else -> {
                    // 1xx/3xx 등 색 규칙 외: 테두리만
                    setColor(Color.TRANSPARENT)
                    val strokeColor = MaterialColors.getColor(
                        view,
                        com.google.android.material.R.attr.colorOnSurface,
                    )
                    setStroke(dpToPx(1.5f), strokeColor)
                }
            }
        }
        view.background = drawable
    }

    private fun dpToPx(dp: Float): Int =
        (dp * resources.displayMetrics.density).toInt()
}
