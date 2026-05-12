plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// --- Versioning ---------------------------------------------------------
// versionName is derived from the latest `app-demo-v*` git tag, with the
// patch number bumped by one. If no such tag exists yet, defaults to
// "0.1.0". versionCode is computed deterministically from versionName so
// every published version maps to a unique integer.
//
// Workflow: tag a release as `app-demo-vX.Y.Z` → next build's versionName
// becomes `X.Y.(Z+1)`. Override with `-PappVersion=...` if needed.

fun runGit(vararg args: String): String? = try {
    val proc = ProcessBuilder("git", *args)
        .directory(rootDir)
        .redirectErrorStream(false)
        .start()
    proc.waitFor()
    if (proc.exitValue() == 0)
        proc.inputStream.bufferedReader().readText().trim().takeIf { it.isNotEmpty() }
    else null
} catch (_: Exception) { null }

fun computeVersionName(): String {
    val override = (project.findProperty("appVersion") as String?)?.takeIf { it.isNotBlank() }
    if (override != null) return override

    val lastTag = runGit("describe", "--tags", "--abbrev=0", "--match", "app-demo-v*")
        ?: return "0.1.0"
    val parts = lastTag.removePrefix("app-demo-v").split(".")
    if (parts.size != 3) return "0.1.0"
    val nums = parts.map { it.toIntOrNull() ?: return "0.1.0" }
    return "${nums[0]}.${nums[1]}.${nums[2] + 1}"
}

fun computeVersionCode(name: String): Int {
    val parts = name.split(".").mapNotNull { it.toIntOrNull() }
    if (parts.size != 3) return 1
    return parts[0] * 10_000 + parts[1] * 100 + parts[2]
}

val appVersionName: String = computeVersionName()
val appVersionCode: Int = computeVersionCode(appVersionName)
logger.lifecycle("Building Localization v$appVersionName (versionCode=$appVersionCode)")

android {
    namespace = "com.sllm.localization"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.sllm.localization"
        minSdk = 26
        targetSdk = 34
        versionCode = appVersionCode
        versionName = appVersionName
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    applicationVariants.all {
        outputs.all {
            (this as com.android.build.gradle.internal.api.BaseVariantOutputImpl)
                .outputFileName = "Localization-v$appVersionName-${buildType.name}.apk"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.activity:activity-ktx:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
