import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Jede Version aus GitHub Actions bekommt eine eigene, steigende Nummer.
// Der Name (z. B. 0.2.7) kommt aus dem Bau-Ablauf, damit App und Release gleich heißen.
val buildNumber = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
val versionLabel = System.getenv("CINEBO_VERSION") ?: "0.2.$buildNumber"

android {
    namespace = "de.botrik.cinebo.test"
    compileSdk = 35

    defaultConfig {
        applicationId = "de.botrik.cinebo.test"
        minSdk = 33
        targetSdk = 35
        versionCode = buildNumber
        versionName = versionLabel
        // Der KI-Chip und seine Bibliotheken gibt es nur für 64-Bit-ARM.
        ndk { abiFilters += "arm64-v8a" }
    }

    // Modelle unkomprimiert, damit sie sich direkt aus der APK in den Speicher abbilden lassen.
    androidResources {
        noCompress += "tflite"
    }

    packaging {
        jniLibs {
            // Der KI-Chip lädt seine Programme (Skel) als Dateien; dafür müssen die
            // Bibliotheken bei der Installation entpackt werden.
            useLegacyPackaging = true
            // Nur, was das OPPO Find N6 (Snapdragon 8 Elite Gen 5, Hexagon V81) braucht.
            // V79 bleibt als Rückfall drin, falls der Chip sich anders meldet.
            excludes += listOf(
                "**/libQnnHtpV68Skel.so", "**/libQnnHtpV68Stub.so",
                "**/libQnnHtpV69Skel.so", "**/libQnnHtpV69Stub.so",
                "**/libQnnHtpV73Skel.so", "**/libQnnHtpV73Stub.so",
                "**/libQnnHtpV75Skel.so", "**/libQnnHtpV75Stub.so",
                "**/libQnnDsp.so", "**/libQnnDspV66Skel.so", "**/libQnnDspV66Stub.so",
                "**/libQnnGpu.so"
            )
        }
    }

    // Nur für die Test-App: fester Schlüssel im Projekt, damit sich jede neue
    // Version über die alte installieren lässt. Die echte App bekommt einen eigenen, geheimen Schlüssel.
    signingConfigs {
        create("test") {
            storeFile = rootProject.file("signing/test-key.p12")
            storeType = "pkcs12"
            storePassword = "cinebo-test"
            keyAlias = "cinebo-test"
            keyPassword = "cinebo-test"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("test")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint {
        abortOnError = false
        checkReleaseBuilds = false
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    val camerax = "1.4.2"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-video:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // KI: TFLite und Qualcomms QNN-Delegate für den KI-Chip, in der Kombination,
    // die auch Qualcomms eigene Beispiel-Apps verwenden.
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.50.0")
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.50.0")
}
