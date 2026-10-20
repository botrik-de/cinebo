import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Jede Version aus GitHub Actions bekommt eine eigene, steigende Nummer.
// Der Name (z. B. 0.2.7) kommt aus dem Bau-Ablauf, damit App und Release gleich heißen.
val buildNumber = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
val versionLabel = System.getenv("CINEBO_VERSION") ?: "0.2.$buildNumber"

// Eine Stelle für Qualcomms QNN-Version (gradle.properties): Der Delegate in der App und die
// QNN-Laufzeit, die die App selbst lädt, müssen zusammenpassen.
val qnnVersion: String = providers.gradleProperty("cinebo.qnnVersion").get()

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
        // Für den Bericht: welche QNN-Version die App erwartet.
        buildConfigField("String", "QNN_VERSION", "\"$qnnVersion\"")
    }

    buildFeatures {
        buildConfig = true
    }

    packaging {
        jniLibs {
            // Bibliotheken gepackt in der APK, beim Installieren entpackt: kleinerer Download.
            useLegacyPackaging = true
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

    // KI: TFLite und Qualcomms QNN-Delegate für den KI-Chip, in der Kombination, die auch
    // Qualcomms eigene Beispiel-Apps verwenden. Die große QNN-Laufzeit (qnn-runtime, gleiche
    // Version) steckt nicht in der APK: Die App lädt sie mit den Modellen selbst
    // (siehe ModelStore und Npu.loadLibraries).
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("com.qualcomm.qti:qnn-litert-delegate:$qnnVersion")

    // Prüfungen ohne Handy, bei jedem Bau (app/src/test)
    testImplementation("junit:junit:4.13.2")
}
