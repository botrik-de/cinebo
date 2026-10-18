package de.botrik.cinebo.test

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.provider.Settings
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextUtils
import android.text.style.ForegroundColorSpan
import android.util.Range
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.ViewOutlineProvider
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.DynamicRange
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.RecordingStats
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.Observer
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.coroutines.resume
import kotlin.math.max
import kotlin.math.min

/**
 * Test-App für Phase 1 (Machbarkeitstest): Steckbrief, 4K-Aufnahmen mit 30 und 60 fps,
 * Stabilisator, Klappen während der Aufnahme und ein Dauertest über 10 Minuten.
 * Alles landet in einem Bericht, den man kopieren oder teilen kann.
 */
class MainActivity : ComponentActivity() {

    private companion object {
        val ORDER = listOf("steckbrief", "k30", "k60", "stab", "klapp", "dauer")

        val TITLES = mapOf(
            "steckbrief" to "Steckbrief",
            "k30" to "4K · 30 fps",
            "k60" to "4K · 60 fps",
            "stab" to "4K · Stabilisator",
            "klapp" to "Klapptest · 30 s",
            "dauer" to "Dauertest · 10 min"
        )

        val DESCRIPTIONS = mapOf(
            "steckbrief" to "Liest aus, was die Kameras fremden Apps erlauben.",
            "k30" to "10 Sekunden in 3840 × 2160 mit 30 Bildern pro Sekunde.",
            "k60" to "Prüft, ob 4K auch mit 60 Bildern pro Sekunde geht.",
            "stab" to "10 Sekunden mit eingeschaltetem Bildstabilisator.",
            "klapp" to "Klappen während der Aufnahme: Läuft sie weiter?",
            "dauer" to "10 Minuten 4K am Stück: Wärme, Akku und Bildrate."
        )

        const val CLIP_SECONDS = 10
        const val FOLD_SECONDS = 30
        const val LONG_SECONDS = 600
        const val LOG_EVERY_S = 30.0

        val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
        val WRAP = ViewGroup.LayoutParams.WRAP_CONTENT
    }

    /** Was der Statuspunkt gerade zeigt. */
    private enum class Phase(val color: Int) {
        IDLE(Palette.textFaint),
        BUSY(Palette.amber),
        REC(Palette.red),
        DONE(Palette.green),
        ERROR(Palette.red)
    }

    // ---------------------------------------------------------------- Zustand

    private lateinit var store: ReportStore
    private var report = JSONObject()
    private var provider: ProcessCameraProvider? = null
    private var cameraLoading = false
    private var preview: Preview? = null
    private val frames = FrameStats()
    @Volatile private var counting = false

    private var job: Job? = null
    private var running: String? = null
    private var stopRequested = false
    private var failed = false
    private var doneText: String? = null
    private var hintOverride: String? = null
    private var events: JSONArray? = null
    private var recordingSince = 0L
    private var lastScreen = ""
    private var resetArmed = false

    private val appVersion: String by lazy {
        try {
            packageManager.getPackageInfo(packageName, PackageManager.PackageInfoFlags.of(0)).versionName ?: "?"
        } catch (_: Exception) {
            "?"
        }
    }

    // ---------------------------------------------------------------- Ansichten

    private lateinit var previewBox: FrameLayout
    private lateinit var previewView: PreviewView
    private lateinit var recBadge: TextView
    private lateinit var permissionPanel: LinearLayout
    private lateinit var statusDot: View
    private lateinit var statusText: TextView
    private lateinit var timeText: TextView
    private lateinit var progress: ProgressLine
    private lateinit var mainButton: PillButton
    private lateinit var foldButton: PillButton
    private lateinit var longButton: PillButton
    private lateinit var hintText: TextView
    private lateinit var resultScroll: ScrollView
    private lateinit var resultCard: LinearLayout
    private val rows = LinkedHashMap<String, ResultRow>()
    private lateinit var errorRow: ResultRow
    private lateinit var errorDivider: View
    private lateinit var resetButton: TextView
    private lateinit var copyButton: PillButton
    private lateinit var shareButton: PillButton

    // ---------------------------------------------------------------- Kamera-Rückmeldungen

    /** Zählt jedes Bild, das die Kamera während einer Aufnahme liefert. */
    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            if (counting) frames.add(result)
        }
    }

    private val permissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            if (has(Manifest.permission.CAMERA)) startCamera() else showPermissionPanel()
        }

    /** Zurück während eines Tests stoppt nur den Test. */
    private val backDuringTest = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() {
            requestStop()
        }
    }

    // ---------------------------------------------------------------- Lebenszyklus

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT)
        )
        super.onCreate(savedInstanceState)
        CrashLog.install(this)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        store = ReportStore(this)
        report = store.load()
        val crash = CrashLog.take(this)
        if (crash != null) {
            report.put("letzterAbsturz", crash)
            store.save(report)
        }
        removeOldClips()

        setContentView(buildUi())
        onBackPressedDispatcher.addCallback(this, backDuringTest)
        lastScreen = screenKey(resources.configuration)
        setStatus(Phase.IDLE, "Kamera startet …")
        render()

        if (has(Manifest.permission.CAMERA) && has(Manifest.permission.RECORD_AUDIO)) {
            startCamera()
        } else {
            permissionRequest.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
        }
    }

    override fun onStart() {
        super.onStart()
        logEvent("im Vordergrund")
    }

    override fun onResume() {
        super.onResume()
        logEvent("weiter")
        if (provider == null && !cameraLoading && has(Manifest.permission.CAMERA)) startCamera()
    }

    override fun onPause() {
        logEvent("pausiert")
        super.onPause()
    }

    override fun onStop() {
        logEvent("im Hintergrund")
        super.onStop()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val key = screenKey(newConfig)
        if (key != lastScreen) {
            logEvent("Bildschirm", JSONObject().put("dp", key).put("dichteDpi", newConfig.densityDpi))
            lastScreen = key
        }
        previewBox.layoutParams = previewBox.layoutParams.apply { height = previewHeight(newConfig) }
    }

    // ---------------------------------------------------------------- Kamera

    private fun startCamera() {
        if (provider != null || cameraLoading) return
        cameraLoading = true
        permissionPanel.visibility = View.GONE
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraLoading = false
            try {
                provider = future.get()
                showPreviewOnly()
                setStatus(Phase.IDLE, "Bereit")
            } catch (e: Exception) {
                setStatus(Phase.ERROR, "Kamera ließ sich nicht öffnen: ${e.message ?: e.javaClass.simpleName}")
            }
            render()
        }, ContextCompat.getMainExecutor(this))
    }

    /** Verbindet Vorschau und, falls angegeben, Videoaufnahme mit der Hauptkamera. */
    @OptIn(ExperimentalCamera2Interop::class)
    private fun bind(videoCapture: VideoCapture<Recorder>?): Camera {
        val p = provider ?: throw IllegalStateException("Kamera noch nicht bereit")
        val builder = Preview.Builder().setResolutionSelector(
            ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .build()
        )
        Camera2Interop.Extender(builder).setSessionCaptureCallback(captureCallback)
        val newPreview = builder.build()
        newPreview.setSurfaceProvider(previewView.surfaceProvider)
        p.unbindAll()
        preview = newPreview
        return if (videoCapture == null) {
            p.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, newPreview)
        } else {
            p.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, newPreview, videoCapture)
        }
    }

    private fun showPreviewOnly() {
        try {
            bind(null)
        } catch (e: Exception) {
            setStatus(Phase.ERROR, "Vorschau ließ sich nicht starten: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    private suspend fun waitForStreaming(timeoutMs: Long) {
        withTimeoutOrNull(timeoutMs) {
            suspendCancellableCoroutine<Unit> { cont ->
                val state = previewView.previewStreamState
                val observer = object : Observer<PreviewView.StreamState> {
                    override fun onChanged(value: PreviewView.StreamState) {
                        if (value == PreviewView.StreamState.STREAMING) {
                            state.removeObserver(this)
                            if (cont.isActive) cont.resume(Unit)
                        }
                    }
                }
                state.observe(this@MainActivity, observer)
                cont.invokeOnCancellation { previewView.post { state.removeObserver(observer) } }
            }
        }
    }

    // ---------------------------------------------------------------- Tests

    private fun onMainButton() {
        if (running != null) requestStop() else runQuickTest()
    }

    private fun requestStop() {
        if (running == null) return
        stopRequested = true
        setStatus(Phase.BUSY, "Stoppt …")
    }

    private fun launchTest(block: suspend () -> Unit) {
        if (job?.isActive == true) return
        if (provider == null) {
            setStatus(Phase.ERROR, "Die Kamera ist noch nicht bereit.")
            return
        }
        stopRequested = false
        failed = false
        doneText = null
        backDuringTest.isEnabled = true
        job = lifecycleScope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                failed = true
                report.put(
                    "letzterFehler", JSONObject()
                        .put("zeit", nowIso())
                        .put("test", running ?: "")
                        .put("text", e.toString())
                        .put("stelle", e.stackTrace.take(8).joinToString(" | "))
                )
                setStatus(Phase.ERROR, "Fehler: ${e.message ?: e.javaClass.simpleName}")
            } finally {
                running = null
                counting = false
                hintOverride = null
                backDuringTest.isEnabled = false
                store.save(report)
                showPreviewOnly()
                if (!failed) {
                    val done = doneText
                    if (stopRequested) setStatus(Phase.IDLE, "Gestoppt")
                    else if (done != null) setStatus(Phase.DONE, done)
                }
                render()
            }
        }
    }

    private fun runQuickTest() = launchTest {
        hintOverride = "Richte die Kamera auf eine Szene, gern mit etwas Bewegung. Der Test dauert etwa eine Minute."
        running = "steckbrief"
        render()
        reveal("steckbrief")
        setStatus(Phase.BUSY, "Liest die Kameradaten …")
        val profile = withContext(Dispatchers.Default) {
            CameraReport.build(this@MainActivity).put("encoder", CodecReport.build())
        }
        profile.put("cameraX", cameraXInfo())
        profile.put("anzeige", displayInfo())
        profile.put("speicherFreiGB", r1(filesDir.usableSpace / 1e9))
        profile.put("waerme", DeviceHeat.read(this))
        DeviceHeat.thresholds(this)?.let { profile.put("waermeSchwellen", it) }
        profile.put("version", appVersion)
        profile.put("zeit", nowIso())
        report.put("steckbrief", profile)
        saveAndRender()

        if (stopRequested) return@launchTest
        clip("k30", 30, stabilize = false)
        if (stopRequested) return@launchTest
        clip("k60", 60, stabilize = false)
        if (stopRequested) return@launchTest
        if (stabilizationSupported()) {
            clip("stab", bestFps(), stabilize = true)
        } else {
            putTest("stab", JSONObject().put("ergebnis", "Für Video bietet die Kamera keinen Stabilisator an."))
        }
        doneText = "Schnelltest fertig"
    }

    private suspend fun clip(key: String, fps: Int, stabilize: Boolean) {
        running = key
        render()
        reveal(key)
        putTest(key, record(key, fps, CLIP_SECONDS, stabilize))
    }

    private fun runFoldTest() = launchTest {
        val fps = bestFps()
        hintOverride = if (resources.configuration.smallestScreenWidthDp >= 600) {
            "Klapp das Handy während der Aufnahme zu, warte drei Sekunden und klapp es wieder auf."
        } else {
            "Klapp das Handy während der Aufnahme auf, warte drei Sekunden und klapp es wieder zu."
        }
        running = "klapp"
        render()
        reveal("klapp")
        putTest("klapp", record("klapp", fps, FOLD_SECONDS, stabilize = false))
        doneText = "Klapptest fertig"
    }

    private fun runLongTest() = launchTest {
        val fps = bestFps()
        hintOverride = "Lass die App offen und leg das Handy hin, Kamera nach vorn. Die Aufnahme läuft 10 Minuten."
        running = "dauer"
        render()
        reveal("dauer")
        val log = JSONArray()
        var next = LOG_EVERY_S
        var prevFrames = -1L
        var prevTs = 0L
        var prevBytes = 0L
        var prevT = 0.0
        val result = record("dauer", fps, LONG_SECONDS, stabilize = false) { t, stats ->
            if (t >= next) {
                val (count, first, last) = frames.snapshot()
                val entry = JSONObject().put("s", next.toInt())
                val fromFrames = if (prevFrames < 0) 1L else prevFrames
                val fromTs = if (prevFrames < 0) first else prevTs
                if (last > fromTs) entry.put("fps", r1((count - fromFrames) / ((last - fromTs) / 1e9)))
                if (stats != null && t > prevT) {
                    entry.put("mbits", r1((stats.numBytesRecorded - prevBytes) * 8 / (t - prevT) / 1e6))
                    prevBytes = stats.numBytesRecorded
                }
                val heat = DeviceHeat.read(this@MainActivity)
                entry.put("waerme", heat)
                log.put(entry)
                prevFrames = count
                prevTs = last
                prevT = t
                next += LOG_EVERY_S
                val parts = mutableListOf("Läuft")
                if (heat.has("akkuGrad")) parts.add("Akku ${de(heat.optDouble("akkuGrad"), 1)} °C")
                parts.add("Wärmestufe ${heat.optString("stufe")}")
                if (entry.has("fps")) parts.add("${de(entry.optDouble("fps"), 1)} fps")
                hintOverride = parts.joinToString(" · ")
                hintText.text = hintOverride
            }
        }
        result.put("verlauf", log)
        putTest("dauer", result)
        doneText = "Dauertest fertig"
    }

    /**
     * Nimmt in 4K 16:9 auf, zählt dabei jedes Kamerabild und wertet danach die Datei aus.
     * Die Datei wird anschließend gelöscht; es bleiben nur die Messwerte.
     */
    @OptIn(ExperimentalCamera2Interop::class)
    private suspend fun record(
        key: String,
        fps: Int,
        seconds: Int,
        stabilize: Boolean,
        onTick: (Double, RecordingStats?) -> Unit = { _, _ -> }
    ): JSONObject {
        val out = JSONObject()
        out.put("version", appVersion)
        out.put("zeit", nowIso())
        out.put(
            "angefragt", JSONObject()
                .put("groesse", "UHD 16:9")
                .put("fps", fps)
                .put("sekunden", seconds)
                .put("stabilisator", stabilize)
        )
        val label = "4K · $fps" + if (stabilize) " · Stabilisator" else ""
        setStatus(Phase.BUSY, "Kamera startet · $label")

        val recorder = Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.UHD))
            .setAspectRatio(AspectRatio.RATIO_16_9)
            .build()
        val builder = VideoCapture.Builder(recorder).setTargetFrameRate(Range(fps, fps))
        if (stabilize) builder.setVideoStabilizationEnabled(true)
        val videoCapture = builder.build()

        val camera = try {
            bind(videoCapture)
        } catch (e: Exception) {
            out.put("fehler", "Kamera ließ sich so nicht starten: ${e.message ?: e.javaClass.simpleName}")
            return out
        }
        out.put("kamera", Camera2CameraInfo.from(camera.cameraInfo).cameraId)
        delay(300)
        waitForStreaming(5000)
        delay(1500) // Belichtung und Weißabgleich einpendeln lassen
        val previewSize = preview?.resolutionInfo?.resolution
        if (previewSize != null) out.put("vorschau", "${previewSize.width}x${previewSize.height}")
        if (stopRequested) {
            out.put("ende", "gestoppt")
            return out
        }

        val folder = getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
        val file = File(folder, "cinebo-$key.mp4")
        file.delete()
        val audio = has(Manifest.permission.RECORD_AUDIO)
        out.put("mitTon", audio)
        val heatStart = DeviceHeat.read(this)
        val started = CompletableDeferred<Unit>()
        val finished = CompletableDeferred<VideoRecordEvent.Finalize>()
        var stats: RecordingStats? = null
        recordingSince = 0L
        events = JSONArray()

        var pending = recorder.prepareRecording(this, FileOutputOptions.Builder(file).build())
        if (audio) pending = pending.withAudioEnabled()
        val recording = pending.start(ContextCompat.getMainExecutor(this)) { event ->
            when (event) {
                is VideoRecordEvent.Start -> {
                    frames.reset()
                    counting = true
                    recordingSince = SystemClock.elapsedRealtime()
                    started.complete(Unit)
                }
                is VideoRecordEvent.Status -> stats = event.recordingStats
                is VideoRecordEvent.Finalize -> {
                    counting = false
                    stats = event.recordingStats
                    finished.complete(event)
                }
            }
        }

        var stoppedByUser = false
        try {
            val waitUntil = SystemClock.elapsedRealtime() + 8000
            while (!started.isCompleted && !finished.isCompleted && SystemClock.elapsedRealtime() < waitUntil) {
                delay(50)
            }
            if (started.isCompleted) {
                val totalMs = seconds * 1000L
                recBadge.visibility = View.VISIBLE
                while (!finished.isCompleted) {
                    if (stopRequested) {
                        stoppedByUser = true
                        break
                    }
                    val elapsed = SystemClock.elapsedRealtime() - recordingSince
                    if (elapsed >= totalMs) break
                    val shown = elapsed / 1000
                    recBadge.text = "REC  ${clock(shown)}"
                    setStatus(
                        Phase.REC, "Aufnahme · $label",
                        "${clock(shown)} / ${clock(seconds.toLong())}", elapsed.toFloat() / totalMs
                    )
                    onTick(elapsed / 1000.0, stats)
                    delay(200)
                }
            }
        } finally {
            counting = false
            recBadge.visibility = View.GONE
            if (!finished.isCompleted) recording.stop()
        }

        setStatus(Phase.BUSY, "Wertet die Aufnahme aus …")
        val end = withTimeoutOrNull(30_000) { finished.await() }
        val endText = when {
            end == null -> "keine Rückmeldung vom Recorder"
            end.hasError() -> finalizeText(end.error)
            !started.isCompleted -> "Aufnahme startete nicht"
            stoppedByUser -> "gestoppt"
            else -> "ok"
        }
        out.put("ende", endText)
        if (end != null && end.hasError()) out.put("fehlerCode", end.error)
        val cause = end?.cause
        if (cause != null) out.put("fehlerDetail", cause.toString())
        val last = stats
        if (last != null) {
            out.put("aufnahmeS", r1(last.recordedDurationNanos / 1e9))
            out.put("tonStatus", audioStateName(last.audioStats.audioState))
        }
        out.put("kameraBilder", frames.toJson())
        out.put("waerme", JSONObject().put("start", heatStart).put("ende", DeviceHeat.read(this)))
        val log = events
        events = null
        if (log != null && log.length() > 0) out.put("ereignisse", log)
        if (file.exists() && file.length() > 0) {
            out.put("datei", withContext(Dispatchers.IO) { VideoFileAnalysis.analyze(file, fps) })
        } else {
            out.put("datei", JSONObject().put("fehler", "keine Datei"))
        }
        file.delete()
        return out
    }

    private fun stabilizationSupported(): Boolean {
        val p = provider ?: return false
        val back = CameraSelector.DEFAULT_BACK_CAMERA.filter(p.availableCameraInfos).firstOrNull() ?: return false
        return try {
            Recorder.getVideoCapabilities(back).isStabilizationSupported
        } catch (_: Exception) {
            false
        }
    }

    /** 60 fps nur, wenn der Test sie in echtem 4K 16:9 geschafft hat. Sonst 30. */
    private fun bestFps(): Int {
        val file = readTests().optJSONObject("k60")?.optJSONObject("datei") ?: return 30
        val fps = file.optDouble("fps", 0.0)
        val w = file.optInt("breite")
        val h = file.optInt("hoehe")
        return if (fps >= 57.0 && max(w, h) >= 3840 && min(w, h) >= 2160) 60 else 30
    }

    private fun finalizeText(code: Int): String = when (code) {
        0 -> "ok"
        1 -> "unbekannter Fehler"
        2 -> "Dateigröße am Limit"
        3 -> "Speicher voll"
        4 -> "Kamera wurde geschlossen"
        5 -> "ungültiges Ziel"
        6 -> "Encoder-Fehler"
        7 -> "Recorder-Fehler"
        8 -> "keine Bilddaten"
        9 -> "Zeitlimit erreicht"
        10 -> "Aufnahme verworfen"
        else -> "Fehler $code"
    }

    private fun audioStateName(state: Int): String = when (state) {
        0 -> "aktiv"
        1 -> "aus"
        2 -> "vom System stummgeschaltet"
        3 -> "Encoder-Fehler"
        4 -> "Mikrofon-Fehler"
        5 -> "stumm"
        else -> "Zustand $state"
    }

    // ---------------------------------------------------------------- Steckbrief-Zusätze

    @OptIn(ExperimentalCamera2Interop::class)
    private fun cameraXInfo(): JSONObject {
        val o = JSONObject()
        val p = provider ?: return o.put("fehler", "Kamera nicht bereit")
        val list = JSONArray()
        for (info in p.availableCameraInfos) {
            val c2 = Camera2CameraInfo.from(info)
            val facing = when (c2.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)) {
                CameraCharacteristics.LENS_FACING_BACK -> "hinten"
                CameraCharacteristics.LENS_FACING_FRONT -> "vorne"
                else -> "extern"
            }
            list.put(JSONObject().put("id", c2.cameraId).put("richtung", facing))
        }
        o.put("kameras", list)
        val back = CameraSelector.DEFAULT_BACK_CAMERA.filter(p.availableCameraInfos).firstOrNull()
            ?: return o.put("fehler", "keine Rückkamera")
        o.put("standardKamera", Camera2CameraInfo.from(back).cameraId)
        val zoom = back.zoomState.value
        if (zoom != null) o.put("zoom", "${r2(zoom.minZoomRatio.toDouble())} bis ${r2(zoom.maxZoomRatio.toDouble())}")
        try {
            val caps = Recorder.getVideoCapabilities(back)
            val qualities = JSONObject()
            for (range in caps.supportedDynamicRanges) {
                qualities.put(dynamicRangeName(range), JSONArray(caps.getSupportedQualities(range).map { qualityName(it) }))
            }
            o.put("videoQualitaeten", qualities)
            o.put("videoStabilisator", caps.isStabilizationSupported)
        } catch (e: Exception) {
            o.put("videoFehler", e.toString())
        }
        return o
    }

    private fun dynamicRangeName(range: DynamicRange): String = when (range) {
        DynamicRange.SDR -> "SDR"
        DynamicRange.HLG_10_BIT -> "HLG 10 bit"
        DynamicRange.HDR10_10_BIT -> "HDR10"
        DynamicRange.HDR10_PLUS_10_BIT -> "HDR10+"
        DynamicRange.DOLBY_VISION_10_BIT -> "Dolby Vision 10 bit"
        DynamicRange.DOLBY_VISION_8_BIT -> "Dolby Vision 8 bit"
        else -> "Kodierung ${range.encoding}, ${range.bitDepth} bit"
    }

    private fun qualityName(q: Quality): String = when (q) {
        Quality.UHD -> "UHD"
        Quality.FHD -> "FHD"
        Quality.HD -> "HD"
        Quality.SD -> "SD"
        else -> q.toString()
    }

    private fun displayInfo(): JSONObject {
        val o = JSONObject()
        val c = resources.configuration
        o.put("dp", screenKey(c))
        o.put("dichteDpi", c.densityDpi)
        val d = display
        if (d != null) {
            o.put("hz", JSONArray(d.supportedModes.map { r1(it.refreshRate.toDouble()) }.distinct().sorted()))
            o.put("hzJetzt", r1(d.refreshRate.toDouble()))
        }
        return o
    }

    // ---------------------------------------------------------------- Bericht

    private fun readTests(): JSONObject = report.optJSONObject("tests") ?: JSONObject()

    private fun putTest(key: String, value: JSONObject) {
        val tests = report.optJSONObject("tests") ?: JSONObject().also { report.put("tests", it) }
        tests.put(key, value)
        saveAndRender()
    }

    private fun saveAndRender() {
        store.save(report)
        render()
    }

    private fun quickTestDone(): Boolean {
        val tests = readTests()
        return report.has("steckbrief") && tests.has("k30") && tests.has("k60") && tests.has("stab")
    }

    private fun reportText(): String {
        val o = JSONObject()
        o.put("app", JSONObject().put("name", "Cinebo Test").put("version", appVersion).put("erstellt", nowIso()))
        val keys = report.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            o.put(k, report.get(k))
        }
        return o.toString(1)
    }

    private fun copyReport() {
        val clipboard = getSystemService(ClipboardManager::class.java)
        clipboard.setPrimaryClip(ClipData.newPlainText("Cinebo-Testbericht", reportText()))
        copyButton.text = "Kopiert"
        copyButton.postDelayed({ copyButton.text = "Bericht kopieren" }, 1800)
    }

    private fun shareReport() {
        try {
            val folder = File(cacheDir, "berichte").apply { mkdirs() }
            folder.listFiles()?.forEach { it.delete() }
            val file = File(folder, "cinebo-testbericht-${fileStamp()}.json")
            file.writeText(reportText())
            val uri = FileProvider.getUriForFile(this, "$packageName.dateien", file)
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "Cinebo-Testbericht")
                clipData = ClipData.newRawUri("Cinebo-Testbericht", uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(send, "Bericht teilen"))
        } catch (e: Exception) {
            setStatus(Phase.ERROR, "Teilen ging nicht: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    private fun onReset() {
        if (running != null) return
        if (!resetArmed) {
            resetArmed = true
            resetButton.text = "Wirklich alle Ergebnisse löschen?"
            resetButton.setTextColor(Palette.red)
            resetButton.postDelayed({
                resetArmed = false
                resetButton.setTextColor(Palette.textDim)
                render()
            }, 3500)
            return
        }
        resetArmed = false
        resetButton.setTextColor(Palette.textDim)
        report = JSONObject()
        store.clear()
        setStatus(Phase.IDLE, "Bereit")
        render()
    }

    // ---------------------------------------------------------------- Ergebnistexte

    private fun summary(key: String): Pair<Mark, String> {
        if (running == key) return Mark.RUNNING to "läuft …"
        val data = if (key == "steckbrief") report.optJSONObject("steckbrief") else readTests().optJSONObject(key)
        if (data == null) return Mark.OPEN to DESCRIPTIONS.getValue(key)
        return when (key) {
            "steckbrief" -> profileSummary(data)
            "klapp" -> foldSummary(data)
            "dauer" -> longSummary(data)
            else -> clipSummary(data)
        }
    }

    private fun profileSummary(s: JSONObject): Pair<Mark, String> {
        val cams = s.optJSONArray("kameras") ?: return Mark.FAIL to "Keine Kameradaten"
        val mainId = s.optJSONObject("cameraX")?.optString("standardKamera").orEmpty()
        var main: JSONObject? = null
        var videoBokeh = false
        var vendorBokeh = false
        for (i in 0 until cams.length()) {
            val c = cams.optJSONObject(i) ?: continue
            if (c.optString("id") == mainId) main = c
            val modes = c.optJSONArray("bokehModi")
            if (modes != null) {
                for (j in 0 until modes.length()) {
                    if (modes.optJSONObject(j)?.optString("modus") == "BOKEH_VIDEO") videoBokeh = true
                }
            }
            val vendor = c.optJSONArray("herstellerModi")
            if (vendor != null) {
                for (j in 0 until vendor.length()) {
                    if (vendor.optString(j) == "BOKEH") vendorBokeh = true
                }
            }
        }
        val parts = mutableListOf("${cams.length()} Kameras")
        val maxFps = main?.optJSONObject("maxFpsJeGroesse")?.optString("3840x2160")?.toDoubleOrNull()
        if (maxFps != null) parts.add("4K 16:9 bis ${de(maxFps, 0)} fps")
        parts.add("Video-Bokeh " + if (videoBokeh) "ja" else "nein")
        parts.add("Hersteller-Bokeh " + if (vendorBokeh) "ja" else "nein")
        return Mark.OK to parts.joinToString(" · ")
    }

    private fun clipSummary(r: JSONObject): Pair<Mark, String> {
        val note = r.optString("ergebnis")
        if (note.isNotEmpty()) return Mark.WARN to note
        val error = r.optString("fehler")
        if (error.isNotEmpty()) return Mark.FAIL to error
        val file = r.optJSONObject("datei")
        if (file == null || !file.has("fps")) {
            val reason = endNote(r) ?: file?.optString("fehler")?.ifEmpty { null } ?: "keine auswertbare Aufnahme"
            return Mark.FAIL to reason.replaceFirstChar { it.uppercase() }
        }
        val wanted = r.optJSONObject("angefragt")?.optInt("fps", 30) ?: 30
        val fps = file.optDouble("fps")
        val lost = file.optInt("verloreneBilder")
        val parts = mutableListOf(sizeText(file), "${de(fps, 1)} fps", lostText(lost))
        if (file.has("mbits")) parts.add("${de(file.optDouble("mbits"), 0)} Mbit/s")
        var mark = if (fps < wanted * 0.95 || lost > 0) Mark.WARN else Mark.OK
        if (wanted >= 60 && fps < 57.0) parts.add("60 fps nicht möglich")
        val end = endNote(r)
        if (end != null) {
            parts.add(end)
            mark = Mark.WARN
        }
        return mark to parts.joinToString(" · ")
    }

    private fun foldSummary(r: JSONObject): Pair<Mark, String> {
        val error = r.optString("fehler")
        if (error.isNotEmpty()) return Mark.FAIL to error
        val list = r.optJSONArray("ereignisse") ?: JSONArray()
        var screenChanges = 0
        var background = false
        for (i in 0 until list.length()) {
            when (list.optJSONObject(i)?.optString("art")) {
                "Bildschirm" -> screenChanges++
                "im Hintergrund" -> background = true
            }
        }
        val file = r.optJSONObject("datei")
        val seconds = file?.optDouble("dauerS", 0.0) ?: 0.0
        val end = endNote(r)
        val parts = mutableListOf<String>()
        var mark: Mark
        if (end == null && seconds >= FOLD_SECONDS - 3) {
            parts.add("Aufnahme lief durch")
            mark = if ((file?.optInt("verloreneBilder") ?: 0) > 0) Mark.WARN else Mark.OK
        } else {
            parts.add("Aufnahme endete nach ${de(seconds, 0)} s" + if (end != null) ": $end" else "")
            mark = Mark.FAIL
        }
        if (screenChanges == 0) {
            parts.add("kein Klappen erkannt, bitte wiederholen")
            if (mark == Mark.OK) mark = Mark.WARN
        } else {
            parts.add("$screenChanges× Bildschirm gewechselt")
        }
        if (file != null && file.has("abstandMaxMs")) {
            parts.add("längste Lücke ${de(file.optDouble("abstandMaxMs"), 0)} ms")
        }
        if (background) parts.add("App war kurz im Hintergrund")
        return mark to parts.joinToString(" · ")
    }

    private fun longSummary(r: JSONObject): Pair<Mark, String> {
        val (baseMark, baseText) = clipSummary(r)
        if (baseMark == Mark.FAIL) return baseMark to baseText
        var mark = baseMark
        val parts = mutableListOf<String>()
        val file = r.optJSONObject("datei")
        if (file != null && file.has("dauerS")) parts.add("${clock(file.optDouble("dauerS").toLong())} min")
        parts.add(baseText)
        val log = r.optJSONArray("verlauf") ?: JSONArray()
        var slowest = Double.MAX_VALUE
        var hottest = "keine"
        for (i in 0 until log.length()) {
            val e = log.optJSONObject(i) ?: continue
            if (e.has("fps")) slowest = min(slowest, e.optDouble("fps"))
            val stage = e.optJSONObject("waerme")?.optString("stufe").orEmpty()
            if (DeviceHeat.stageOrder.indexOf(stage) > DeviceHeat.stageOrder.indexOf(hottest)) hottest = stage
        }
        val wanted = r.optJSONObject("angefragt")?.optInt("fps", 30) ?: 30
        if (slowest != Double.MAX_VALUE && slowest < wanted * 0.95) {
            parts.add("langsamste 30 s: ${de(slowest, 1)} fps")
            mark = Mark.WARN
        }
        val heat = r.optJSONObject("waerme")
        val start = heat?.optJSONObject("start")
        val stop = heat?.optJSONObject("ende")
        if (start != null && stop != null) {
            if (start.has("akkuGrad") && stop.has("akkuGrad")) {
                parts.add("Akku ${de(start.optDouble("akkuGrad"), 1)} → ${de(stop.optDouble("akkuGrad"), 1)} °C")
            }
            if (start.has("akkuProzent") && stop.has("akkuProzent")) {
                parts.add("${start.optInt("akkuProzent")} → ${stop.optInt("akkuProzent")} %")
            }
        }
        parts.add("Wärmestufe höchstens $hottest")
        if (DeviceHeat.stageOrder.indexOf(hottest) >= 2) mark = Mark.WARN
        return mark to parts.joinToString(" · ")
    }

    private fun endNote(r: JSONObject): String? = when (val end = r.optString("ende")) {
        "", "ok" -> null
        "gestoppt" -> "vorzeitig gestoppt"
        else -> end
    }

    private fun sizeText(file: JSONObject): String {
        val w = file.optInt("breite")
        val h = file.optInt("hoehe")
        return if (w > 0 && h > 0) "${max(w, h)} × ${min(w, h)}" else "Größe unbekannt"
    }

    private fun lostText(n: Int): String = when (n) {
        0 -> "kein Bild verloren"
        1 -> "1 Bild verloren"
        else -> "$n Bilder verloren"
    }

    private fun errorText(): String {
        val crash = report.optString("letzterAbsturz")
        val text = crash.ifEmpty { report.optJSONObject("letzterFehler")?.optString("text").orEmpty() }
        if (text.isEmpty()) return ""
        val line = text.lines().firstOrNull { it.contains("Exception") || it.contains("Error") } ?: text.lines().first()
        val kind = if (crash.isNotEmpty()) "Absturz" else "Fehler"
        return "$kind: ${line.trim().take(180)} (steht im Bericht)"
    }

    private fun defaultHint(quickDone: Boolean): String {
        val tests = readTests()
        val noMic = if (has(Manifest.permission.RECORD_AUDIO)) "" else " Ohne Mikrofonzugriff laufen die Tests ohne Ton."
        return when {
            !has(Manifest.permission.CAMERA) -> "Die Test-App braucht Zugriff auf Kamera und Mikrofon."
            !quickDone -> "Starte den Schnelltest. Er dauert etwa eine Minute: Steckbrief, 4K mit 30 und 60 fps und der Stabilisator.$noMic"
            !tests.has("klapp") || !tests.has("dauer") ->
                "Jetzt Klapptest und Dauertest, beide mit 4K · ${bestFps()}. Danach den Bericht kopieren und mir in den Chat einfügen.$noMic"
            else -> "Alles gemessen. Kopiere den Bericht und füge ihn bei mir im Chat ein."
        }
    }

    // ---------------------------------------------------------------- Oberfläche

    private fun render() {
        val busy = running != null
        val ready = provider != null
        val quickDone = quickTestDone()
        mainButton.text = when {
            busy -> "Stopp"
            quickDone -> "Schnelltest wiederholen"
            else -> "Schnelltest starten"
        }
        if (busy) mainButton.filled(Palette.red, Color.WHITE) else mainButton.filled(Palette.text, Palette.background)
        mainButton.isEnabled = busy || ready
        foldButton.isEnabled = !busy && ready && quickDone
        longButton.isEnabled = !busy && ready && quickDone
        val hasReport = report.length() > 0
        copyButton.isEnabled = hasReport && !busy
        shareButton.isEnabled = hasReport && !busy
        resetButton.visibility = if (hasReport && !busy) View.VISIBLE else View.GONE
        if (!resetArmed) resetButton.text = "Ergebnisse löschen"
        for ((key, row) in rows) {
            val (mark, text) = summary(key)
            row.show(mark, text)
        }
        val error = errorText()
        errorRow.visibility = if (error.isEmpty()) View.GONE else View.VISIBLE
        errorDivider.visibility = errorRow.visibility
        if (error.isNotEmpty()) errorRow.show(Mark.FAIL, error)
        hintText.text = hintOverride ?: defaultHint(quickDone)
    }

    private fun setStatus(phase: Phase, text: String, time: String? = null, fraction: Float? = null) {
        (statusDot.background as GradientDrawable).setColor(phase.color)
        statusText.text = text
        timeText.text = time.orEmpty()
        timeText.visibility = if (time == null) View.GONE else View.VISIBLE
        progress.visibility = if (fraction == null) View.INVISIBLE else View.VISIBLE
        if (fraction != null) {
            progress.color = phase.color
            progress.fraction = fraction
        }
    }

    private fun showPermissionPanel() {
        permissionPanel.visibility = View.VISIBLE
        setStatus(Phase.ERROR, "Kein Kamerazugriff")
        render()
    }

    private fun askAgain() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            permissionRequest.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
        } else {
            startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null)))
        }
    }

    /** Scrollt die Ergebnisliste so, dass der laufende Test zu sehen ist. */
    private fun reveal(key: String) {
        val row = rows[key] ?: return
        resultScroll.post { resultScroll.smoothScrollTo(0, max(0, resultCard.top + row.top - dp(8f))) }
    }

    private fun buildUi(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Palette.background)
        }

        // Kopfzeile
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val title = label(20f, Palette.text, Fonts.medium).apply {
            text = SpannableStringBuilder("Cinebo")
                .append(" Test", ForegroundColorSpan(Palette.textDim), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        val version = label(13f, Palette.textFaint).apply {
            text = "Version $appVersion"
            fontFeatureSettings = "tnum"
        }
        header.addView(title, lp(0, WRAP, 1f))
        header.addView(version, lp(WRAP, WRAP))
        root.addView(header, lp(MATCH, dp(48f)))

        // Vorschau mit Aufnahme-Anzeige
        previewBox = FrameLayout(this).apply {
            background = GradientDrawable().apply {
                setColor(Color.BLACK)
                cornerRadius = dpf(18f)
            }
            outlineProvider = ViewOutlineProvider.BACKGROUND
            clipToOutline = true
        }
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.BLACK)
        }
        previewBox.addView(previewView, FrameLayout.LayoutParams(MATCH, MATCH))
        recBadge = label(13f, Color.WHITE, Fonts.medium).apply {
            background = GradientDrawable().apply {
                setColor(0x99000000.toInt())
                cornerRadius = dpf(14f)
            }
            setPadding(dp(10f), dp(5f), dp(12f), dp(5f))
            compoundDrawablePadding = dp(7f)
            setCompoundDrawablesRelativeWithIntrinsicBounds(dot(Palette.red, 8f), null, null, null)
            fontFeatureSettings = "tnum"
            gravity = Gravity.CENTER_VERTICAL
            visibility = View.GONE
        }
        previewBox.addView(recBadge, FrameLayout.LayoutParams(WRAP, WRAP).apply {
            gravity = Gravity.TOP or Gravity.START
            setMargins(dp(12f), dp(12f), 0, 0)
        })
        permissionPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(dp(24f), dp(24f), dp(24f), dp(24f))
            visibility = View.GONE
        }
        val permissionText = label(15f, Palette.text).apply {
            text = "Ohne Kamera und Mikrofon kann die Test-App nichts messen."
            gravity = Gravity.CENTER
        }
        val permissionButton = PillButton(this).apply {
            text = "Zugriff erlauben"
            filled(Palette.text, Palette.background)
            setOnClickListener { askAgain() }
        }
        permissionPanel.addView(permissionText, lp(MATCH, WRAP))
        permissionPanel.addView(permissionButton, lp(WRAP, dp(48f)).apply { topMargin = dp(16f) })
        previewBox.addView(permissionPanel, FrameLayout.LayoutParams(MATCH, MATCH))
        root.addView(previewBox, lp(MATCH, previewHeight(resources.configuration)).apply { topMargin = dp(4f) })

        // Status und Fortschritt
        val statusRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        statusDot = View(this).apply { background = dot(Palette.textFaint, 10f) }
        statusText = label(15f, Palette.text, Fonts.medium).apply {
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
        }
        timeText = label(15f, Palette.textDim, Fonts.medium).apply { fontFeatureSettings = "tnum" }
        statusRow.addView(statusDot, lp(dp(10f), dp(10f)).apply { marginEnd = dp(10f) })
        statusRow.addView(statusText, lp(0, WRAP, 1f))
        statusRow.addView(timeText, lp(WRAP, WRAP).apply { marginStart = dp(12f) })
        root.addView(statusRow, lp(MATCH, WRAP).apply { topMargin = dp(16f) })
        progress = ProgressLine(this).apply { visibility = View.INVISIBLE }
        root.addView(progress, lp(MATCH, dp(3f)).apply { topMargin = dp(10f) })

        // Knöpfe
        mainButton = PillButton(this).apply { setOnClickListener { onMainButton() } }
        root.addView(mainButton, lp(MATCH, dp(52f)).apply { topMargin = dp(14f) })
        val pair = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        foldButton = PillButton(this).apply {
            text = "Klapptest · 30 s"
            setOnClickListener { runFoldTest() }
        }
        longButton = PillButton(this).apply {
            text = "Dauertest · 10 min"
            setOnClickListener { runLongTest() }
        }
        pair.addView(foldButton, lp(0, dp(48f), 1f).apply { marginEnd = dp(5f) })
        pair.addView(longButton, lp(0, dp(48f), 1f).apply { marginStart = dp(5f) })
        root.addView(pair, lp(MATCH, WRAP).apply { topMargin = dp(10f) })
        hintText = label(13f, Palette.textDim)
        root.addView(hintText, lp(MATCH, WRAP).apply { topMargin = dp(12f) })

        // Ergebnisse
        resultScroll = ScrollView(this).apply {
            isVerticalScrollBarEnabled = false
            overScrollMode = View.OVER_SCROLL_IF_CONTENT_SCROLLS
        }
        val content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        resultCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = GradientDrawable().apply {
                setColor(Palette.surface)
                cornerRadius = dpf(18f)
            }
        }
        ORDER.forEachIndexed { index, key ->
            if (index > 0) resultCard.addView(divider(), dividerParams())
            val row = ResultRow(this, TITLES.getValue(key))
            rows[key] = row
            resultCard.addView(row, lp(MATCH, WRAP))
        }
        errorDivider = divider()
        resultCard.addView(errorDivider, dividerParams())
        errorRow = ResultRow(this, "Fehler")
        resultCard.addView(errorRow, lp(MATCH, WRAP))
        content.addView(resultCard, lp(MATCH, WRAP))
        resetButton = label(13f, Palette.textDim, Fonts.medium).apply {
            gravity = Gravity.CENTER
            minHeight = dp(44f)
            setOnClickListener { onReset() }
        }
        content.addView(resetButton, lp(MATCH, WRAP).apply { topMargin = dp(4f) })
        resultScroll.addView(content, FrameLayout.LayoutParams(MATCH, WRAP))
        root.addView(resultScroll, lp(MATCH, 0, 1f).apply { topMargin = dp(16f) })

        // Bericht
        val bottom = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        copyButton = PillButton(this).apply {
            text = "Bericht kopieren"
            setOnClickListener { copyReport() }
        }
        shareButton = PillButton(this).apply {
            text = "Teilen"
            setOnClickListener { shareReport() }
        }
        bottom.addView(copyButton, lp(0, dp(48f), 1f).apply { marginEnd = dp(5f) })
        bottom.addView(shareButton, lp(0, dp(48f), 1f).apply { marginStart = dp(5f) })
        root.addView(bottom, lp(MATCH, WRAP).apply { topMargin = dp(10f) })

        val gutter = dp(16f)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.setPadding(bars.left + gutter, bars.top + dp(4f), bars.right + gutter, bars.bottom + dp(12f))
            WindowInsetsCompat.CONSUMED
        }
        return root
    }

    private fun divider(): View = View(this).apply { setBackgroundColor(Palette.line) }

    private fun dividerParams(): LinearLayout.LayoutParams = lp(MATCH, 1).apply { marginStart = dp(50f) }

    private fun lp(width: Int, height: Int, weight: Float = 0f) = LinearLayout.LayoutParams(width, height, weight)

    private fun previewHeight(config: Configuration): Int = dp(max(180f, config.screenHeightDp * 0.30f))

    private fun screenKey(config: Configuration) = "${config.screenWidthDp}x${config.screenHeightDp}"

    private fun has(permission: String) =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    private fun logEvent(kind: String, details: JSONObject = JSONObject()) {
        val list = events ?: return
        val t = if (recordingSince > 0) (SystemClock.elapsedRealtime() - recordingSince) / 1000.0 else -1.0
        list.put(details.put("art", kind).put("s", r1(t)))
    }

    /** Falls die App mitten in einer Aufnahme beendet wurde, liegen hier noch große Dateien. */
    private fun removeOldClips() {
        getExternalFilesDir(Environment.DIRECTORY_MOVIES)?.listFiles()?.forEach {
            if (it.name.endsWith(".mp4")) it.delete()
        }
    }
}
