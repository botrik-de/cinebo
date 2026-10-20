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
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.provider.Settings
import android.text.TextUtils
import android.util.Range
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.ViewOutlineProvider
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
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
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
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
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.Observer
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Test-App, Stufe 2b des Machbarkeitstests: Wie schnell laufen Tiefe (Depth Anything V2) und
 * Verfolgung (EdgeTAM) auf dem KI-Chip in seiner besten Einstellung, mit schnelleren Bausteinen
 * und als Schnellverfolgung, auch während einer 4K-Aufnahme und über 10 Minuten? Alles landet
 * in einem Bericht, den man kopieren oder teilen kann.
 */
class MainActivity : ComponentActivity() {

    private companion object {
        val ORDER = listOf("kichip", "leistung", "tiefe", "verfolgung", "live", "kidauer")

        val TITLES = mapOf(
            "kichip" to "KI-Chip",
            "leistung" to "Leistungsstufen · KI-Chip",
            "tiefe" to "Tiefe · Depth Anything",
            "verfolgung" to "Verfolgung · EdgeTAM",
            "live" to "Live mit 4K · 30 s",
            "kidauer" to "KI-Dauertest · 10 min"
        )

        val DESCRIPTIONS = mapOf(
            "kichip" to "Lädt beide KI-Modelle auf den KI-Chip und misst, wie lange das dauert.",
            "leistung" to "Misst jedes Modell in jeder Leistungsstufe des KI-Chips und nimmt die schnellste.",
            "tiefe" to "Wie schnell und wie genau der KI-Chip die Tiefe eines Bildes schätzt.",
            "verfolgung" to "Wie schnell EdgeTAM ein Objekt Bild für Bild wiederfindet, mit Speicher und als Schnellverfolgung.",
            "live" to "30 Sekunden 4K mit 30 fps, beide Modelle rechnen auf dem Kamerabild mit.",
            "kidauer" to "10 Minuten 4K mit beiden Modellen: Tempo, Wärme und Akku."
        )

        const val LIVE_SECONDS = 30
        const val LONG_SECONDS = 600
        const val LOG_EVERY_S = 30.0
        const val FPS = 30

        /** Ziel aus dem Plan: Die KI schafft 10–15 Durchgänge pro Sekunde (beide Modelle zusammen). */
        const val TARGET_HZ = 10.0

        val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
        val WRAP = ViewGroup.LayoutParams.WRAP_CONTENT
    }

    /** Was der Statuspunkt gerade zeigt. */
    private enum class Phase(val color: Int) {
        IDLE(Palette.textFaint),
        BUSY(Palette.amber),
        LIVE(Palette.green),
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

    private val modelStore by lazy { ModelStore(applicationContext) }
    @Volatile private var modelsOk = false
    /** Läuft das Herunterladen? Eigenes Feld, denn im finally des Jobs gilt der Job selbst noch als aktiv. */
    private var downloading = false
    private var downloadFailure: ModelStore.Failure? = null
    /** Worauf das Herunterladen gerade wartet ([DownloadUi.WAIT_WIFI], [DownloadUi.WAIT_INTERNET]), sonst null. */
    private var downloadWaiting: String? = null
    /** Datenvolumen darf genutzt werden: „Trotzdem jetzt herunterladen“ wurde getippt. */
    @Volatile private var allowMetered = false
    private val remaining = RemainingTime()
    /** Was mit der Kamera nicht stimmt, als Satz für den Hinweis; null, wenn sie läuft. */
    private var cameraProblem: String? = null
    private val engineLazy = lazy { AiEngine(applicationContext) }
    private val engine by engineLazy
    private val pool = FramePool()
    private val prep by lazy { FramePrep() }
    private val analyzerExecutor = Executors.newSingleThreadExecutor { Thread(it, "analyse") }
    @Volatile private var feeding = false
    @Volatile private var prepError: String? = null
    private var analysisSize: String? = null
    private var liveJob: Job? = null
    private var liveError: String? = null
    private var badgeJob: Job? = null
    @Volatile private var liveVisible = false

    private var job: Job? = null
    private var running: String? = null
    private var stopRequested = false
    private var failed = false
    private var doneText: String? = null
    private var hintOverride: String? = null
    private var events: JSONArray? = null
    private var recordingSince = 0L
    private var testStartedAt = 0L
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
    private lateinit var stage: StageLayout
    private lateinit var previewView: PreviewView
    private lateinit var maskOverlay: MaskOverlay
    private lateinit var depthView: DepthView
    private lateinit var recBadge: TextView
    private lateinit var aiBadge: TextView
    private lateinit var permissionPanel: LinearLayout
    private lateinit var statusDot: View
    private lateinit var statusText: TextView
    private lateinit var timeText: TextView
    private lateinit var progress: ProgressLine
    private lateinit var mainButton: PillButton
    private lateinit var previewButton: PillButton
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

    /** Bereitet Kamerabilder für die KI vor, aber nur, wenn sie gerade welche braucht. */
    private val analyzer = ImageAnalysis.Analyzer { image ->
        try {
            if (!feeding) return@Analyzer
            val slot = pool.acquire() ?: return@Analyzer
            try {
                prep.fill(image, slot)
                pool.publish(slot)
            } catch (e: Throwable) {
                pool.giveBack(slot)
                prepError = e.toString().take(300)
            }
        } finally {
            image.close()
        }
    }

    /** Bilder der KI für die Vorschau (kommt vom KI-Thread). */
    private val sink = object : LiveSink {
        override fun onPictures(depth: IntArray, mask: IntArray, present: Boolean) {
            previewBox.post {
                if (!liveVisible) return@post
                depthView.show(depth)
                if (present) maskOverlay.showMask(mask) else maskOverlay.clearMask()
            }
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

        // Eigener Bericht für Stufe 2; der Bericht von Stufe 1 bleibt unberührt liegen.
        store = ReportStore(this, "bericht-stufe2.json")
        report = store.load()
        val crash = CrashLog.take(this)
        if (crash != null) {
            report.put("letzterAbsturz", crash)
            store.save(report)
        }
        removeOldClips()
        modelStore.cleanup()
        modelsOk = modelStore.allReady()

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
        if (!modelsOk) startDownload()
    }

    // ---------------------------------------------------------------- KI-Modelle laden

    /**
     * Lädt die KI-Modelle einmal in den App-Speicher herunter; setzt nach Abbrüchen von selbst
     * fort. Ohne Datenvolumen (WLAN) geht es gleich los; im Mobilnetz wartet es aufs WLAN, bis
     * jemand „Trotzdem jetzt herunterladen“ tippt. Ohne Netz wartet es, bis das Handy online ist.
     * Wie es ausging, steht danach im Bericht, auch ohne Test.
     */
    private fun startDownload() {
        if (modelsOk || downloading) return
        if (!modelStore.hasList) {
            downloadFailure = ModelStore.Failure(ModelStore.Kind.BUILD, "die Liste der KI-Modelle fehlt oder ist ungültig")
            noteDownload(null)
            showIdleStatus()
            render()
            return
        }
        downloading = true
        downloadFailure = null
        downloadWaiting = null
        remaining.reset()
        val startedAt = SystemClock.elapsedRealtime()
        showDownloadProgress(modelStore.doneBytes(), modelStore.totalBytes(), null)
        lifecycleScope.launch {
            var closed = false
            try {
                modelStore.download(gate = { networkGate() }) { done, all, retry ->
                    runOnUiThread { showDownloadProgress(done, all, retry) }
                }
                modelsOk = modelStore.allReady()
                if (!modelsOk) downloadFailure = ModelStore.Failure(ModelStore.Kind.BUILD, "nicht alle KI-Modelle sind vollständig")
            } catch (e: CancellationException) {
                closed = true // Ansicht geschlossen: beim nächsten Start geht es an derselben Stelle weiter
                throw e
            } catch (e: ModelStore.Failure) {
                downloadFailure = e
            } catch (e: Exception) {
                downloadFailure = ModelStore.Failure(ModelStore.Kind.NETWORK, (e.message ?: e.javaClass.simpleName).take(200), e)
            } finally {
                downloading = false
                downloadWaiting = null
                if (!closed) {
                    noteDownload((SystemClock.elapsedRealtime() - startedAt) / 1000.0)
                    if (running == null) {
                        if (modelsOk && provider != null && cameraProblem == null) setStatus(Phase.DONE, "KI-Modelle heruntergeladen, bereit")
                        else showIdleStatus()
                    }
                    render()
                }
            }
        }
        render()
    }

    /** Hält im Bericht fest, wie das Herunterladen ausging, damit sich auch ein Fehler weitergeben lässt. */
    private fun noteDownload(seconds: Double?) {
        val failure = downloadFailure
        val o = JSONObject().put("zeit", nowIso()).put("version", appVersion)
        if (failure == null) {
            o.put("ergebnis", "fertig")
        } else {
            o.put("ergebnis", "unterbrochen")
            o.put("art", when (failure.kind) {
                ModelStore.Kind.NETWORK -> "netz"
                ModelStore.Kind.STORAGE -> "speicher"
                ModelStore.Kind.BUILD -> "bau"
            })
            o.put("meldung", failure.message.orEmpty())
            failure.cause?.let { o.put("ursache", it.toString().take(300)) }
        }
        if (seconds != null) o.put("sekunden", r1(seconds))
        o.put("modelle", modelStore.describe())
        report.put("herunterladen", o)
        store.save(report)
    }

    private fun downloadState() = DownloadUi.State(
        hasList = modelStore.hasList,
        downloading = downloading,
        waiting = downloadWaiting,
        failure = downloadFailure,
        doneBytes = modelStore.doneBytes(),
        totalBytes = modelStore.totalBytes(),
    )

    private fun showDownloadProgress(done: Long, total: Long, retry: ModelStore.Retry?) {
        // Späte Meldungen nach dem Ende des Downloads sind überholt.
        if (modelsOk || !downloading || total <= 0) return
        val waiting = DownloadUi.waitingFrom(retry, allowMetered)
        if (waiting != downloadWaiting) {
            downloadWaiting = waiting
            render() // Knopf und Hinweis hängen davon ab
        }
        if (running != null) return
        val left = if (retry == null) {
            remaining.update(SystemClock.elapsedRealtime(), done, total)
        } else {
            remaining.pause()
            null
        }
        show(DownloadUi.progressLine(done, total, retry, waiting, left))
    }

    private fun show(line: DownloadUi.Line) {
        val phase = when (line.tone) {
            DownloadUi.Tone.IDLE -> Phase.IDLE
            DownloadUi.Tone.BUSY -> Phase.BUSY
            DownloadUi.Tone.ERROR -> Phase.ERROR
            DownloadUi.Tone.DONE -> Phase.DONE
        }
        setStatus(phase, line.text, line.time, line.fraction)
    }

    /**
     * Ob das Herunterladen jetzt eine Verbindung aufbauen darf; sonst, worauf es wartet. Wird vor
     * jeder Verbindung gefragt, also auch, wenn unterwegs das WLAN wegfällt.
     */
    private fun networkGate(): String? = try {
        val cm = applicationContext.getSystemService(ConnectivityManager::class.java)
        val caps = cm?.getNetworkCapabilities(cm.activeNetwork)
        when {
            caps == null || !caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) -> DownloadUi.WAIT_INTERNET
            caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) ||
                caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_TEMPORARILY_NOT_METERED) -> null
            allowMetered -> null
            else -> DownloadUi.WAIT_WIFI
        }
    } catch (_: Exception) {
        null // im Zweifel nicht blockieren
    }

    /** Statuszeile, wenn gerade kein Test läuft und nichts heruntergeladen wird. */
    private fun showIdleStatus() {
        DownloadUi.idleLine(downloadState(), modelsOk, cameraProblem != null, provider != null)?.let { show(it) }
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
        // Ohne sichtbare App filmt die Kamera nicht weiter: Tests und Vorschau halten an,
        // statt mit falschen Werten weiterzulaufen. Geprüft wird nach 1,5 s, weil manche
        // Geräte die App beim Kamerastart ganz kurz stoppen.
        if (running != null) {
            previewView.postDelayed({
                val visible = lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)
                if (!visible && running != null) requestStop()
            }, 1500)
        }
        super.onStop()
    }

    override fun onDestroy() {
        feeding = false
        if (engineLazy.isInitialized()) engine.shutdown()
        analyzerExecutor.shutdown()
        super.onDestroy()
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
                if (running == null && showPreviewOnly()) {
                    cameraProblem = null
                    showIdleStatus()
                }
            } catch (e: Exception) {
                cameraProblem = "Die Kamera ließ sich nicht öffnen (${e.message ?: e.javaClass.simpleName}). " +
                    "Schließ andere Kamera-Apps und starte die Test-App neu."
                setStatus(Phase.ERROR, "Kamera ließ sich nicht öffnen")
            }
            render()
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Verbindet Vorschau, falls angegeben Videoaufnahme, und falls gewünscht Bilder für die KI
     * mit der Hauptkamera. Schafft die Kamera die Bilder für die KI neben 4K nicht in 1280 × 720,
     * versucht sie 640 × 360 und zuletzt ganz ohne; was geklappt hat, steht in [analysisSize].
     */
    @OptIn(ExperimentalCamera2Interop::class)
    private fun bind(videoCapture: VideoCapture<Recorder>?, withAnalysis: Boolean = false): Camera {
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
        analysisSize = null
        if (!withAnalysis) return bindUseCases(p, listOfNotNull(newPreview, videoCapture))
        var lastError: Exception? = null
        for (size in listOf(Size(1280, 720), Size(640, 360))) {
            val analysis = buildAnalysis(size)
            try {
                val camera = bindUseCases(p, listOfNotNull(newPreview, videoCapture, analysis))
                val real = analysis.resolutionInfo?.resolution
                analysisSize = if (real != null) "${real.width}x${real.height}" else "${size.width}x${size.height}"
                return camera
            } catch (e: Exception) {
                lastError = e
                p.unbindAll()
            }
        }
        analysisSize = "keine: ${lastError?.message ?: "unbekannt"}".take(200)
        return bindUseCases(p, listOfNotNull(newPreview, videoCapture))
    }

    private fun bindUseCases(p: ProcessCameraProvider, useCases: List<UseCase>): Camera =
        p.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, *useCases.toTypedArray())

    private fun buildAnalysis(size: Size): ImageAnalysis =
        ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(ResolutionStrategy(size, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                    .build()
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { it.setAnalyzer(analyzerExecutor, analyzer) }

    /** Nur die Vorschau; gibt zurück, ob sie läuft. Ein Fehler steht danach in der Statuszeile. */
    private fun showPreviewOnly(): Boolean = try {
        bind(null)
        cameraProblem = null
        true
    } catch (e: Exception) {
        cameraProblem = "Die Vorschau ließ sich nicht starten (${e.message ?: e.javaClass.simpleName}). Starte die Test-App neu."
        setStatus(Phase.ERROR, "Vorschau ließ sich nicht starten")
        false
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
        if (!modelsOk) {
            if (!downloading) {
                startDownload()
            } else if (downloadWaiting == DownloadUi.WAIT_WIFI) {
                // Gewollt übers Datenvolumen: Die nächste Prüfung lässt die Verbindung zu.
                allowMetered = true
                downloadWaiting = null
                show(DownloadUi.progressLine(modelStore.doneBytes(), modelStore.totalBytes(), ModelStore.Retry(0, DownloadUi.CONNECTING), null, null))
                render()
            }
            return
        }
        if (running == null) {
            runQuickTest()
        } else if (SystemClock.elapsedRealtime() - testStartedAt > 1500) {
            // Ein versehentlicher Doppeltipp beim Starten soll den Test nicht gleich stoppen.
            requestStop()
        }
    }

    private fun requestStop() {
        if (running == null) return
        stopRequested = true
        // Messungen hören nach dem laufenden Schritt auf, die Vorschau sofort.
        engine.requestCancel()
        if (running == "vorschau") engine.requestStop()
        setStatus(Phase.BUSY, "Stoppt …")
    }

    private fun launchTest(block: suspend () -> Unit) {
        if (job?.isActive == true) return
        if (provider == null) {
            setStatus(Phase.ERROR, "Die Kamera ist noch nicht bereit.")
            return
        }
        stopRequested = false
        engine.beginWork()
        failed = false
        doneText = null
        testStartedAt = SystemClock.elapsedRealtime()
        archiveErrors()
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
                val was = running
                if (liveJob != null) stopLive()
                running = null
                counting = false
                hintOverride = null
                backDuringTest.isEnabled = false
                store.save(report)
                // Wurde die Ansicht inzwischen geschlossen, nichts mehr an der Kamera ändern.
                if (lifecycle.currentState.isAtLeast(Lifecycle.State.CREATED)) {
                    val previewOk = showPreviewOnly()
                    if (!failed && previewOk) {
                        val done = doneText
                        when {
                            was == "vorschau" -> setStatus(Phase.IDLE, "Bereit")
                            stopRequested -> setStatus(Phase.IDLE, "Gestoppt")
                            done != null -> setStatus(Phase.DONE, done)
                        }
                    }
                    render()
                }
            }
        }
    }

    /** Fehler aus früheren Läufen bleiben im Bericht, die Fehlerzeile zeigt aber nur neue. */
    private fun archiveErrors() {
        for ((key, archive) in listOf("letzterFehler" to "fruehereFehler", "letzterAbsturz" to "fruehereAbstuerze")) {
            if (!report.has(key)) continue
            val list = report.optJSONArray(archive) ?: JSONArray().also { report.put(archive, it) }
            list.put(report.get(key))
            report.remove(key)
        }
    }

    /**
     * KI-Schnelltest: KI-Chip prüfen und alle Modelle laden (wie in Stufe 2), die Leistungsstufen
     * des Chips vergleichen, in der schnellsten jedes Modell und jede Kette messen, mit der CPU und
     * Metas Original vergleichen und zuletzt 30 Sekunden 4K filmen, während beide mitrechnen.
     */
    private fun runQuickTest() = launchTest {
        hintOverride = "Lass die App offen und richte die Kamera auf eine Szene mit einem Gegenstand in der Mitte. " +
            "Der Test dauert beim ersten Mal etwa zehn Minuten, danach etwa fünf."
        val tests = readTests()
        tests.remove("live")
        report.remove("ki")

        running = "kichip"
        render()
        reveal("kichip")
        setStatus(Phase.BUSY, "Prüft den KI-Chip …")
        val device = withContext(engine.dispatcher) { Npu.info(applicationContext) }
        device.put("waerme", DeviceHeat.read(this))
        DeviceHeat.thresholds(this)?.let { device.put("waermeSchwellen", it) }
        device.put("version", appVersion)
        device.put("modelle", modelStore.describe())
        device.put("zeit", nowIso())
        report.put("geraet", device)
        val ki = JSONObject().put("version", appVersion).put("zeit", nowIso())
        report.put("ki", ki)
        saveAndRender()

        // Frisch laden wie in Stufe 2 (Dauerleistung, Standard-Übersetzung), damit die Ladezeit
        // zählt und vergleichbar bleibt; beim zweiten Mal aus dem Zwischenspeicher.
        withContext(engine.dispatcher) {
            engine.closeAll()
            engine.configureLive(ChipSetting.DEFAULT, ModelId.ATTN, qcMemory = false, metaDecoder = true)
        }
        val load = withContext(engine.dispatcher) {
            engine.load(ModelId.entries) { done, total, _ ->
                runOnUiThread { loadingStatus(done, total) }
            }
        }
        ki.put("laden", load)
        saveAndRender()
        val usable = withContext(engine.dispatcher) { engine.loaded(ModelId.DEPTH) && engine.loaded(ModelId.ENC) }
        if (!usable) throw IllegalStateException("Die Modelle laufen nicht auf dem KI-Chip. Die Gründe stehen im Bericht.")
        if (stopRequested) return@launchTest

        running = "leistung"
        render()
        reveal("leistung")
        val frame = cameraFrame()
        try {
            ki.put("bild", frameInfo(frame))
            val perf = withContext(engine.dispatcher) {
                engine.perfMatrix(frame) { what, done, total ->
                    runOnUiThread { setStatus(Phase.BUSY, "Leistungsstufen · ${min(done + 1, total)} von $total · $what", null, done / total.toFloat()) }
                }
            }
            ki.put("leistung", perf)
            saveAndRender()
            if (stopRequested) return@launchTest
            // In der gewählten Einstellung neu laden; was dort noch nicht übersetzt ist, übersetzt QNN jetzt.
            val reload = withContext(engine.dispatcher) {
                engine.load(ModelId.entries) { done, total, _ -> runOnUiThread { loadingStatus(done, total) } }
            }
            ki.put("ladenGewaehlt", reload)
            saveAndRender()
            if (stopRequested) return@launchTest

            running = "tiefe"
            render()
            reveal("tiefe")
            val bench = withContext(engine.dispatcher) {
                engine.bench(frame) { what -> runOnUiThread { setStatus(Phase.BUSY, "Misst $what …") } }
            }
            ki.put("einzeln", bench)
            saveAndRender()
            if (stopRequested) return@launchTest

            running = "verfolgung"
            render()
            reveal("verfolgung")
            val accuracy = withContext(engine.dispatcher) {
                engine.accuracy(frame) { what -> runOnUiThread { setStatus(Phase.BUSY, "$what …") } }
            }
            ki.put("genauigkeit", accuracy)
            saveAndRender()
            if (stopRequested) return@launchTest
            val video = withContext(engine.dispatcher) {
                engine.videoCheck { what -> runOnUiThread { setStatus(Phase.BUSY, "$what …") } }
            }
            ki.put("pruefvideo", video)
            // Live nimmt den Decoder, der auf dem Chip im Prüfvideo am genauesten traf. Bericht und
            // Anzeige folgen: „jeBild“ ist die volle Kette, die live läuft.
            val decoder = withContext(engine.dispatcher) { engine.chooseDecoderAfterVideo(video) }
            val single = ki.optJSONObject("einzeln")
            if (decoder != null && single != null) {
                single.put("liveDecoder", decoder).put("liveDecoderGrund", "Prüfvideo auf dem Chip")
                val live = single.optJSONObject("ketten")?.optJSONObject(AiEngine.liveChainKey(decoder == "meta"))
                if (live != null) single.put("jeBild", live) else single.remove("jeBild")
            }
            saveAndRender()
        } finally {
            pool.release(frame)
        }
        withContext(engine.dispatcher) { engine.unloadExtras() }
        if (stopRequested) return@launchTest

        running = "live"
        render()
        reveal("live")
        putTest("live", recordWithAi("live", LIVE_SECONDS))
        doneText = "KI-Schnelltest fertig"
    }

    /** Ein vorbereitetes Kamerabild für die KI (ohne Aufnahme); der Aufrufer gibt es mit pool.release zurück. */
    private suspend fun cameraFrame(): FrameSlot {
        setStatus(Phase.BUSY, "Holt ein Kamerabild …")
        bind(null, withAnalysis = true)
        delay(300)
        waitForStreaming(5000)
        delay(1500) // Belichtung und Weißabgleich einpendeln lassen
        return grabFrame() ?: throw IllegalStateException(
            "Die Kamera liefert keine Bilder für die KI (${analysisSize ?: "?"})" +
                (prepError?.let { ". Vorbereitung: $it" } ?: ".")
        )
    }

    private fun runLongTest() = launchTest {
        hintOverride = "Leg das Handy hin, Kamera nach vorn auf eine Szene mit einem Gegenstand in der Mitte. Die App bleibt offen, die Aufnahme läuft 10 Minuten."
        running = "kidauer"
        render()
        reveal("kidauer")
        // Liefert die Kamera neben 4K keine Bilder für die KI, rechnet sie auf einem Standbild. Das
        // entsteht sonst im Schnelltest; nach einem Neustart der App holt der Dauertest es selbst.
        // Klappt das nicht, läuft der Dauertest trotzdem (die KI rechnet dann nur auf Kamerabildern).
        var stillMissing: String? = null
        if (!withContext(engine.dispatcher) { engine.hasStill() }) {
            try {
                val frame = cameraFrame()
                try {
                    withContext(engine.dispatcher) { engine.setStill(frame) }
                } finally {
                    pool.release(frame)
                }
            } catch (e: CancellationException) {
                // Abbruch (auch eine Unterart von IllegalStateException) beendet den Test wie bisher.
                throw e
            } catch (e: IllegalStateException) {
                stillMissing = e.message ?: e.toString()
            }
        }
        val result = recordWithAi("kidauer", LONG_SECONDS)
        stillMissing?.let { result.put("ohneStandbild", it) }
        putTest("kidauer", result)
        doneText = "KI-Dauertest fertig"
    }

    /** KI-Vorschau ohne Aufnahme: Tiefe und Verfolgung live ansehen, Objekt antippen. */
    private fun onPreviewButton() {
        if (running == "vorschau") {
            requestStop()
            return
        }
        if (running != null || !modelsOk) return
        launchTest {
            running = "vorschau"
            hintOverride = "Tippe im Kamerabild auf ein Objekt: Die gelbe Markierung folgt ihm. Rechts die Tiefe, hell ist nah."
            render()
            setStatus(Phase.BUSY, "Bereitet die KI vor …")
            val load = ensureLiveModels()
            if (load != null) report.optJSONObject("ki")?.put("ladenVorschau", load)
            bind(null, withAnalysis = true)
            startLive(allowStill = false)
            setStatus(Phase.LIVE, "KI-Vorschau läuft")
            try {
                while (!stopRequested && liveJob?.isActive == true) delay(200)
            } finally {
                stopLive()
            }
            liveError?.let { throw IllegalStateException(it) }
        }
    }

    /** Lädt die Modelle für den Live-Betrieb, falls nötig. Gibt das Protokoll zurück oder null. */
    private suspend fun ensureLiveModels(): JSONObject? {
        // Leistungstest und Einzelmessung haben Einstellung und Bausteine gewählt; das gilt auch nach einem Neustart.
        val ki = ki()
        val setting = ChipSetting.fromKey(ki?.optJSONObject("leistung")?.optString("gewaehlt"))
        val chosenAttn = ki?.optJSONObject("einzeln")?.optString("liveMemoryAttention").orEmpty()
        val attn = ModelId.entries.firstOrNull { it.key == chosenAttn }
        val qcMemory = ki?.optJSONObject("einzeln")?.optString("liveMemoryEncoder")?.takeIf { it.isNotEmpty() }?.let { it == "qualcomm" }
        val metaDecoder = ki?.optJSONObject("einzeln")?.optString("liveDecoder")?.takeIf { it.isNotEmpty() }?.let { it == "meta" }
        val ready = withContext(engine.dispatcher) {
            engine.configureLive(setting, attn, qcMemory, metaDecoder)
            engine.liveReady()
        }
        if (ready) return null
        val load = withContext(engine.dispatcher) {
            engine.load(engine.liveSet()) { done, total, _ -> runOnUiThread { loadingStatus(done, total) } }
        }
        if (!withContext(engine.dispatcher) { engine.liveReady() }) {
            report.put("letzteLadung", load)
            throw IllegalStateException("Die KI-Modelle ließen sich nicht laden. Die Gründe stehen im Bericht.")
        }
        return load
    }

    private fun loadingStatus(done: Int, total: Int) {
        if (total == 0) return
        val n = min(done + 1, total)
        setStatus(Phase.BUSY, "Lädt die Modelle auf den KI-Chip · $n von $total", null, done / total.toFloat())
    }

    /** Holt ein vorbereitetes Kamerabild; der Aufrufer gibt es mit pool.release zurück. */
    private suspend fun grabFrame(): FrameSlot? {
        prepError = null
        pool.drain()
        feeding = true
        try {
            return withContext(Dispatchers.IO) { pool.take(6000) }
        } finally {
            feeding = false
            pool.drain()
        }
    }

    private fun frameInfo(frame: FrameSlot): JSONObject {
        val o = JSONObject()
        o.put("kameraBild", "${frame.width}x${frame.height}")
        o.put("drehung", frame.rotation)
        o.put("analyse", analysisSize ?: "?")
        o.put("vorbereitungMs", r1(frame.prepMs))
        prepError?.let { o.put("vorbereitungFehler", it) }
        return o
    }

    private fun startLive(allowStill: Boolean) {
        pool.drain()
        prepError = null
        liveError = null
        feeding = true
        engine.beginLive()
        engine.stats.reset()
        pool.resetCounters()
        liveVisible = true
        stage.showDepth = true
        maskOverlay.onTap = { u, v -> engine.track(u, v) }
        liveJob = lifecycleScope.launch(engine.dispatcher) {
            try {
                engine.runLive(pool, sink, allowStill)
            } catch (e: Throwable) {
                liveError = "KI-Schleife: $e".take(400)
            }
        }
        startBadge()
    }

    private suspend fun stopLive() {
        engine.requestStop()
        liveJob?.join()
        liveJob = null
        feeding = false
        pool.drain()
        liveVisible = false
        maskOverlay.onTap = null
        maskOverlay.clearMask()
        depthView.clear()
        stage.showDepth = false
        badgeJob?.cancel()
        badgeJob = null
        aiBadge.visibility = View.GONE
    }

    private fun startBadge() {
        badgeJob?.cancel()
        aiBadge.text = "KI startet …"
        aiBadge.visibility = View.VISIBLE
        badgeJob = lifecycleScope.launch {
            var lastCycles = engine.stats.cycles()
            var lastT = SystemClock.elapsedRealtime()
            while (isActive) {
                delay(1000)
                val cycles = engine.stats.cycles()
                val now = SystemClock.elapsedRealtime()
                if (cycles < lastCycles) lastCycles = 0
                val hz = (cycles - lastCycles) * 1000.0 / max(1L, now - lastT)
                lastCycles = cycles
                lastT = now
                aiBadge.text = if (cycles == 0) "KI startet …" else "KI · ${de(hz, 0)}/s"
            }
        }
    }

    /**
     * 4K mit 30 fps aufnehmen, während beide Modelle auf den Kamerabildern rechnen.
     * Alle 30 Sekunden ein Eintrag: Tempo der KI, Bildrate, Wärme und Akku.
     */
    private suspend fun recordWithAi(key: String, seconds: Int): JSONObject {
        setStatus(Phase.BUSY, "Bereitet die KI vor …")
        val load = ensureLiveModels()
        startLive(allowStill = true)
        val log = JSONArray()
        var next = LOG_EVERY_S
        var prevFrames = -1L
        var prevTs = 0L
        var prevBytes = 0L
        var prevT = 0.0
        // Die KI-Zahlen gelten nur für die Zeit der Aufnahme: Beim Ende der Aufnahme werden sie
        // festgehalten und die KI angehalten, bevor die Datei ausgewertet wird.
        var aiTotal: JSONObject? = null
        var prepared = 0L
        var skipped = 0L
        val result = try {
            record(
                key, FPS, seconds, stabilize = false, withAnalysis = true,
                onStart = {
                    engine.stats.reset()
                    pool.resetCounters()
                },
                onStop = {
                    aiTotal = engine.stats.totalJson()
                    prepared = pool.prepared
                    skipped = pool.skipped
                    engine.requestStop()
                },
                shouldStop = { liveJob?.isActive == false }
            ) { t, stats ->
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
                    val window = engine.stats.closeWindow()
                    entry.put("ki", window)
                    val heat = DeviceHeat.read(this@MainActivity)
                    entry.put("waerme", heat)
                    log.put(entry)
                    prevFrames = count
                    prevTs = last
                    prevT = t
                    next += LOG_EVERY_S
                    if (seconds > LIVE_SECONDS) {
                        val parts = mutableListOf("Läuft")
                        if (window.has("hz")) parts.add("KI ${de(window.optDouble("hz"), 0)}/s")
                        if (entry.has("fps")) parts.add("${de(entry.optDouble("fps"), 1)} fps")
                        if (heat.has("akkuGrad")) parts.add("Akku ${de(heat.optDouble("akkuGrad"), 1)} °C")
                        parts.add("Wärmestufe ${heat.optString("stufe")}")
                        hintOverride = parts.joinToString(" · ")
                        hintText.text = hintOverride
                    }
                }
            }
        } finally {
            stopLive()
        }
        val ai = aiTotal ?: engine.stats.totalJson()
        ai.put("kameraBilderFuerKi", if (aiTotal != null) prepared else pool.prepared)
        ai.put("kameraBilderVerworfen", if (aiTotal != null) skipped else pool.skipped)
        ai.put("analyse", analysisSize ?: "?")
        val (setting, chainName) = withContext(engine.dispatcher) { engine.chip to engine.liveTrackerName() }
        ai.put("einstellung", setting.key)
        ai.put("verfolgungKette", chainName)
        ai.put("memoryAttention", engine.liveAttn.key)
        prepError?.let { ai.put("vorbereitungFehler", it) }
        liveError?.let { ai.put("fehler", it) }
        result.put("ki", ai)
        result.put("verlauf", log)
        if (load != null) result.put("laden", load)
        return result
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
        withAnalysis: Boolean = false,
        recordingHint: String? = null,
        onStart: () -> Unit = {},
        onStop: () -> Unit = {},
        shouldStop: () -> Boolean = { false },
        onTick: (Double, RecordingStats?) -> Unit = { _, _ -> }
    ): JSONObject {
        val out = JSONObject()
        out.put("version", appVersion)
        out.put("zeit", nowIso())
        // Lebenszyklus ab dem ersten Moment mitschreiben, auch während die Kamera startet.
        recordingSince = 0L
        events = JSONArray()
        out.put(
            "angefragt", JSONObject()
                .put("groesse", "UHD 16:9")
                .put("fps", fps)
                .put("sekunden", seconds)
                .put("stabilisator", stabilize)
                .put("kiBilder", withAnalysis)
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
            bind(videoCapture, withAnalysis)
        } catch (e: Exception) {
            out.put("fehler", "Kamera ließ sich so nicht starten: ${e.message ?: e.javaClass.simpleName}")
            return attachEvents(out)
        }
        out.put("kamera", Camera2CameraInfo.from(camera.cameraInfo).cameraId)
        if (withAnalysis) out.put("analyse", analysisSize ?: "?")
        delay(300)
        waitForStreaming(5000)
        delay(1500) // Belichtung und Weißabgleich einpendeln lassen
        val previewSize = preview?.resolutionInfo?.resolution
        if (previewSize != null) out.put("vorschau", "${previewSize.width}x${previewSize.height}")
        if (stopRequested) {
            out.put("ende", "gestoppt")
            return attachEvents(out)
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
        frames.reset() // keine Zahlen aus der vorigen Aufnahme, falls diese nie startet

        val recording = try {
            var pending = recorder.prepareRecording(this, FileOutputOptions.Builder(file).build())
            if (audio) pending = pending.withAudioEnabled()
            pending.start(ContextCompat.getMainExecutor(this)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> {
                        frames.reset()
                        counting = true
                        recordingSince = SystemClock.elapsedRealtime()
                        onStart()
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
        } catch (e: Exception) {
            out.put("fehler", "Aufnahme ließ sich nicht starten: ${e.message ?: e.javaClass.simpleName}")
            return attachEvents(out)
        }

        var stoppedByUser = false
        var aborted = false
        try {
            val waitUntil = SystemClock.elapsedRealtime() + 8000
            while (!started.isCompleted && !finished.isCompleted && SystemClock.elapsedRealtime() < waitUntil) {
                delay(50)
            }
            if (started.isCompleted) {
                val totalMs = seconds * 1000L
                recBadge.visibility = View.VISIBLE
                if (recordingHint != null) {
                    hintOverride = recordingHint
                    hintText.text = recordingHint
                }
                while (!finished.isCompleted) {
                    if (stopRequested) {
                        stoppedByUser = true
                        break
                    }
                    if (shouldStop()) {
                        aborted = true
                        break
                    }
                    val elapsed = SystemClock.elapsedRealtime() - recordingSince
                    if (elapsed >= totalMs) {
                        onTick(elapsed / 1000.0, stats) // letzter Abschnitt, z. B. die Minute 9:30–10:00
                        break
                    }
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
            onStop()
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
            aborted -> "abgebrochen, weil die KI nicht mehr lief"
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
        attachEvents(out)
        if (file.exists() && file.length() > 0) {
            val analysis = try {
                withContext(Dispatchers.IO) { VideoFileAnalysis.analyze(file, fps) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                JSONObject().put("fehler", "Auswertung ging nicht: ${e.message ?: e.javaClass.simpleName}")
            }
            out.put("datei", analysis)
        } else {
            out.put("datei", JSONObject().put("fehler", "keine Datei"))
        }
        file.delete()
        return out
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

    private fun ki(): JSONObject? = report.optJSONObject("ki")

    private fun quickTestDone(): Boolean = ki()?.has("einzeln") == true && readTests().has("live")

    private fun reportText(): String {
        val o = JSONObject()
        o.put("app", JSONObject().put("name", "CineBo Test").put("stufe", "2b").put("version", appVersion).put("erstellt", nowIso()))
        val keys = report.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            o.put(k, report.get(k))
        }
        return o.toString(1)
    }

    private fun copyReport() {
        val clipboard = getSystemService(ClipboardManager::class.java)
        clipboard.setPrimaryClip(ClipData.newPlainText("CineBo-KI-Bericht", reportText()))
        copyButton.text = "Kopiert"
        copyButton.postDelayed({ copyButton.text = "Bericht kopieren" }, 1800)
    }

    private fun shareReport() {
        try {
            val folder = File(cacheDir, "berichte").apply { mkdirs() }
            folder.listFiles()?.forEach { it.delete() }
            val file = File(folder, "cinebo-ki-bericht-${fileStamp()}.json")
            file.writeText(reportText())
            val uri = FileProvider.getUriForFile(this, "$packageName.dateien", file)
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "CineBo-KI-Bericht")
                clipData = ClipData.newRawUri("CineBo-KI-Bericht", uri)
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
        showIdleStatus()
        render()
    }

    // ---------------------------------------------------------------- Ergebnistexte

    private fun summary(key: String): Pair<Mark, String> {
        if (running == key) return Mark.RUNNING to "läuft …"
        return when (key) {
            "kichip" -> chipSummary()
            "leistung" -> perfSummary()
            "tiefe" -> depthSummary()
            "verfolgung" -> trackSummary()
            else -> readTests().optJSONObject(key)?.let { liveSummary(key, it) }
                ?: (Mark.OPEN to DESCRIPTIONS.getValue(key))
        }
    }

    private fun openRow(key: String): Pair<Mark, String> = Mark.OPEN to DESCRIPTIONS.getValue(key)

    private fun chipSummary(): Pair<Mark, String> {
        val device = report.optJSONObject("geraet")
        device?.optString("fehler")?.takeIf { it.isNotEmpty() }?.let {
            return Mark.FAIL to "KI-Chip nicht erreichbar: ${it.take(160)}"
        }
        val load = ki()?.optJSONObject("laden") ?: return openRow("kichip")
        load.optString("fehler").takeIf { it.isNotEmpty() }?.let { return Mark.FAIL to it.take(200) }
        var ok = 0
        var allCached = true
        val errors = mutableListOf<String>()
        val partial = mutableListOf<String>()
        for (id in ModelId.entries) {
            val m = load.optJSONObject(id.key) ?: continue
            if (m.has("fehler")) {
                errors += id.title
                continue
            }
            ok++
            if (!m.optBoolean("ausZwischenspeicher")) allCached = false
            val chip = m.optJSONObject("aufDemChip")
            if (chip != null && chip.optInt("schritte") < chip.optInt("vonSchritten")) {
                partial += "${id.title} (${chip.optInt("schritte")} von ${chip.optInt("vonSchritten")})"
            }
        }
        val parts = mutableListOf<String>()
        device?.optString("qnnSdk")?.takeIf { it.isNotEmpty() }?.let { parts += "QNN $it" }
        val seconds = load.optDouble("gesamtMs", 0.0) / 1000.0
        parts += "$ok Modelle in ${de(seconds, if (seconds < 10) 1 else 0)} s geladen" +
            if (allCached && ok > 0) ", aus dem Zwischenspeicher" else ""
        var mark = Mark.OK
        if (errors.isNotEmpty()) {
            parts += "Fehler bei: ${errors.joinToString()}"
            mark = if (ok == 0) Mark.FAIL else Mark.WARN
        }
        if (partial.isNotEmpty()) {
            parts += "nur teilweise auf dem Chip: ${partial.joinToString()}"
            if (mark == Mark.OK) mark = Mark.WARN
        }
        return mark to parts.joinToString(" · ")
    }

    private fun depthSummary(): Pair<Mark, String> {
        val ki = ki() ?: return openRow("tiefe")
        val measured = ki.optJSONObject("einzeln")?.optJSONObject("modelle")?.optJSONObject(ModelId.DEPTH.key)
            ?: return openRow("tiefe")
        measured.optString("fehler").takeIf { it.isNotEmpty() }?.let { return Mark.FAIL to "Messung gescheitert: ${it.take(160)}" }
        val m = measured.optJSONObject("imModell") ?: return openRow("tiefe")
        val ms = m.optDouble("medianMs")
        val parts = mutableListOf("${de(ms, 0)} ms je Durchgang auf dem KI-Chip, also bis zu ${de(1000.0 / ms, 0)} pro Sekunde")
        var mark = if (ms <= 1000.0 / TARGET_HZ) Mark.OK else Mark.WARN
        val acc = ki.optJSONObject("genauigkeit")?.optJSONObject("tiefe")
        if (acc != null) {
            if (acc.has("cpuMs")) parts += "CPU ${de(acc.optDouble("cpuMs"), 0)} ms"
            val dev = acc.optDouble("abweichungProzent", Double.NaN)
            if (!dev.isNaN()) {
                parts += "weicht ${de(dev, 1)} % von der CPU ab"
                if (dev > 5.0) mark = Mark.WARN
            }
            if (acc.has("ungueltigeWerte")) {
                parts += "${acc.optInt("ungueltigeWerte")} ungültige Werte"
                mark = Mark.FAIL
            }
        }
        return mark to parts.joinToString(" · ")
    }

    /** Welche Leistungsstufe des KI-Chips gewonnen hat und was sie gegenüber Stufe 2 bringt. */
    private fun perfSummary(): Pair<Mark, String> {
        val perf = ki()?.optJSONObject("leistung") ?: return openRow("leistung")
        val rows = perf.optJSONArray("einstellungen") ?: JSONArray()
        fun row(key: String): JSONObject? = (0 until rows.length()).map { rows.getJSONObject(it) }.firstOrNull { it.optString("einstellung") == key }
        val chosenKey = perf.optString("gewaehlt")
        val chosen = row(chosenKey)
        val stopped = perf.optBoolean("abgebrochen")
        if (chosen == null) {
            if (stopped) return Mark.OPEN to "Gestoppt, bevor alle Einstellungen gemessen waren; die bisherige Einstellung bleibt"
            return Mark.WARN to (perf.optString("fehler").ifEmpty { "Keine Einstellung lief vollständig" })
        }
        val base = row(ChipSetting.DEFAULT.key)
        val parts = mutableListOf("Am schnellsten: ${perf.optString("gewaehltTitel")}")
        fun model(r: JSONObject?, id: ModelId) = r?.optJSONObject("modelle")?.optJSONObject(id.key)?.optDouble("medianMs", Double.NaN) ?: Double.NaN
        val depthNow = model(chosen, ModelId.DEPTH)
        val depthBase = model(base, ModelId.DEPTH)
        val encNow = model(chosen, ModelId.ENC)
        val encBase = model(base, ModelId.ENC)
        if (!depthNow.isNaN()) parts += "Tiefe ${de(depthNow, 0)} ms" + if (!depthBase.isNaN() && chosenKey != ChipSetting.DEFAULT.key) " statt ${de(depthBase, 0)}" else ""
        if (!encNow.isNaN()) parts += "Encoder ${de(encNow, 0)} ms" + if (!encBase.isNaN() && chosenKey != ChipSetting.DEFAULT.key) " statt ${de(encBase, 0)}" else ""
        val full = chosen.optDouble("vollMs", Double.NaN)
        val fast = chosen.optDouble("schnellMs", Double.NaN)
        if (!full.isNaN()) parts += "volle Verfolgung mit Tiefe rund ${de(full, 0)} ms"
        if (!fast.isNaN()) parts += "Schnellverfolgung mit Tiefe rund ${de(fast, 0)} ms"
        if (stopped) parts += "Vergleich mit festgehaltenem Takt gestoppt"
        if (perf.has("gesamtS")) parts += "Test ${de(perf.optDouble("gesamtS") / 60.0, 1)} min"
        return Mark.OK to parts.joinToString(" · ")
    }

    private fun trackSummary(): Pair<Mark, String> {
        val ki = ki() ?: return openRow("verfolgung")
        val single = ki.optJSONObject("einzeln") ?: return openRow("verfolgung")
        // Gestoppt: Die Ketten liefen mit einer vorläufigen Wahl, die nicht gilt; live bleibt die bisherige.
        if (single.optBoolean("abgebrochen")) return Mark.OPEN to "Gestoppt, bevor alle Ketten gemessen waren; live bleibt die bisherige Wahl"
        val chains = single.optJSONObject("ketten") ?: return openRow("verfolgung")
        // Nur die volle Verfolgung, die live läuft (mit unserem Decoder wie im Original, sonst Qualcomms).
        val per = single.optJSONObject("jeBild")
        fun median(o: JSONObject?, name: String) = o?.optJSONObject(name)?.optDouble("medianMs", Double.NaN) ?: Double.NaN
        val parts = mutableListOf<String>()
        var best = 0.0
        val both = median(per, "beide")
        val trackMs = median(per, "verfolgung")
        if (!trackMs.isNaN()) {
            // Teile, die nicht gemessen wurden (etwa nach einem Fehler), fehlen in der Klammer.
            val pieces = listOf("Encoder" to "encoder", "Memory-Attention" to "speicherAttention", "Decoder" to "decoder", "Memory-Encoder" to "speicherEncoder")
                .mapNotNull { (title, key) -> median(per, key).takeIf { !it.isNaN() }?.let { "$title ${de(it, 0)}" } }
            parts += "Mit Speicher: ${de(trackMs, 0)} ms je Bild" + if (pieces.isNotEmpty()) " (${pieces.joinToString(" · ")} ms)" else ""
        } else {
            // Keine volle Kette lief durch: Was schiefging, steht bei ihr.
            listOfNotNull(per, chains.optJSONObject("voll"), chains.optJSONObject("vollQualcomm"))
                .firstNotNullOfOrNull { it.optString("fehler").takeIf { e -> e.isNotEmpty() } }
                ?.let { parts += "Verfolgung mit Speicher: Fehler ${it.take(120)}" }
        }
        if (!both.isNaN()) {
            parts += "mit Tiefe ${de(both, 0)} ms, bis zu ${de(1000.0 / both, 0)} pro Sekunde"
            best = max(best, 1000.0 / both)
        }
        val fast = chains.optJSONObject("schnell")
        val fastBoth = median(fast, "beide")
        if (!fastBoth.isNaN()) {
            parts += "Schnellverfolgung mit Tiefe ${de(fastBoth, 0)} ms, bis zu ${de(1000.0 / fastBoth, 0)} pro Sekunde"
            best = max(best, 1000.0 / fastBoth)
        }
        var mark = if (best >= TARGET_HZ) Mark.OK else Mark.WARN
        val video = ki.optJSONObject("pruefvideo")
        fun value(o: JSONObject?, k: String) = o?.optDouble(k, Double.NaN) ?: Double.NaN
        // Je Kette: wie gleich die Maske Metas Original ist, auf dem Chip und (Messlatte) im CPU-Nachbau des Exports.
        fun videoPart(name: String, o: JSONObject?): String? {
            val chip = value(o, "maskeGleichMittel")
            if (chip.isNaN()) return null
            val cpu = value(o, "cpuNachbauMittel")
            return "$name ${(chip * 100).roundToInt()} %" + if (!cpu.isNaN()) " (CPU ${(cpu * 100).roundToInt()} %)" else ""
        }
        val videoParts = listOfNotNull(
            videoPart("voll", video?.optJSONObject("voll")),
            videoPart("mit Qualcomms Decoder", video?.optJSONObject("vollQualcomm")),
            videoPart("schnell", video?.optJSONObject("schnell")),
        )
        val videoStopped = video?.optBoolean("abgebrochen") == true
        if (videoParts.isNotEmpty()) {
            parts += "Prüfvideo, Maske gleich wie in Metas Original: " + videoParts.joinToString(", ") +
                if (videoStopped) " (gestoppt)" else ""
            // Es zählt die volle Kette, die live läuft (nach dem Prüfvideo gewählt). Fehlt ihr Wert,
            // weil das Prüfvideo gestoppt wurde, ist das kein Befund.
            val metaLive = single.optString("liveDecoder") != "qualcomm"
            val liveIou = value(video?.optJSONObject(AiEngine.liveChainKey(metaLive)), "maskeGleichMittel")
            if (liveIou < 0.9 || (liveIou.isNaN() && !videoStopped)) mark = Mark.WARN
        } else if (videoStopped) {
            parts += "Prüfvideo gestoppt"
        } else {
            val acc = ki.optJSONObject("genauigkeit")?.optJSONObject("verfolgung")
            val iou = acc?.optDouble("folgeMaskeGleich", Double.NaN) ?: Double.NaN
            if (!iou.isNaN()) parts += "Maske zu ${(iou * 100).roundToInt()} % wie auf der CPU"
        }
        if (parts.isEmpty()) return openRow("verfolgung")
        return mark to parts.joinToString(" · ")
    }

    private fun liveSummary(key: String, r: JSONObject): Pair<Mark, String> {
        val error = r.optString("fehler")
        if (error.isNotEmpty()) return Mark.FAIL to error
        val ai = r.optJSONObject("ki")
        val aiError = ai?.optString("fehler").orEmpty()
        if (aiError.isNotEmpty()) return Mark.FAIL to aiError
        val file = r.optJSONObject("datei")
        val parts = mutableListOf<String>()
        var mark = Mark.OK
        if (key == "kidauer" && file != null && file.has("dauerS")) parts += "${clock(file.optDouble("dauerS").toLong())} min"
        if (ai != null && ai.has("hz")) {
            val hz = ai.optDouble("hz")
            var text = "KI: ${de(hz, 0)} Durchgänge pro Sekunde"
            val log = r.optJSONArray("verlauf")
            if (key == "kidauer" && log != null && log.length() > 1) {
                var lo = Double.MAX_VALUE
                var hi = 0.0
                for (i in 0 until log.length()) {
                    val w = log.optJSONObject(i)?.optJSONObject("ki")?.optDouble("hz", Double.NaN) ?: Double.NaN
                    if (w.isNaN()) continue
                    lo = min(lo, w)
                    hi = max(hi, w)
                }
                if (lo <= hi) text = "KI: ${de(lo, 0)} bis ${de(hi, 0)} Durchgänge pro Sekunde"
                if (lo < TARGET_HZ) mark = Mark.WARN
            }
            parts += text
            if (hz < TARGET_HZ) mark = Mark.WARN
            if (ai.optInt("aufStandbild") > 0) {
                parts += "ohne Kamerabilder für die KI (Standbild)"
                mark = Mark.WARN
            }
            aspectNote(ai.optString("analyse"))?.let {
                parts += it
                mark = Mark.WARN
            }
        } else {
            parts += "KI lief nicht mit"
            mark = Mark.WARN
        }
        if (file != null && file.has("fps")) {
            val fps = file.optDouble("fps")
            val lost = lostAfterStart(file)
            parts += "${sizeText(file)} mit ${de(fps, 1)} fps"
            parts += lostText(lost)
            if (fps < FPS * 0.95 || lost > 0) mark = Mark.WARN
        } else {
            val reason = endNote(r) ?: file?.optString("fehler")?.ifEmpty { null } ?: "keine auswertbare Aufnahme"
            return Mark.FAIL to reason.replaceFirstChar { it.uppercase() }
        }
        ai?.optInt("objektGefundenProzent", -1)?.takeIf { it >= 0 }?.let { parts += "Objekt in $it % der Durchgänge gefunden" }
        ai?.optJSONObject("ms")?.optJSONObject("alterErgebnis")?.optDouble("median", Double.NaN)?.takeIf { !it.isNaN() }?.let {
            parts += "Ergebnis ${de(it, 0)} ms nach dem Kamerabild fertig"
        }
        if (key == "kidauer") {
            val heat = r.optJSONObject("waerme")
            val start = heat?.optJSONObject("start")
            val stop = heat?.optJSONObject("ende")
            if (start != null && stop != null) {
                if (start.has("akkuGrad") && stop.has("akkuGrad")) {
                    parts += "Akku ${de(start.optDouble("akkuGrad"), 1)} → ${de(stop.optDouble("akkuGrad"), 1)} °C"
                }
                if (start.has("akkuProzent") && stop.has("akkuProzent")) {
                    parts += "${start.optInt("akkuProzent")} → ${stop.optInt("akkuProzent")} %"
                }
            }
            var hottest = "keine"
            val log = r.optJSONArray("verlauf") ?: JSONArray()
            for (i in 0 until log.length()) {
                val stage = log.optJSONObject(i)?.optJSONObject("waerme")?.optString("stufe").orEmpty()
                if (DeviceHeat.stageOrder.indexOf(stage) > DeviceHeat.stageOrder.indexOf(hottest)) hottest = stage
            }
            parts += "Wärmestufe höchstens $hottest"
            if (DeviceHeat.stageOrder.indexOf(hottest) >= 2) mark = Mark.WARN
        }
        endNote(r)?.let {
            parts += it
            mark = Mark.WARN
        }
        return mark to parts.joinToString(" · ")
    }

    /** Liefert die Kamera die KI-Bilder nicht in 16:9, sitzt die Markierung nicht genau auf dem Bild. */
    private fun aspectNote(size: String?): String? {
        val m = Regex("(\\d+)x(\\d+)").find(size ?: return null) ?: return null
        val w = m.groupValues[1].toDouble()
        val h = m.groupValues[2].toDouble()
        if (w <= 0 || h <= 0) return null
        val ratio = max(w, h) / min(w, h)
        return if (kotlin.math.abs(ratio - 16.0 / 9.0) > 0.03) "KI-Bilder nicht in 16:9 ($size), Markierung kann versetzt sein" else null
    }

    private fun endNote(r: JSONObject): String? = when (val end = r.optString("ende")) {
        "", "ok" -> null
        "gestoppt" -> "vorzeitig gestoppt"
        else -> end
    }

    private fun sizeText(file: JSONObject): String {
        val w = file.optInt("breite")
        val h = file.optInt("hoehe")
        return when {
            max(w, h) >= 3840 && min(w, h) >= 2160 -> "4K"
            w > 0 && h > 0 -> "${max(w, h)} × ${min(w, h)}"
            else -> "Größe unbekannt"
        }
    }

    /** Verlorene Bilder ohne die Startlücke. */
    private fun lostAfterStart(file: JSONObject): Int =
        if (file.has("verloreneBilderNachStart")) file.optInt("verloreneBilderNachStart") else file.optInt("verloreneBilder")

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
        val noMic = if (has(Manifest.permission.RECORD_AUDIO)) "" else " Ohne Mikrofonzugriff laufen die Aufnahmen ohne Ton."
        val camera = cameraProblem
        return when {
            camera != null && !modelsOk -> "$camera ${DownloadUi.hint(downloadState())}"
            camera != null -> camera
            !modelsOk -> DownloadUi.hint(downloadState())
            !has(Manifest.permission.CAMERA) -> "Die Test-App braucht Zugriff auf Kamera und Mikrofon."
            !quickDone -> "Starte den KI-Schnelltest. Er lädt beide KI-Modelle auf den KI-Chip, sucht dessen schnellste " +
                "Leistungsstufe, misst Tempo und Genauigkeit und filmt dann 30 Sekunden in 4K, während beide mitrechnen. " +
                "Beim ersten Mal übersetzt der Chip jedes Modell neu; rechne mit etwa zehn Minuten, danach etwa fünf.$noMic"
            repeatReason("live") != null -> repeatReason("live")!!
            !tests.has("kidauer") -> "Jetzt der KI-Dauertest: Handy hinlegen, Kamera auf eine Szene mit einem Gegenstand " +
                "in der Mitte. Vorher kannst du in der KI-Vorschau auf Dinge tippen und zusehen, wie die Markierung folgt.$noMic"
            repeatReason("kidauer") != null -> repeatReason("kidauer")!!
            else -> "Alles gemessen. Kopiere den Bericht und füge ihn bei mir im Chat ein. " +
                "In der KI-Vorschau kannst du Tiefe und Verfolgung weiter ausprobieren."
        }
    }

    /** Warum ein Test noch einmal laufen sollte, oder null. */
    private fun repeatReason(key: String): String? {
        val r = readTests().optJSONObject(key) ?: return null
        if (r.has("fehler")) return null
        return when (r.optString("ende")) {
            "gestoppt" -> if (key == "kidauer") {
                "Der KI-Dauertest wurde vorzeitig gestoppt. Starte ihn bitte noch einmal, er braucht die vollen 10 Minuten."
            } else {
                "Der KI-Schnelltest wurde unterbrochen. Starte ihn bitte noch einmal und lass die App dabei offen."
            }
            "Kamera wurde geschlossen", "Aufnahme startete nicht" ->
                "Die Aufnahme brach ab, weil die Kamera geschlossen wurde. Lief die App die ganze Zeit sichtbar, ist das schon ein Ergebnis; sonst wiederhole den Test."
            else -> null
        }
    }

    // ---------------------------------------------------------------- Oberfläche

    private fun render() {
        val busy = running != null
        val previewing = running == "vorschau"
        val ready = provider != null
        val quickDone = quickTestDone()
        if (!modelsOk) {
            val state = downloadState()
            mainButton.text = DownloadUi.buttonText(state)
            mainButton.filled(Palette.text, Palette.background)
            mainButton.isEnabled = DownloadUi.buttonEnabled(state)
            previewButton.text = "KI-Vorschau"
            previewButton.outlined()
            previewButton.isEnabled = false
            longButton.isEnabled = false
        } else {
            mainButton.text = when {
                busy && !previewing -> "Stopp"
                quickDone -> "KI-Schnelltest wiederholen"
                else -> "KI-Schnelltest starten"
            }
            if (busy && !previewing) mainButton.filled(Palette.red, Color.WHITE) else mainButton.filled(Palette.text, Palette.background)
            mainButton.isEnabled = (busy && !previewing) || (!busy && ready)
            previewButton.text = if (previewing) "Vorschau beenden" else "KI-Vorschau"
            if (previewing) previewButton.filled(Palette.green, Palette.background) else previewButton.outlined()
            previewButton.isEnabled = previewing || (!busy && ready)
            longButton.isEnabled = !busy && ready && quickDone
        }
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
        cameraProblem = "Ohne Zugriff auf Kamera und Mikrofon kann die Test-App nichts messen. Tippe auf „Zugriff erlauben“."
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

    /** Kleine Plakette über dem Bild, z. B. REC oder das Tempo der KI. */
    private fun badge(dotColor: Int?): TextView = label(12f, Color.WHITE, Fonts.medium).apply {
        background = GradientDrawable().apply {
            setColor(0x99000000.toInt())
            cornerRadius = dpf(13f)
        }
        setPadding(dp(if (dotColor != null) 9f else 10f), dp(4f), dp(11f), dp(4f))
        if (dotColor != null) {
            compoundDrawablePadding = dp(6f)
            setCompoundDrawablesRelativeWithIntrinsicBounds(dot(dotColor, 7f), null, null, null)
        }
        fontFeatureSettings = "tnum"
        gravity = Gravity.CENTER_VERTICAL
        maxLines = 1
        ellipsize = TextUtils.TruncateAt.END
    }

    private fun panel(): FrameLayout = FrameLayout(this).apply {
        background = GradientDrawable().apply {
            setColor(Color.BLACK)
            cornerRadius = dpf(16f)
        }
        outlineProvider = ViewOutlineProvider.BACKGROUND
        clipToOutline = true
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
        val wordmark = ImageView(this).apply {
            setImageResource(R.drawable.wordmark)
            contentDescription = "CineBo"
            adjustViewBounds = true
        }
        val stageBadge = label(11f, Palette.textDim, Fonts.medium).apply {
            text = "TEST · KI"
            letterSpacing = 0.08f
            includeFontPadding = false
            setLineSpacing(0f, 1f)
            setPadding(dp(7f), dp(3f), dp(7f), dp(3f))
            background = GradientDrawable().apply {
                cornerRadius = dpf(5f)
                setStroke(dp(1f), Palette.line)
            }
        }
        val version = label(13f, Palette.textFaint).apply {
            text = "Version $appVersion"
            fontFeatureSettings = "tnum"
        }
        header.addView(wordmark, lp(WRAP, dp(20f)))
        header.addView(stageBadge, lp(WRAP, WRAP).apply { marginStart = dp(10f) })
        header.addView(View(this), lp(0, 1, 1f))
        header.addView(version, lp(WRAP, WRAP))
        root.addView(header, lp(MATCH, dp(48f)))

        // Bühne: Kamerabild mit Maske, daneben die Tiefe
        previewBox = FrameLayout(this)
        val cameraPanel = panel()
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.BLACK)
        }
        cameraPanel.addView(previewView, FrameLayout.LayoutParams(MATCH, MATCH))
        maskOverlay = MaskOverlay(this)
        cameraPanel.addView(maskOverlay, FrameLayout.LayoutParams(MATCH, MATCH))
        recBadge = badge(Palette.red).apply { visibility = View.GONE }
        cameraPanel.addView(recBadge, FrameLayout.LayoutParams(WRAP, WRAP).apply {
            gravity = Gravity.TOP or Gravity.START
            setMargins(dp(10f), dp(10f), dp(10f), 0)
        })
        aiBadge = badge(Palette.green).apply { visibility = View.GONE }
        cameraPanel.addView(aiBadge, FrameLayout.LayoutParams(WRAP, WRAP).apply {
            gravity = Gravity.BOTTOM or Gravity.START
            setMargins(dp(10f), 0, dp(10f), dp(10f))
        })
        val depthPanel = panel()
        depthView = DepthView(this)
        depthPanel.addView(depthView, FrameLayout.LayoutParams(MATCH, MATCH))
        val depthBadge = badge(null).apply { text = "Tiefe · hell ist nah" }
        depthPanel.addView(depthBadge, FrameLayout.LayoutParams(WRAP, WRAP).apply {
            gravity = Gravity.TOP or Gravity.START
            setMargins(dp(10f), dp(10f), dp(10f), 0)
        })
        stage = StageLayout(this, cameraPanel, depthPanel)
        previewBox.addView(stage, FrameLayout.LayoutParams(MATCH, MATCH))
        permissionPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(dp(24f), dp(24f), dp(24f), dp(24f))
            background = GradientDrawable().apply {
                setColor(Palette.surface)
                cornerRadius = dpf(16f)
            }
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
        previewButton = PillButton(this).apply {
            text = "KI-Vorschau"
            setOnClickListener { onPreviewButton() }
        }
        longButton = PillButton(this).apply {
            text = "KI-Dauertest · 10 min"
            setOnClickListener { runLongTest() }
        }
        pair.addView(previewButton, lp(0, dp(48f), 1f).apply { marginEnd = dp(5f) })
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

    private fun previewHeight(config: Configuration): Int = dp(max(200f, config.screenHeightDp * 0.36f))

    private fun screenKey(config: Configuration) = "${config.screenWidthDp}x${config.screenHeightDp}"

    private fun has(permission: String) =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    /** Hängt die mitgeschriebenen Ereignisse an das Ergebnis und beendet das Mitschreiben. */
    private fun attachEvents(out: JSONObject): JSONObject {
        val log = events
        events = null
        if (log != null && log.length() > 0) out.put("ereignisse", log)
        return out
    }

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
