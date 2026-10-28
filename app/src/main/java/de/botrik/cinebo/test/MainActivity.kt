package de.botrik.cinebo.test

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Intent
import android.content.IntentFilter
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
import android.os.BatteryManager
import android.os.Bundle
import android.os.Environment
import android.os.PowerManager
import android.os.SystemClock
import android.provider.MediaStore
import android.provider.Settings
import android.text.TextUtils
import android.util.Range
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
import androidx.camera.core.CameraEffect
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
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
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import kotlin.coroutines.resume
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Test-App, Stufe 3 des Machbarkeitstests: Die Grafik-Pipeline sitzt als CameraX-Effekt vor
 * Vorschau und Videoaufnahme. Sie gibt der KI echte Kamerabilder während der 4K-Aufnahme,
 * legt die Maske der Verfolgung auf genau das Bild, zu dem sie gehört, und macht den
 * Hintergrund unscharf; eine Wärmeregelung nimmt die KI bei Hitze zurück. Clips mit und ohne
 * Unschärfe landen in der Galerie, alle Messwerte in einem Bericht zum Kopieren oder Teilen.
 */
class MainActivity : ComponentActivity() {

    private companion object {
        val ORDER = listOf("kichip", "leistung", "tiefe", "verfolgung", "live", "vergleich", "strom", "kidauer")

        val TITLES = mapOf(
            "kichip" to "KI-Chip",
            "leistung" to "Leistungsstufen · KI-Chip",
            "tiefe" to "Tiefe · Depth Anything",
            "verfolgung" to "Verfolgung · EdgeTAM",
            "live" to "4K mit Unschärfe · 30 s",
            "vergleich" to "Vergleichsclip · 10 s",
            "strom" to "Strombedarf · gut 4 min",
            "kidauer" to "KI-Dauertest · 10 min"
        )

        val DESCRIPTIONS = mapOf(
            "kichip" to "Lädt beide KI-Modelle auf den KI-Chip und misst, wie lange das dauert.",
            "leistung" to "Misst jedes Modell in jeder Leistungsstufe des KI-Chips und nimmt die schnellste.",
            "tiefe" to "Wie schnell und wie genau der KI-Chip die Tiefe eines Bildes schätzt.",
            "verfolgung" to "Wie schnell EdgeTAM ein Motiv Bild für Bild wiederfindet, mit Speicher und als Schnellverfolgung.",
            "live" to "30 Sekunden 4K mit 30 fps: Die KI verfolgt das Motiv, der Hintergrund wird unscharf. Der Clip kommt in die Galerie.",
            "vergleich" to "10 Sekunden 4K ohne KI und Grafik-Pipeline, zum Vergleich mit dem Unschärfe-Clip und mit der OPPO-Kamera. Der Clip kommt in die Galerie.",
            "strom" to "Teil des KI-Dauertests: misst in Watt, was Bildschirm, Kamera, 4K-Aufnahme, Unschärfe, KI-Bilder, KI und Tiefe einzeln verbrauchen. Daran liegt die Wärme.",
            "kidauer" to "Ohne Zutun, etwa 25 bis 40 Minuten: abkühlen, Strombedarf messen, wieder abkühlen, dann 10 Minuten 4K mit Unschärfe und Wärmeregelung. Tempo, Wärme, Strom und Akku."
        )

        const val LIVE_SECONDS = 30
        const val COMPARE_SECONDS = 10
        const val LONG_SECONDS = 600
        const val LOG_EVERY_S = 30.0
        const val FPS = 30

        /** Strombedarf: einschwingen nach neuer Kamera-Bindung, nach einem Wechsel der KI, dann messen (in zwei Hälften). */
        const val POWER_SETTLE_BIND_MS = 8_000L
        const val POWER_SETTLE_SWITCH_MS = 5_000L
        const val POWER_HALF_MS = 10_000L

        /** Vor dem Dauertest höchstens so lange abkühlen lassen (Sekunden); kühl heißt 30 s am Stück. */
        const val COOL_MAX_S = 900L
        const val COOL_HOLD_MS = 30_000L

        /** Countdown vor der Aufnahme des Dauertests. */
        const val COUNTDOWN_MS = 10_000L

        /** Bildschirmhelligkeit in Tests (fest, damit der Strom vergleichbar bleibt) und beim Abkühlen. */
        const val TEST_BRIGHTNESS = 0.5f
        const val COOL_BRIGHTNESS = 0.05f

        /** Strombedarf: Abstand der KI-Durchgänge in ms (5 pro Sekunde wie live ohne Hitze). */
        const val POWER_AI_INTERVAL_MS = 200L

        /** Abschnitte des Strombedarfs der Reihe nach, mit ihrem Titel (siehe measurePower). */
        val POWER_PHASES = linkedMapOf(
            "aus" to "Kamera aus",
            "kamera" to "Kamera an",
            "aufnahme" to "4K-Aufnahme",
            "unschaerfe" to "Unschärfe",
            "kibilder" to "KI-Bilder",
            "ki" to "KI",
            "kiTiefe" to "KI mit Tiefe",
            "spitze" to "KI auf Spitzenleistung",
            "aufnahme2" to "4K-Aufnahme noch einmal",
        )

        /** So oft (Sekunden) steht der Stand für den Beendigungsgrund neu bereit (siehe ExitHistory.setState). */
        const val STATE_EVERY_S = 10.0

        /** Anfang von „ende“, wenn die App einen Test wegen Hitze selbst beendet hat. */
        const val HEAT_STOP = "Wegen Hitze selbst beendet"

        /** Mindestens so viel Akku (%) für den Dauertest: Er braucht samt Abkühlen etwa 15 %. */
        const val MIN_BATTERY_PERCENT = 40

        /** Über so viele Sekunden am Ende des Dauertests zeigt die Zusammenfassung den Trend. */
        const val TREND_S = 180.0

        /** „ende“ eines Tests, solange er läuft (steht so im gespeicherten Zwischenstand). */
        const val RUNNING = "läuft"

        /** „ende“ eines Tests, den das System mittendrin beendet hat (beim nächsten Start gesetzt). */
        const val KILLED = "App wurde beendet"

        /** Album der Clips in der Galerie. */
        const val GALLERY_FOLDER = "CineBo"

        /** Ziel aus dem Plan für die Einzelmessung: Die KI schafft 10–15 Durchgänge pro Sekunde (beide Modelle zusammen). */
        const val TARGET_HZ = 10.0

        /**
         * Darunter liegen die Masken der KI mehr als 0,2 s auseinander und die Unschärfe hinkt bei
         * Bewegung sichtbar; die Wärmeregelung darf so weit bremsen, das steht dann gelb da.
         */
        const val MIN_LIVE_HZ = 5.0

        /** Findet die KI das Motiv seltener, wurde der Hintergrund zu selten unscharf. */
        const val MIN_FOUND_PERCENT = 80

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
    /** Die Grafik-Pipeline der aktuellen Bindung (null: nur Vorschau). */
    private var pipeline: BokehProcessor? = null
    /** Für ein einzelnes Kamerabild ohne laufende KI: die Pipeline liefert, solange [on] gilt. */
    private val grabDemand = object : AiDemand {
        @Volatile var on = false
        override val nextTakeNs: Long get() = 0L
        override val waiting: Boolean get() = on
        override val depthNeeded: Boolean get() = true
    }
    private var liveJob: Job? = null
    private var liveError: String? = null
    private var badgeJob: Job? = null

    private var job: Job? = null
    private var running: String? = null
    private var stopRequested = false
    private var failed = false
    private var doneText: String? = null
    /** Wie die Statuszeile das Ende zeigt: grün, oder rot, wenn eine Aufnahme nicht geklappt hat. */
    private var donePhase = Phase.DONE
    private var hintOverride: String? = null
    /** Hinweis nach einem Test, bis zum nächsten (etwa: Ladekabel abziehen); sonst null. */
    private var hintAfter: String? = null
    /** Warum die KI-Vorschau von selbst endete (etwa Hitze); null = normal. */
    private var stopNote: String? = null
    /** Der Dauertest wartet aufs Abkühlen: Die App darf dabei in den Hintergrund. */
    @Volatile private var cooling = false
    /** Warum die App einen Test wegen Hitze selbst beendet (Text für „ende“), sonst null. */
    private var heatStop: String? = null
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
    private lateinit var tapOverlay: TapOverlay
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

        // Eigener Bericht für Stufe 3; die Berichte früherer Stufen bleiben unberührt liegen.
        store = ReportStore(this, "bericht-stufe3.json")
        report = store.load()
        val crash = CrashLog.take(this)
        if (crash != null) report.put("letzterAbsturz", crash)
        // Warum Android die App zuletzt beendet hat; ein Test, der noch auf „läuft“ steht, wurde
        // mittendrin beendet und bekommt jetzt den passenden Grund.
        val exits = ExitHistory.read(this)
        closeUnfinishedTests(exits)
        startNewVersion()
        report.put("appVersion", appVersion)
        report.put("beendigungen", exits)
        store.save(report)
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
                // Beim Abkühlen läuft keine Kamera: Der Dauertest wartet, bis die App wieder sichtbar ist.
                if (!visible && running != null && !cooling) requestStop()
            }, 1500)
        }
        super.onStop()
    }

    override fun onDestroy() {
        grabDemand.on = false
        // CameraX löst die Kamera mit dem Lebenszyklus; die Pipeline wartet, bis CameraX sie freigibt.
        pipeline?.release()
        pipeline = null
        if (engineLazy.isInitialized()) engine.shutdown()
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

    /** Helligkeit nur dieses Fensters (0 bis 1; BRIGHTNESS_OVERRIDE_NONE = wie eingestellt). */
    private fun setBrightness(value: Float) {
        window.attributes = window.attributes.apply { screenBrightness = value }
    }

    /** Gewünschte Bildwiederholrate dieses Fensters (0 = wie das System will). */
    private fun setRefreshRate(hz: Float) {
        window.attributes = window.attributes.apply { preferredRefreshRate = hz }
    }

    /** Was der Bildschirm gerade tut: Größe (auf- oder zugeklappt), Bildwiederholrate, Helligkeit dieses Fensters. */
    private fun displayInfo(): JSONObject {
        val o = JSONObject()
        val config = resources.configuration
        o.put("dp", "${config.screenWidthDp}x${config.screenHeightDp}")
        try {
            o.put("hz", r1(display?.refreshRate?.toDouble() ?: Double.NaN))
        } catch (_: Exception) {
        }
        o.put("helligkeitFenster", window.attributes.screenBrightness.toDouble())
        try {
            o.put("helligkeitSystem", Settings.System.getInt(contentResolver, Settings.System.SCREEN_BRIGHTNESS))
        } catch (_: Exception) {
        }
        return o
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
     * Verbindet Vorschau und, falls angegeben, Videoaufnahme mit der Hauptkamera. Mit [effect]
     * sitzt die Grafik-Pipeline als CameraX-Effekt davor: CameraX teilt dann einen Kamerastrom,
     * schickt ihn durch die Pipeline und kopiert ihre Ausgabe in Vorschau und Video. Die Pipeline
     * der vorigen Bindung gibt diese Funktion frei, sobald CameraX sie nicht mehr braucht.
     */
    @OptIn(ExperimentalCamera2Interop::class)
    private fun bind(videoCapture: VideoCapture<Recorder>?, effect: BokehProcessor? = null): Camera {
        val p = provider ?: throw IllegalStateException("Kamera noch nicht bereit")
        // Streng 16:9: Sonst wählt CameraX 1.4 für den geteilten Strom ein 4:3-Sensorformat, das
        // Video wäre ein hochgerechneter Ausschnitt und die Vorschau zeigte mehr als das Video.
        val builder = Preview.Builder().setResolutionSelector(
            ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy(AspectRatio.RATIO_16_9, AspectRatioStrategy.FALLBACK_RULE_NONE))
                .build()
        )
        val interop = Camera2Interop.Extender(builder).setSessionCaptureCallback(captureCallback)
        if (videoCapture != null || effect != null) {
            // Geteilter Strom: CameraX 1.4 übernimmt die Bildrate der Videoaufnahme dann nicht.
            interop.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(FPS, FPS))
        }
        val newPreview = builder.build()
        newPreview.setSurfaceProvider(previewView.surfaceProvider)
        p.unbindAll()
        pipeline?.release()
        pipeline = effect
        preview = newPreview
        val group = UseCaseGroup.Builder().addUseCase(newPreview)
        if (videoCapture != null) group.addUseCase(videoCapture)
        if (effect != null) {
            val targets = CameraEffect.PREVIEW or if (videoCapture != null) CameraEffect.VIDEO_CAPTURE else 0
            group.addEffect(PipelineEffect(targets, effect))
        }
        return p.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, group.build())
    }

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
        heatStop = null
        hintAfter = null
        engine.beginWork()
        failed = false
        stopNote = null
        doneText = null
        donePhase = Phase.DONE
        testStartedAt = SystemClock.elapsedRealtime()
        archiveErrors()
        backDuringTest.isEnabled = true
        // Feste Helligkeit und 60 Hz: Der Bildschirm verbraucht dann in jedem Test gleich viel.
        setBrightness(TEST_BRIGHTNESS)
        setRefreshRate(60f)
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
                // Auch wenn die Ansicht geschlossen wurde: KI-Schleife sauber beenden und den Bericht sichern.
                if (liveJob != null) withContext(NonCancellable) { stopLive() }
                running = null
                counting = false
                hintOverride = null
                cooling = false
                backDuringTest.isEnabled = false
                setBrightness(WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE)
                setRefreshRate(0f)
                store.save(report)
                // Wurde die Ansicht inzwischen geschlossen, nichts mehr an der Kamera ändern.
                if (lifecycle.currentState.isAtLeast(Lifecycle.State.CREATED)) {
                    val previewOk = showPreviewOnly()
                    if (!failed && previewOk) {
                        val done = doneText
                        when {
                            was == "vorschau" -> stopNote?.let { setStatus(Phase.ERROR, it) } ?: setStatus(Phase.IDLE, "Bereit")
                            stopRequested -> setStatus(Phase.IDLE, "Gestoppt")
                            done != null -> setStatus(donePhase, done)
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
     * Metas Original vergleichen, dann 30 Sekunden 4K mit Unschärfe filmen, während beide Modelle
     * mitrechnen, und zuletzt 10 Sekunden ohne Unschärfe zum Vergleich.
     */
    private fun runQuickTest() = launchTest {
        hintOverride = "Stell das Handy fest hin und richte die Kamera auf eine Person oder einen Gegenstand in der Mitte, " +
            "mit etwas Abstand zum Hintergrund; so zeigen beide Clips dieselbe Szene. Lass die App offen und das Ladekabel ab. " +
            "Der Test dauert etwa acht Minuten."
        val tests = readTests()
        tests.remove("live")
        tests.remove("vergleich")
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
        val (frame, frameInfo) = cameraFrame()
        try {
            ki.put("bild", frameInfo)
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
        putTest("live", recordWithAi("live", LIVE_SECONDS, keepClip = true))
        if (stopRequested) return@launchTest

        running = "vergleich"
        render()
        reveal("vergleich")
        putTest("vergleich", recordComparison())
        markDone("KI-Schnelltest fertig", "live", "vergleich")
    }

    /** Text für das Ende eines Tests; rot, wenn eine seiner Aufnahmen nicht geklappt hat oder die App ihn wegen Hitze beendet hat. */
    private fun markDone(text: String, vararg keys: String) {
        val tests = readTests()
        val heat = keys.any { tests.optJSONObject(it)?.optString("ende")?.startsWith(HEAT_STOP) == true }
        val failedRows = keys.any { key -> tests.optJSONObject(key)?.let { recordingSummary(key, it).first } == Mark.FAIL }
        doneText = when {
            heat -> "Wegen Hitze vorzeitig beendet"
            failedRows -> "$text · mit Fehlern"
            else -> text
        }
        donePhase = if (heat || failedRows) Phase.ERROR else Phase.DONE
    }

    /**
     * Ein vorbereitetes Kamerabild für die KI (ohne Aufnahme), aus der Grafik-Pipeline wie live,
     * samt Angaben für den Bericht. Danach läuft wieder die einfache Vorschau, damit die Pipeline
     * die Messungen nicht mit Grafikarbeit stört. Der Aufrufer gibt das Bild mit pool.release zurück.
     */
    private suspend fun cameraFrame(): Pair<FrameSlot, JSONObject> {
        setStatus(Phase.BUSY, "Holt ein Kamerabild …")
        val effect = BokehProcessor(applicationContext, strength = 0f, aiPool = pool, aiDemand = grabDemand, thumbnailAfter = 0)
        bind(null, effect)
        delay(300)
        waitForStreaming(5000)
        delay(1500) // Belichtung und Weißabgleich einpendeln lassen
        val frame = grabFrame()
        if (frame == null) {
            // Alles, was die Pipeline weiß, in den Bericht; die Meldung bekommt nur den ersten Hinweis.
            ki()?.put("bild", JSONObject().put("fehler", "keine Kamerabilder für die KI").put("pipeline", effect.stats.toJson()))
            throw IllegalStateException("Die Grafik-Pipeline liefert keine Kamerabilder für die KI" + pipelineNote(effect))
        }
        val info = frameInfo(frame, effect)
        showPreviewOnly()
        return frame to info
    }

    /** Was die Pipeline über sich weiß, als Satzende für eine Fehlermeldung. */
    private fun pipelineNote(effect: BokehProcessor): String {
        val o = effect.stats.toJson()
        val notes = o.optJSONArray("hinweise")
        val first = if (notes != null && notes.length() > 0) notes.optString(0) else null
        return when {
            first != null -> ": ${first.take(200)}"
            o.optInt("bilderEin") == 0 -> ": Die Kamera gab ihr kein Bild."
            else -> "."
        }
    }

    /**
     * KI-Dauertest, ganz ohne Zutun: abkühlen, Strombedarf messen (gut 4 min), wieder abkühlen,
     * dann 10 Minuten 4K mit KI. Jede Phase steht sofort im gespeicherten Bericht („phase“), damit
     * ein Rauswurf durch das System zeigt, wo er geschah. Vorher: kein Ladekabel, Akku ab 40 %,
     * Energiesparmodus aus. Wird es zu heiß, beendet die App den Test selbst.
     */
    private fun runLongTest() = launchTest {
        running = "kidauer"
        render()
        reveal("kidauer")
        precheckLong()?.let { (status, hint) ->
            doneText = status
            donePhase = Phase.ERROR
            hintAfter = hint
            return@launchTest
        }
        val entry = JSONObject().put("version", appVersion).put("zeit", nowIso()).put("ende", RUNNING)
            .put("phase", "abkuehlen1").put("startMs", System.currentTimeMillis()).put("pid", android.os.Process.myPid())
            .put("anzeige", displayInfo())
        putTestQuiet("kidauer", entry)
        guarded("kidauer", entry) {
            entry.put("abkuehlen1", waitUntilCool(entry))
            if (stopRequested) return@guarded endLong(entry, "gestoppt")

            setPhase(entry, "strom")
            running = "strom"
            render()
            reveal("strom")
            hintOverride = "Misst jetzt gut 4 Minuten den Strombedarf: Bildschirm, Kamera, 4K-Aufnahme, Unschärfe, KI und Tiefe einzeln. " +
                "Bitte nichts anfassen. Die Aufnahmen dabei löscht die App gleich wieder."
            putTest("strom", measurePower())
            running = "kidauer"
            render()
            reveal("kidauer")
            heatStop?.let { return@guarded endLong(entry, it) }
            if (stopRequested) return@guarded endLong(entry, "gestoppt")

            setPhase(entry, "abkuehlen2")
            entry.put("abkuehlen2", waitUntilCool(entry))
            if (stopRequested) return@guarded endLong(entry, "gestoppt")
            setPhase(entry, "countdown")
            countdown()
            if (stopRequested) return@guarded endLong(entry, "gestoppt")

            hintOverride = "Läuft 10 Minuten. Lass die App offen und das Handy, wo es steht. Wird es zu heiß, beendet die App den Test selbst und speichert alles."
            val result = recordWithAi("kidauer", LONG_SECONDS, partial = entry)
            for (k in listOf("abkuehlen1", "abkuehlen2", "anzeige")) entry.opt(k)?.let { result.put(k, it) }
            putTest("kidauer", result)
            markDone("KI-Dauertest fertig", "kidauer")
        }
    }

    /** Neue Phase des Dauertests, gleich gespeichert (zeigt nach einem Rauswurf, wo er geschah). */
    private fun setPhase(entry: JSONObject, phase: String) {
        entry.put("phase", phase)
        putTestQuiet("kidauer", entry)
        noteState("kidauer", phase)
    }

    /** Dauertest endet vor der Aufnahme: „gestoppt“ oder ein Hitzestopp. */
    private fun endLong(entry: JSONObject, end: String) {
        entry.put("ende", end)
        putTest("kidauer", entry)
        noteState("kidauer", "ende")
        if (end.startsWith(HEAT_STOP)) {
            doneText = "Wegen Hitze vorzeitig beendet"
            donePhase = Phase.ERROR
        }
    }

    /**
     * Führt [block] aus; scheitert er mit einem Fehler, steht im gespeicherten Zwischenstand
     * [entry] von [key] dieser Fehler statt „läuft“ (sonst hielte ihn der nächste Start für einen
     * Rauswurf durch das System).
     */
    private suspend fun <T> guarded(key: String, entry: JSONObject, block: suspend () -> T): T = try {
        block()
    } catch (e: CancellationException) {
        throw e
    } catch (e: Exception) {
        entry.put("ende", "Fehler: ${e.message ?: e.javaClass.simpleName}".take(300))
        putTestQuiet(key, entry)
        throw e
    }

    /** Was gegen einen Dauertest spricht: (kurzer Status, Hinweis), oder null, wenn alles bereit ist. */
    private fun precheckLong(): Pair<String, String>? {
        val heat = DeviceHeat.read(this)
        if (heat.optBoolean("laedt")) return "Bitte Ladekabel abziehen" to
            "Zieh das Ladekabel ab und starte den Dauertest dann noch einmal. Mit Kabel lässt sich der Strom nicht messen, und Laden wärmt den Akku."
        val level = heat.optInt("akkuProzent", 100)
        if (level < MIN_BATTERY_PERCENT) return "Akku unter $MIN_BATTERY_PERCENT %" to
            "Lade das Handy auf mindestens 60 %, zieh das Kabel ab und starte den Dauertest dann. Er wartet von selbst, bis das Handy abgekühlt ist."
        val pm = getSystemService(PowerManager::class.java)
        if (pm?.isPowerSaveMode == true) return "Energiesparmodus ist an" to
            "Schalte den Energiesparmodus aus und starte den Dauertest dann noch einmal. Er bremst Kamera und KI."
        return null
    }

    /**
     * Wartet, bis das Handy abgekühlt ist: Wärmestufe „keine“ und der Akku höchstens 3 °C wärmer als
     * beim Start des Schnelltests (31 bis 34 °C), und das 30 Sekunden am Stück; höchstens 15
     * Minuten. Die Kamera ist dabei aus und der Bildschirm gedimmt. Verlässt man die App, wartet
     * sie weiter und macht weiter, sobald sie wieder sichtbar ist. Gibt zurück, wie es ausging.
     */
    private suspend fun waitUntilCool(entry: JSONObject): JSONObject {
        val start = report.optJSONObject("geraet")?.optJSONObject("waerme")?.optDouble("akkuGrad", Double.NaN) ?: Double.NaN
        val target = PipelineMath.coolTarget(start)
        val out = JSONObject().put("zielGrad", target)
        val first = DeviceHeat.read(this)
        out.put("vorher", first)
        val phase = entry.optString("phase")
        val began = SystemClock.elapsedRealtime()
        provider?.unbindAll()
        pipeline?.release()
        pipeline = null
        preview = null
        cooling = true
        setBrightness(COOL_BRIGHTNESS)
        hintOverride = "Die Kamera ist aus und der Bildschirm gedimmt, damit das Handy schneller abkühlt. Lass die App offen. " +
            "Es geht von selbst weiter, sobald der Akku höchstens ${de(target, 1)} °C hat und die Wärmestufe „keine“ ist, spätestens nach 15 Minuten."
        hintText.text = hintOverride
        var coolSince = 0L
        var nextState = 0L
        try {
            while (!stopRequested) {
                val heat = DeviceHeat.read(this)
                val stage = heat.optString("stufe")
                val temp = heat.optDouble("akkuGrad", Double.NaN)
                val now = SystemClock.elapsedRealtime()
                val waited = (now - began) / 1000
                if (now >= nextState) {
                    noteState("kidauer", phase, waited.toDouble(), DeviceHeat.stageOrder.indexOf(stage), temp)
                    nextState = now + (STATE_EVERY_S * 1000).toLong()
                }
                if (PipelineMath.isCool(stage, temp, target)) {
                    if (coolSince == 0L) coolSince = now
                } else {
                    coolSince = 0L
                }
                if (coolSince > 0L && now - coolSince >= COOL_HOLD_MS) {
                    out.put("gewartetS", waited).put("nachher", heat)
                    break
                }
                if (waited >= COOL_MAX_S) {
                    out.put("gewartetS", waited).put("nachher", heat).put("nichtAbgekuehlt", true)
                    break
                }
                val tempText = if (temp.isNaN()) "" else " · Akku ${de(temp, 1)} °C"
                setStatus(Phase.BUSY, "Kühlt ab$tempText · Wärmestufe: $stage", "${clock(waited)} / ${clock(COOL_MAX_S)}", waited / COOL_MAX_S.toFloat())
                delay(2000)
            }
            // Weiter erst, wenn die App wieder zu sehen ist (die Kamera braucht das).
            while (!stopRequested && !lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) delay(500)
        } finally {
            cooling = false
            setBrightness(TEST_BRIGHTNESS)
        }
        if (stopRequested) out.put("gestoppt", true)
        return out
    }

    /** 10 Sekunden mit Kamerabild vor der Aufnahme: Ausrichtung prüfen, dann nichts mehr anfassen. */
    private suspend fun countdown() {
        showPreviewOnly()
        hintOverride = "Gleich beginnt die Aufnahme. Prüf, ob das Motiv in der Mitte des Bildes ist; danach das Handy nicht mehr anfassen."
        hintText.text = hintOverride
        val end = SystemClock.elapsedRealtime() + COUNTDOWN_MS
        while (!stopRequested) {
            val left = end - SystemClock.elapsedRealtime()
            if (left <= 0) break
            setStatus(Phase.BUSY, "Aufnahme beginnt in ${(left + 999) / 1000} s", null, 1f - left / COUNTDOWN_MS.toFloat())
            delay(min(left, 250L))
        }
    }

    /** Wärmestufe (0 keine bis 6 Abschaltung) und Akkutemperatur in °C (NaN: unbekannt), jetzt gelesen. */
    private fun heatNow(): Pair<Int, Double> {
        val status = try {
            getSystemService(PowerManager::class.java)?.currentThermalStatus ?: 0
        } catch (_: Exception) {
            0
        }
        val tenths = try {
            registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        } catch (_: Exception) {
            null
        } ?: Int.MIN_VALUE
        return status to if (tenths == Int.MIN_VALUE) Double.NaN else tenths / 10.0
    }

    /** Warum es zu heiß ist („Wärmestufe Notfall“, „Akku 44,2 °C“), oder null (siehe [PipelineMath.tooHot]). */
    private fun heatReason(status: Int, batteryC: Double): String? = when {
        !PipelineMath.tooHot(status, batteryC) -> null
        status >= PipelineMath.EMERGENCY_STATUS -> "Wärmestufe ${DeviceHeat.stageNameAt(status)}"
        else -> "Akku ${de(batteryC, 1)} °C"
    }

    /**
     * Stand für den Beendigungsgrund (siehe [ExitHistory.setState]): Version, Test, Phase, Zeit,
     * Wärmestufe, Akku, Leistung und KI-Tempo, soweit bekannt. Beendet Android die App, steht er
     * beim nächsten Start in „beendigungen“ („zustand“). Ohne Wärmeangaben werden sie gelesen.
     */
    private fun noteState(
        test: String, phase: String, seconds: Double = Double.NaN, status: Int = -1,
        batteryC: Double = Double.NaN, watts: Double = Double.NaN, aiHz: Double = Double.NaN,
    ) {
        var st = status
        var bat = batteryC
        if (st < 0) {
            val (s, b) = heatNow()
            st = s
            if (bat.isNaN()) bat = b
        }
        ExitHistory.setState(
            this, StateText.line(
                "v" to appVersion, "test" to test, "ph" to phase,
                "t" to if (seconds.isNaN()) null else clock(seconds.toLong()),
                "st" to st, "bat" to bat, "W" to watts, "ki" to aiHz,
            )
        )
    }

    /**
     * KI-Vorschau ohne Aufnahme: das Bild mit Unschärfe, wie es aufgenommen würde. Antippen
     * wählt das Motiv, das scharf bleibt.
     */
    private fun onPreviewButton() {
        if (running == "vorschau") {
            requestStop()
            return
        }
        if (running != null || !modelsOk) return
        launchTest {
            running = "vorschau"
            hintOverride = "Tippe im Kamerabild auf eine Person oder einen Gegenstand: Das ganze Motiv bleibt scharf, der Hintergrund wird unscharf. " +
                "Tippst du gleich noch einmal auf dieselbe Stelle, nimmt die KI einen kleineren Teil. Ohne Tippen nimmt sie, was in der Mitte ist. " +
                "Das Bild läuft 0,4 Sekunden hinterher, damit Motiv und Unschärfe genau zusammenpassen."
            render()
            setStatus(Phase.BUSY, "Bereitet die KI vor …")
            val load = ensureLiveModels()
            if (load != null) report.optJSONObject("ki")?.put("ladenVorschau", load)
            if (stopRequested) return@launchTest
            val effect = BokehProcessor(applicationContext, strength = 1f, aiPool = pool, aiDemand = engine, thumbnailAfter = 0)
            bind(null, effect)
            startLive(effect)
            setStatus(Phase.LIVE, "KI-Vorschau läuft")
            var hot: String? = null
            try {
                var nextCheck = 0L
                while (!stopRequested && liveJob?.isActive == true) {
                    // Wird es zu heiß, hört die Vorschau selbst auf, bevor das System die App beendet.
                    val now = SystemClock.elapsedRealtime()
                    if (now >= nextCheck) {
                        nextCheck = now + 2000
                        hot = heatReason(engine.heatStatus, heatNow().second)
                        if (hot != null) break
                    }
                    delay(200)
                }
            } finally {
                withContext(NonCancellable) { stopLive() }
                // Die letzte Vorschau steht im Bericht, damit sich auch dort etwas beurteilen lässt,
                // samt jedem Antippen (alle Masken und die Wahl).
                val o = JSONObject().put("zeit", nowIso()).put("version", appVersion)
                    .put("ki", engine.stats.totalJson()).put("pipeline", effect.stats.toJson())
                    .put("antippen", withContext(NonCancellable + engine.dispatcher) { engine.startsJson() })
                liveError?.let { o.put("fehler", it) }
                hot?.let { o.put("ende", "$HEAT_STOP: $it") }
                ki()?.put("vorschau", o)
            }
            hot?.let {
                stopNote = "Vorschau wegen Hitze beendet ($it)"
                hintAfter = "Das Handy ist zu warm geworden. Lass es ein paar Minuten abkühlen, bevor du die Kamera wieder nutzt."
            }
            liveError?.let { throw IllegalStateException(it) }
        }
    }

    /** Lädt die Modelle für den Live-Betrieb, falls nötig. Gibt das Protokoll zurück oder null. */
    private suspend fun ensureLiveModels(): JSONObject? {
        // Leistungstest und Einzelmessung haben Einstellung und Bausteine gewählt; das gilt auch nach einem Neustart.
        // Live immer auf Dauerleistung (mit der Übersetzung der schnellsten Einstellung): In 0.3.15
        // lief live die Spitzenleistung, und das Handy wurde im Dauertest zu heiß.
        val ki = ki()
        val chosen = ChipSetting.fromKey(ki?.optJSONObject("leistung")?.optString("gewaehlt"))
        val chosenAttn = ki?.optJSONObject("einzeln")?.optString("liveMemoryAttention").orEmpty()
        val attn = ModelId.entries.firstOrNull { it.key == chosenAttn }
        val qcMemory = ki?.optJSONObject("einzeln")?.optString("liveMemoryEncoder")?.takeIf { it.isNotEmpty() }?.let { it == "qualcomm" }
        val metaDecoder = ki?.optJSONObject("einzeln")?.optString("liveDecoder")?.takeIf { it.isNotEmpty() }?.let { it == "meta" }
        val ready = withContext(engine.dispatcher) {
            // Ohne Wahl im Bericht die geladene Übersetzung, aber auch dann auf Dauerleistung.
            engine.configureLive((chosen ?: engine.chip).forLive(), attn, qcMemory, metaDecoder)
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

    /** Holt ein frisches Kamerabild aus der Pipeline; der Aufrufer gibt es mit pool.release zurück. */
    private suspend fun grabFrame(): FrameSlot? {
        pool.drain()
        grabDemand.on = true
        try {
            return withContext(Dispatchers.IO) { pool.take(6000) }
        } finally {
            grabDemand.on = false
            pool.drain()
        }
    }

    private fun frameInfo(frame: FrameSlot, effect: BokehProcessor): JSONObject {
        val o = JSONObject()
        o.put("kameraBild", "${frame.width}x${frame.height}")
        o.put("quelle", "Grafik-Pipeline")
        o.put("vorbereitungMs", r1(frame.prepMs))
        o.put("pipeline", effect.stats.toJson())
        // Was die KI bekommt, als kleines Bild: Ausrichtung und Farben auf einen Blick.
        try {
            o.put("kiBildJpegBase64", frame.thumbnailJpegBase64())
        } catch (e: Exception) {
            o.put("kiBildFehler", e.toString().take(200))
        }
        return o
    }

    /**
     * Startet die KI-Schleife auf den Bildern von [effect]: Die Pipeline bereitet die Bilder im
     * Takt der KI vor und bekommt jede Maske zurück. Antippen im Bild wählt ein neues Motiv.
     */
    private fun startLive(effect: BokehProcessor) {
        pool.drain()
        liveError = null
        engine.beginLive()
        engine.stats.reset()
        engine.maskSink = effect
        pool.resetCounters()
        tapOverlay.onTap = { u, v -> engine.track(u, v) }
        liveJob = lifecycleScope.launch(engine.dispatcher) {
            try {
                engine.runLive(pool)
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
        engine.maskSink = null
        pool.drain()
        tapOverlay.onTap = null
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
     * 4K mit 30 fps durch die Grafik-Pipeline aufnehmen: Beide Modelle rechnen auf den
     * Kamerabildern, die Maske der Verfolgung macht den Hintergrund unscharf, die Wärmeregelung
     * nimmt die KI bei Hitze zurück. Alle 30 Sekunden ein Eintrag: Tempo der KI, Bildrate,
     * Pipeline, Wärme, Strom und Akku; derselbe Stand liegt dann auch schon im gespeicherten
     * Bericht („ende“: „läuft“), damit ein Rauswurf durch das System nichts mehr löscht. Wird es
     * zu heiß (siehe [PipelineMath.tooHot]), beendet die App die Aufnahme selbst. Mit [keepClip]
     * kommt der Clip in die Galerie; [delayFrames] ist die Verzögerung der Pipeline. [partial]:
     * Zwischenstand, in den die Aufnahme schreibt (der Dauertest gibt seinen mit), sonst ein neuer.
     */
    private suspend fun recordWithAi(
        key: String,
        seconds: Int,
        keepClip: Boolean = false,
        delayFrames: Int = BokehProcessor.DEFAULT_DELAY,
        partial: JSONObject? = null,
    ): JSONObject {
        setStatus(Phase.BUSY, "Bereitet die KI vor …")
        val load = ensureLiveModels()
        if (stopRequested) return JSONObject().put("version", appVersion).put("zeit", nowIso()).put("ende", "gestoppt")
        val effect = BokehProcessor(
            applicationContext, strength = 1f, requestedDelay = delayFrames, aiPool = pool, aiDemand = engine,
            thumbnailAfter = if (keepClip) BokehProcessor.THUMBNAIL_AFTER else 0
        )
        startLive(effect)
        val log = JSONArray()
        val power = PowerMeter(this)
        val thermal = ThermalLog(this)
        // Zwischenstand: steht ab dem Start der Aufnahme im gespeicherten Bericht.
        val progress = (partial ?: JSONObject().put("version", appVersion).put("zeit", nowIso()))
            .put("verzoegerungBilder", delayFrames)
        var next = LOG_EVERY_S
        var prevFrames = -1L
        var prevTs = 0L
        var prevBytes = 0L
        var prevT = 0.0
        // Stand für den Beendigungsgrund: alle 10 s und bei jedem Wechsel der Wärmestufe.
        var nextState = 0.0
        var stateT = 0.0
        var stateCycles = 0
        var stateStatus = -1
        // Die KI- und Pipeline-Zahlen gelten nur für die Zeit der Aufnahme: Beim Ende der Aufnahme
        // werden sie festgehalten; die KI hält an, sobald die Datei fertig ist.
        var aiTotal: JSONObject? = null
        var pipelineTotal: JSONObject? = null
        var prepared = 0L
        var skipped = 0L
        val result = try {
            record(
                key, FPS, seconds, stabilize = false, effect = effect,
                clipName = if (keepClip) "CineBo-Unschaerfe-${fileStamp()}.mp4" else null,
                onStart = {
                    engine.stats.reset()
                    effect.stats.resetCounters()
                    pool.resetCounters()
                    power.start()
                    thermal.start()
                    if (!progress.has("startMs")) progress.put("startMs", System.currentTimeMillis())
                    progress.put("aufnahmeStartMs", System.currentTimeMillis()).put("pid", android.os.Process.myPid())
                        .put("phase", "aufnahme").put("ende", RUNNING)
                    putTestQuiet(key, progress)
                    noteState(key, "aufnahme", 0.0)
                },
                onStop = {
                    aiTotal = engine.stats.totalJson()
                    pipelineTotal = effect.stats.toJson()
                    prepared = pool.prepared
                    skipped = pool.skipped
                },
                afterStop = {
                    // Die Datei ist fertig: KI anhalten und ihre Anzeige gleich ausblenden.
                    engine.requestStop()
                    badgeJob?.cancel()
                    badgeJob = null
                    aiBadge.visibility = View.GONE
                },
                stopReason = {
                    val hot = heatReason(thermal.status, power.lastTempC)
                    when {
                        hot != null -> "$HEAT_STOP: $hot nach ${clock((SystemClock.elapsedRealtime() - recordingSince) / 1000)} min"
                        liveJob?.isActive == false -> "abgebrochen, weil die KI nicht mehr lief"
                        else -> null
                    }
                }
            ) { t, stats ->
                if (t >= nextState || thermal.status != stateStatus) {
                    val cycles = engine.stats.cycles()
                    val hz = if (t - stateT >= 1.0) (cycles - stateCycles) / (t - stateT) else Double.NaN
                    noteState(key, "aufnahme", t, thermal.status, power.lastTempC, power.recentWatts(), hz)
                    stateT = t
                    stateCycles = cycles
                    stateStatus = thermal.status
                    nextState = t + STATE_EVERY_S
                }
                if (t >= next) {
                    val (count, first, last) = frames.snapshot()
                    val entry = JSONObject().put("s", next.toInt())
                    val fromFrames = if (prevFrames < 0) 1L else prevFrames
                    val fromTs = if (prevFrames < 0) first else prevTs
                    if (last > fromTs) entry.put("kameraFps", r1((count - fromFrames) / ((last - fromTs) / 1e9)))
                    if (stats != null && t > prevT) {
                        entry.put("mbits", r1((stats.numBytesRecorded - prevBytes) * 8 / (t - prevT) / 1e6))
                        prevBytes = stats.numBytesRecorded
                    }
                    val window = engine.stats.closeWindow()
                    entry.put("ki", window)
                    entry.put("regelstufe", DeviceHeat.stageNameAt(engine.heatLevel))
                    entry.put("pipeline", effect.stats.closeWindow())
                    val heat = DeviceHeat.read(this@MainActivity)
                    entry.put("waerme", heat)
                    val watts = power.closeWindow()
                    entry.put("leistung", watts)
                    log.put(entry)
                    prevFrames = count
                    prevTs = last
                    prevT = t
                    next += LOG_EVERY_S
                    // Zwischenstand sichern (Schreiben im Hintergrund).
                    progress.put("laufS", r1(t)).put("verlauf", log).put("ki", engine.stats.totalJson())
                        .put("pipeline", effect.stats.toJson()).put("leistung", power.total())
                        .put("waermeStufen", thermal.toJson())
                    saveReportAsync()
                    if (seconds > LIVE_SECONDS) {
                        val parts = mutableListOf("Läuft")
                        if (window.has("hz")) parts.add("KI ${de(window.optDouble("hz"), 1)}/s")
                        if (entry.has("kameraFps")) parts.add("${de(entry.optDouble("kameraFps"), 1)} fps")
                        if (watts.has("watt")) parts.add("${de(watts.optDouble("watt"), 1)} W")
                        if (heat.has("akkuGrad")) parts.add("Akku ${de(heat.optDouble("akkuGrad"), 1)} °C")
                        parts.add("Wärmestufe: ${heat.optString("stufe")}")
                        hintOverride = parts.joinToString(" · ")
                        hintText.text = hintOverride
                    }
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            // Sonst stünde der Zwischenstand weiter auf „läuft“ und gälte beim nächsten Start als Rauswurf.
            if (progress.optString("ende") == RUNNING) {
                progress.put("ende", "Fehler: ${e.message ?: e.javaClass.simpleName}".take(300))
                putTestQuiet(key, progress)
            }
            throw e
        } finally {
            power.stop()
            thermal.stop()
            withContext(NonCancellable) { stopLive() }
        }
        val ai = aiTotal ?: engine.stats.totalJson()
        // Wie oft die Pipeline ein KI-Bild vorbereitete, und wie oft sie es ließ, weil das vorige noch nicht genommen war.
        ai.put("kiBilderVorbereitet", if (aiTotal != null) prepared else pool.prepared)
        ai.put("kiBildLagNochBereit", if (aiTotal != null) skipped else pool.skipped)
        val (setting, chainName) = withContext(engine.dispatcher) { engine.chip to engine.liveTrackerName() }
        ai.put("einstellung", setting.key)
        ai.put("verfolgungKette", chainName)
        ai.put("memoryAttention", engine.liveAttn.key)
        ai.put("waermeregelung", withContext(engine.dispatcher) { engine.heatJson() })
        ai.put("antippen", withContext(engine.dispatcher) { engine.startsJson() })
        liveError?.let { ai.put("fehler", it) }
        result.put("ki", ai)
        result.put("pipeline", pipelineTotal ?: effect.stats.toJson())
        result.put("verlauf", log)
        result.put("leistung", power.total())
        result.put("waermeStufen", thermal.toJson())
        result.put("verzoegerungBilder", delayFrames)
        for (k in listOf("startMs", "aufnahmeStartMs")) progress.optLong(k, 0L).takeIf { it > 0 }?.let { result.put(k, it) }
        if (load != null) result.put("laden", load)
        noteState(key, "fertig")
        return result
    }

    /**
     * Vergleichsclip: 10 Sekunden 4K ohne Grafik-Pipeline, so wie CameraX allein filmt. Neben dem
     * Unschärfe-Clip zeigt er, was die Pipeline am Bild ändert (Farben, Schärfe), neben einem Clip
     * der OPPO-Kamera, wie weit CameraX von ihr entfernt ist. Kommt in die Galerie.
     */
    private suspend fun recordComparison(): JSONObject =
        record("vergleich", FPS, COMPARE_SECONDS, stabilize = false, clipName = "CineBo-Vergleich-${fileStamp()}.mp4")

    /**
     * Strombedarf, Abschnitt für Abschnitt, jeweils nur eine Sache anders als davor, damit sich
     * jeder Unterschied einem Baustein zuschreiben lässt ([POWER_PHASES]):
     *  aus: Kamera aus, Bildschirm an · kamera: Vorschau · aufnahme: 4K-Aufnahme ·
     *  in einer zweiten 4K-Aufnahme durch die Grafik-Pipeline: unschaerfe: Unschärfe mit fester
     *  Maske · kibilder: dazu KI-Bilder und Masken 5-mal pro Sekunde wie live, aber ohne KI-Chip
     *  ([SyntheticAi]) · ki: die echte KI auf Dauerleistung, 5 pro Sekunde, ohne Tiefe · kiTiefe:
     *  dazu die Tiefe bei jedem Durchgang · spitze: wie „ki“, aber auf Spitzenleistung (so lief
     *  0.3.15) · aufnahme2: noch einmal die 4K-Aufnahme, um eine Drift durch Erwärmung zu sehen.
     * Je Abschnitt einschwingen (8 s nach neuer Bindung der Kamera, 5 s nach einem Wechsel in der
     * Pipeline), dann 2 × 10 s messen; weichen die Hälften mehr als 5 % ab, steht es dabei. Die
     * Aufnahmen löscht die App gleich wieder. Jeder Abschnitt steht sofort im Bericht. Wird es zu
     * heiß, hört die Messung auf ([heatStop]).
     */
    private suspend fun measurePower(): JSONObject {
        val out = JSONObject().put("version", appVersion).put("zeit", nowIso()).put("ende", RUNNING)
            .put("startMs", System.currentTimeMillis()).put("pid", android.os.Process.myPid())
            .put("einschwingenS", JSONArray().put(POWER_SETTLE_BIND_MS / 1000).put(POWER_SETTLE_SWITCH_MS / 1000))
            .put("messenS", 2 * POWER_HALF_MS / 1000).put("kiAbstandMs", POWER_AI_INTERVAL_MS).put("anzeige", displayInfo())
        val heat = DeviceHeat.read(this)
        out.put("waermeVorher", heat)
        if (heat.optBoolean("laedt")) return out.put("ende", "ok").put("fehler", "Nicht gemessen: Ladekabel steckte")
        val phases = JSONArray()
        out.put("abschnitte", phases)
        putTestQuiet("strom", out)
        return guarded("strom", out) {
            // Die KI vorher laden: Das Laden gehört in keinen Abschnitt.
            setStatus(Phase.BUSY, "Strombedarf · bereitet die KI vor …")
            ensureLiveModels()?.let { out.put("laden", it) }
            val meter = PowerMeter(this)
            val cpuStart = android.os.Process.getElapsedCpuTime()
            meter.start()
            try {
                powerPhases(meter) { o ->
                    phases.put(o)
                    putTestQuiet("strom", out)
                }
            } finally {
                meter.stop()
                engine.fixedLive = null
                if (liveJob != null) withContext(NonCancellable) { stopLive() }
            }
            out.put("gesamt", meter.total())
            out.put("cpuS", r1((android.os.Process.getElapsedCpuTime() - cpuStart) / 1000.0))
            out.put("ende", heatStop ?: if (stopRequested) "gestoppt" else "ok")
            out.put("waermeNachher", DeviceHeat.read(this))
            if (powerGo()) {
                // Live wieder auf Dauerleistung, damit die Aufnahme gleich beginnen kann.
                setStatus(Phase.BUSY, "Bereitet die KI vor …")
                ensureLiveModels()?.let { out.put("ladenDanach", it) }
            }
            out
        }
    }

    /** Die Abschnitte des Strombedarfs der Reihe nach (siehe [measurePower]); [add] nimmt jeden fertigen auf. */
    private suspend fun powerPhases(meter: PowerMeter, add: (JSONObject) -> Unit) {
        fun failed(key: String, problem: String) = add(JSONObject().put("abschnitt", key).put("fehler", problem))
        // aus, kamera, aufnahme: ohne Grafik-Pipeline
        provider?.unbindAll()
        pipeline?.release()
        pipeline = null
        preview = null
        add(powerMeasure("aus", meter, POWER_SETTLE_BIND_MS, null))
        if (!powerGo()) return
        if (!showPreviewOnly()) return failed("kamera", cameraProblem ?: "Vorschau ließ sich nicht starten")
        waitForStreaming(5000)
        add(powerMeasure("kamera", meter, POWER_SETTLE_BIND_MS, null))
        if (!powerGo()) return
        powerRecording(null) { add(powerMeasure("aufnahme", meter, POWER_SETTLE_BIND_MS, null)) }?.let { failed("aufnahme", it) }
        if (!powerGo()) return

        // unschaerfe bis spitze: eine Aufnahme durch die Grafik-Pipeline; was die Masken liefert, wechselt mitten drin.
        val synthetic = SyntheticAi(pool, hz = 1000.0 / POWER_AI_INTERVAL_MS, depthEvery = 0)
        val effect = BokehProcessor(
            applicationContext, strength = 1f, aiPool = pool, aiDemand = synthetic, thumbnailAfter = 0,
            staticMask = SyntheticAi.ELLIPSE
        )
        synthetic.sink = effect
        val problem = try {
            powerRecording(effect) {
                add(powerMeasure("unschaerfe", meter, POWER_SETTLE_BIND_MS, null))
                if (powerGo()) {
                    // KI-Bilder und Masken wie live, aber ohne KI-Chip.
                    pool.drain()
                    effect.staticMask = null
                    synthetic.start()
                    add(powerMeasure("kibilder", meter, POWER_SETTLE_SWITCH_MS, synthetic))
                    withContext(Dispatchers.IO) { synthetic.stop() }
                }
                if (powerGo()) {
                    // Die echte KI im selben Takt, ohne Tiefe (wie live bei starker Hitze, aber 5-mal pro Sekunde).
                    synthetic.sink = null
                    engine.fixedLive = AiEngine.FixedLive(POWER_AI_INTERVAL_MS, 0)
                    effect.aiDemand = engine
                    startPowerLive(effect)
                    add(powerMeasure("ki", meter, POWER_SETTLE_SWITCH_MS, engine))
                }
                if (powerGo() && liveJob?.isActive == true) {
                    engine.fixedLive = AiEngine.FixedLive(POWER_AI_INTERVAL_MS, 1)
                    add(powerMeasure("kiTiefe", meter, POWER_SETTLE_SWITCH_MS, engine))
                }
                if (powerGo()) add(powerBurst(effect, meter))
                withContext(NonCancellable) { stopLive() }
                engine.fixedLive = null
            }
        } finally {
            withContext(NonCancellable + Dispatchers.IO) { synthetic.stop() }
        }
        problem?.let { failed("unschaerfe", it) }
        if (!powerGo()) return
        powerRecording(null) { add(powerMeasure("aufnahme2", meter, POWER_SETTLE_BIND_MS, null)) }?.let { failed("aufnahme2", it) }
    }

    /** Abschnitt „spitze“: dieselbe KI, neu geladen auf Spitzenleistung, wie sie in 0.3.15 live lief. */
    private suspend fun powerBurst(effect: BokehProcessor, meter: PowerMeter): JSONObject {
        withContext(NonCancellable) { stopLive() }
        setStatus(Phase.BUSY, "Strombedarf · lädt die KI für Spitzenleistung …")
        val (load, ready) = withContext(engine.dispatcher) {
            engine.configureLive(ChipSetting(PerfMode.BURST, engine.chip.opt), null, null, null)
            val l = if (engine.liveReady()) null else engine.load(engine.liveSet()) { done, total, _ -> runOnUiThread { loadingStatus(done, total) } }
            l to engine.liveReady()
        }
        if (!ready) {
            return JSONObject().put("abschnitt", "spitze").put("fehler", "Die KI ließ sich auf Spitzenleistung nicht laden")
                .also { o -> load?.let { o.put("laden", it) } }
        }
        engine.fixedLive = AiEngine.FixedLive(POWER_AI_INTERVAL_MS, 0)
        startPowerLive(effect)
        return powerMeasure("spitze", meter, POWER_SETTLE_SWITCH_MS, engine).also { o -> load?.let { o.put("laden", it) } }
    }

    /** KI-Schleife für den Strombedarf: wie [startLive], aber ohne Antippen (ein Tipp würde die Messung stören). */
    private fun startPowerLive(effect: BokehProcessor) {
        startLive(effect)
        tapOverlay.onTap = null
    }

    /** Der Strombedarf misst weiter: nicht gestoppt und nicht zu heiß. */
    private fun powerGo(): Boolean = !stopRequested && heatStop == null

    /**
     * Eine 4K-Aufnahme für den Strombedarf ([effect]: durch die Grafik-Pipeline, sonst CameraX
     * allein), während [body] misst; die Datei löscht die App gleich wieder. Gibt einen Fehler
     * zurück, oder null.
     */
    private suspend fun powerRecording(effect: BokehProcessor?, body: suspend () -> Unit): String? {
        val recorder = Recorder.Builder().setQualitySelector(QualitySelector.from(Quality.UHD)).build()
        val capture = VideoCapture.Builder(recorder).setTargetFrameRate(Range(FPS, FPS)).build()
        try {
            bind(capture, effect)
        } catch (e: Exception) {
            return "Kamera ließ sich so nicht starten: ${e.message ?: e.javaClass.simpleName}"
        }
        waitForStreaming(5000)
        val file = File(getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir, "cinebo-strom.mp4")
        file.delete()
        val started = CompletableDeferred<Unit>()
        val finished = CompletableDeferred<VideoRecordEvent.Finalize>()
        val recording = try {
            var pending = recorder.prepareRecording(this, FileOutputOptions.Builder(file).build())
            if (has(Manifest.permission.RECORD_AUDIO)) pending = pending.withAudioEnabled()
            pending.start(ContextCompat.getMainExecutor(this)) { e ->
                when (e) {
                    is VideoRecordEvent.Start -> started.complete(Unit)
                    is VideoRecordEvent.Finalize -> finished.complete(e)
                }
            }
        } catch (e: Exception) {
            return "Aufnahme ließ sich nicht starten: ${e.message ?: e.javaClass.simpleName}"
        }
        recBadge.text = "REC"
        recBadge.visibility = View.VISIBLE
        try {
            withTimeoutOrNull(5000) { started.await() }
            if (!started.isCompleted) return "Aufnahme startete nicht"
            body()
            if (finished.isCompleted) {
                val end = finished.await()
                return "Aufnahme brach vorzeitig ab: " + if (end.hasError()) finalizeText(end.error) else "ohne Fehlercode"
            }
            return null
        } finally {
            recBadge.visibility = View.GONE
            if (!finished.isCompleted) recording.stop()
            withContext(NonCancellable) { withTimeoutOrNull(5000) { finished.await() } }
            file.delete()
        }
    }

    /**
     * Ein Abschnitt des Strombedarfs: [settle] einschwingen, dann zwei Hälften zu je
     * [POWER_HALF_MS] messen. [source]: woher die KI-Durchgänge kommen ([SyntheticAi] oder die KI),
     * für ihr Tempo im Abschnitt.
     */
    private suspend fun powerMeasure(key: String, meter: PowerMeter, settle: Long, source: Any?): JSONObject {
        val title = POWER_PHASES[key] ?: key
        val o = JSONObject().put("abschnitt", key).put("titel", title)
        noteState("strom", key)
        waitPhase(key, settle, "schwingt ein")
        if (!powerGo()) return o.put("ende", heatStop ?: "gestoppt")
        fun cycles(): Int = when (source) {
            is SyntheticAi -> source.count
            is AiEngine -> source.stats.cycles()
            else -> 0
        }
        meter.closeWindow()
        val before = cycles()
        val cpu0 = android.os.Process.getElapsedCpuTime()
        val began = SystemClock.elapsedRealtime()
        waitPhase(key, POWER_HALF_MS, "misst, 1. Hälfte")
        val first = meter.closeWindow()
        if (powerGo()) waitPhase(key, POWER_HALF_MS, "misst, 2. Hälfte")
        val second = meter.closeWindow()
        val seconds = (SystemClock.elapsedRealtime() - began) / 1000.0
        o.put("haelften", JSONArray().put(first).put(second))
        val w1 = first.optDouble("watt", Double.NaN)
        val w2 = second.optDouble("watt", Double.NaN)
        if (!w1.isNaN() && !w2.isNaN()) {
            val mean = (w1 + w2) / 2.0
            o.put("watt", r2(mean))
            if (mean > 0 && kotlin.math.abs(w1 - w2) / mean > 0.05) o.put("schwankt", true)
        }
        if (source != null && seconds > 0) o.put("kiProSekunde", r1((cycles() - before) / seconds))
        // Rechenzeit aller Threads der App, in Prozent eines Kerns.
        if (seconds > 0) o.put("cpuProzentApp", r1((android.os.Process.getElapsedCpuTime() - cpu0) / (seconds * 10.0)))
        o.put("waerme", DeviceHeat.read(this))
        if (source is AiEngine) {
            o.put("einstellung", source.chip.key)
            liveError?.let { o.put("fehler", it) }
        }
        if (!powerGo()) o.put("ende", heatStop ?: "gestoppt")
        return o
    }

    /**
     * Wartet [ms] und zeigt dabei, welcher Abschnitt wie weit ist; prüft alle 2 s die Wärme.
     * Ein Stopp oder Hitze ([heatStop]) beendet das Warten.
     */
    private suspend fun waitPhase(key: String, ms: Long, what: String) {
        val title = POWER_PHASES[key] ?: key
        val step = POWER_PHASES.keys.indexOf(key) + 1
        val end = SystemClock.elapsedRealtime() + ms
        var nextCheck = 0L
        while (powerGo()) {
            val now = SystemClock.elapsedRealtime()
            val left = end - now
            if (left <= 0) break
            if (now >= nextCheck) {
                nextCheck = now + 2000
                val (status, battery) = heatNow()
                heatReason(status, battery)?.let { heatStop = "$HEAT_STOP: $it beim Strombedarf" }
            }
            val done = ms - left
            setStatus(Phase.BUSY, "Strombedarf $step/${POWER_PHASES.size} · $title · $what", "${done / 1000} / ${ms / 1000} s", done / ms.toFloat())
            delay(min(left, 500L))
        }
    }

    /**
     * Nimmt in 4K 16:9 auf, zählt dabei jedes Kamerabild und wertet danach die Datei aus. Mit
     * [effect] läuft alles durch die Grafik-Pipeline. Mit [clipName] kommt der Clip danach in die
     * Galerie (Album CineBo), sonst wird die Datei gelöscht und es bleiben nur die Messwerte.
     * [onStop] hält beim Ende der Aufnahme die Zahlen fest, [afterStop] kommt, wenn die Datei steht.
     * Liefert [stopReason] einen Text, endet die Aufnahme vorzeitig mit diesem Grund.
     */
    @OptIn(ExperimentalCamera2Interop::class)
    private suspend fun record(
        key: String,
        fps: Int,
        seconds: Int,
        stabilize: Boolean,
        effect: BokehProcessor? = null,
        clipName: String? = null,
        onStart: () -> Unit = {},
        onStop: () -> Unit = {},
        afterStop: () -> Unit = {},
        stopReason: () -> String? = { null },
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
                .put("pipeline", effect != null)
                .put("unschaerfe", effect?.blurs == true)
        )
        val label = (if (effect?.blurs == true) "4K mit Unschärfe" else "4K") + if (stabilize) " · Stabilisator" else ""
        setStatus(Phase.BUSY, "Kamera startet · $label")

        // Ohne eigenes Seitenverhältnis ist 4K genau die Größe aus dem Profil des Encoders (3840 × 2160).
        val recorder = Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.UHD))
            .build()
        val builder = VideoCapture.Builder(recorder).setTargetFrameRate(Range(fps, fps))
        if (stabilize) builder.setVideoStabilizationEnabled(true)
        val videoCapture = builder.build()

        if (stopRequested) {
            out.put("ende", "gestoppt")
            return attachEvents(out)
        }
        val camera = try {
            bind(videoCapture, effect)
        } catch (e: Exception) {
            out.put("fehler", "Kamera ließ sich so nicht starten: ${e.message ?: e.javaClass.simpleName}")
            return attachEvents(out)
        }
        out.put("kamera", Camera2CameraInfo.from(camera.cameraInfo).cameraId)
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
        var abortReason: String? = null
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
                    val reason = stopReason()
                    if (reason != null) {
                        abortReason = reason
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
            // Die Pipeline gibt jedes Bild um ihre Verzögerung später aus, mit dem Zeitstempel der
            // Kamera. Der Encoder wartet nach dem Stopp auf das erste Bild ab der Stoppzeit (bis
            // zu 1 s), so kommt das Ende der Aufnahme noch vollständig hinein.
            if (!finished.isCompleted) recording.stop()
        }

        setStatus(Phase.BUSY, "Wertet die Aufnahme aus …")
        // Die KI rechnet weiter, bis die Datei fertig ist: Auch die letzten, verzögert
        // ausgegebenen Bilder bekommen ihre Masken.
        val end = try {
            withTimeoutOrNull(30_000) { finished.await() }
        } finally {
            afterStop()
        }
        val endText = when {
            end == null -> "keine Rückmeldung vom Recorder"
            end.hasError() -> finalizeText(end.error)
            !started.isCompleted -> "Aufnahme startete nicht"
            stoppedByUser -> "gestoppt"
            abortReason != null -> abortReason
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
            if (clipName != null) {
                setStatus(Phase.BUSY, "Legt den Clip in die Galerie …")
                out.put("galerie", saveToGallery(file, clipName))
            }
        } else {
            out.put("datei", JSONObject().put("fehler", "keine Datei"))
        }
        file.delete()
        return out
    }

    /**
     * Kopiert einen Clip in die Galerie, Album [GALLERY_FOLDER]. Ein halb geschriebener Eintrag
     * wird bei einem Fehler wieder entfernt; der Fehler steht dann im Bericht.
     */
    private suspend fun saveToGallery(file: File, name: String): JSONObject = withContext(Dispatchers.IO) {
        val folder = "${Environment.DIRECTORY_MOVIES}/$GALLERY_FOLDER"
        val out = JSONObject().put("name", name).put("ordner", folder)
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, name)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, folder)
            put(MediaStore.Video.Media.IS_PENDING, 1)
        }
        val resolver = contentResolver
        val uri = try {
            resolver.insert(MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values)
        } catch (e: Exception) {
            return@withContext out.put("fehler", "Galerie nimmt keinen Clip an: ${e.message ?: e.javaClass.simpleName}")
        } ?: return@withContext out.put("fehler", "Galerie nimmt keinen Clip an")
        try {
            val stream = resolver.openOutputStream(uri) ?: throw IOException("kein Schreibzugang")
            stream.use { target -> file.inputStream().use { it.copyTo(target, 1 shl 20) } }
            values.clear()
            values.put(MediaStore.Video.Media.IS_PENDING, 0)
            if (resolver.update(uri, values, null, null) == 0) throw IOException("Clip ließ sich in der Galerie nicht freigeben")
            out.put("mb", (file.length() / 1e6).roundToInt())
        } catch (e: Exception) {
            // Den halb geschriebenen Eintrag wieder entfernen; scheitert auch das, bleibt die Messung trotzdem.
            runCatching { resolver.delete(uri, null, null) }
            out.put("fehler", "Clip ließ sich nicht in die Galerie legen: ${e.message ?: e.javaClass.simpleName}")
        }
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

    /** Wie [putTest], aber ohne neu zu zeichnen und im Hintergrund gespeichert (während einer Aufnahme). */
    private fun putTestQuiet(key: String, value: JSONObject) {
        val tests = report.optJSONObject("tests") ?: JSONObject().also { report.put("tests", it) }
        tests.put(key, value)
        saveReportAsync()
    }

    /**
     * Speichert den Bericht, ohne den Hauptthread mit dem Schreiben aufzuhalten. Den Text baut der
     * Hauptthread (der Bericht ändert sich nur dort); geschrieben wird der Reihe nach, und ein
     * älterer Stand überschreibt nie einen neueren ([ReportStore]).
     */
    private fun saveReportAsync() {
        val seq = store.nextSeq()
        val text = report.toString()
        saver.execute { store.saveText(text, seq) }
    }

    /** Schreibt Berichte nacheinander auf einem eigenen Thread. */
    private val saver = java.util.concurrent.Executors.newSingleThreadExecutor { r -> Thread(r, "bericht").apply { isDaemon = true } }

    /**
     * Tests, die beim Start noch auf „läuft“ stehen, wurden mittendrin beendet (meist vom System).
     * Sie bekommen „App wurde beendet“ und ihren Beendigungsgrund: den ihres Prozesses (pid),
     * sonst den ersten nach ihrem Start.
     */
    private fun closeUnfinishedTests(exits: JSONArray) {
        val tests = report.optJSONObject("tests") ?: return
        val keys = mutableListOf<String>()
        tests.keys().forEach { keys += it }
        for (key in keys) {
            val t = tests.optJSONObject(key) ?: continue
            if (t.optString("ende") != RUNNING) continue
            t.put("ende", KILLED)
            val since = t.optLong("startMs", 0L)
            val pid = t.optInt("pid", 0)
            var match: JSONObject? = null
            for (i in 0 until exits.length()) {
                val e = exits.optJSONObject(i) ?: continue
                val ms = e.optLong("zeitMs", 0L)
                if (ms < since) continue
                if (pid > 0 && e.optInt("pid") == pid) {
                    match = e
                    break
                }
                if (match == null || ms < match.optLong("zeitMs")) match = e
            }
            match?.let { t.put("beendigung", it) }
        }
    }

    /**
     * Nach einem Update beginnt der Bericht neu, damit sich Messungen zweier Versionen nicht
     * mischen. Der alte bleibt als eigene Datei liegen; was er über Rauswürfe und Abstürze weiß,
     * steht kurz im neuen („fruehereVersion“).
     */
    private fun startNewVersion() {
        val previous = reportVersion(report)
        if (previous.isEmpty() || previous == appVersion) return
        val note = JSONObject().put("version", previous)
        store.archive("bericht-stufe3-$previous", report)?.let { note.put("datei", it) }
        val tests = report.optJSONObject("tests")
        val ended = JSONObject()
        tests?.keys()?.forEach { key ->
            val t = tests.optJSONObject(key) ?: return@forEach
            if (t.optString("ende") != KILLED) return@forEach
            val o = JSONObject()
            for (k in listOf("phase", "laufS", "beendigung", "waermeStufen", "leistung")) t.opt(k)?.let { o.put(k, it) }
            val log = t.optJSONArray("verlauf")
            log?.optJSONObject(log.length() - 1)?.let { o.put("letzterEintrag", it) }
            ended.put(key, o)
        }
        if (ended.length() > 0) note.put("beendeteTests", ended)
        report.optString("letzterAbsturz").takeIf { it.isNotEmpty() }?.let { note.put("absturz", it.take(3000)) }
        report = JSONObject().put("fruehereVersion", note)
    }

    /** Mit welcher Version der Bericht entstand (ältere Berichte: aus Gerät, KI oder einem Test); leer, wenn er leer ist. */
    private fun reportVersion(r: JSONObject): String {
        r.optString("appVersion").takeIf { it.isNotEmpty() }?.let { return it }
        r.optJSONObject("geraet")?.optString("version")?.takeIf { it.isNotEmpty() }?.let { return it }
        r.optJSONObject("ki")?.optString("version")?.takeIf { it.isNotEmpty() }?.let { return it }
        val tests = r.optJSONObject("tests") ?: return ""
        for (key in tests.keys()) tests.optJSONObject(key)?.optString("version")?.takeIf { it.isNotEmpty() }?.let { return it }
        return ""
    }

    private fun saveAndRender() {
        store.save(report)
        render()
    }

    private fun ki(): JSONObject? = report.optJSONObject("ki")

    private fun quickTestDone(): Boolean = ki()?.has("einzeln") == true && readTests().has("live")

    private fun reportText(): String {
        val o = JSONObject()
        o.put("app", JSONObject().put("name", "CineBo Test").put("stufe", "3").put("version", appVersion).put("erstellt", nowIso()))
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
        hintAfter = null
        store.clear()
        showIdleStatus()
        render()
    }

    // ---------------------------------------------------------------- Ergebnistexte

    private fun summary(key: String): Pair<Mark, String> {
        // Läuft gerade: der Test selbst, oder einer, dessen Zwischenstand jetzt geschrieben wird (der Dauertest beim Strombedarf).
        if (running == key || (running != null && readTests().optJSONObject(key)?.optString("ende") == RUNNING)) return Mark.RUNNING to "läuft …"
        return when (key) {
            "kichip" -> chipSummary()
            "leistung" -> perfSummary()
            "tiefe" -> depthSummary()
            "verfolgung" -> trackSummary()
            "strom" -> powerSummary()
            else -> readTests().optJSONObject(key)?.let { recordingSummary(key, it) }
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

    /**
     * Zusammenfassung einer Aufnahme durch die Pipeline: zuerst das Video (darum geht es in der
     * Abnahme), dann KI und Unschärfe, beim Dauertest Wärme und Akku, zuletzt wo der Clip liegt.
     */
    private fun recordingSummary(key: String, r: JSONObject): Pair<Mark, String> {
        val error = r.optString("fehler")
        if (error.isNotEmpty()) return Mark.FAIL to error
        val ai = r.optJSONObject("ki")
        val aiError = ai?.optString("fehler").orEmpty()
        if (aiError.isNotEmpty()) return Mark.FAIL to aiError
        val file = r.optJSONObject("datei")
        val end = r.optString("ende")
        if (end == KILLED || end == RUNNING) return killedSummary(r, key)
        if (file == null || !file.has("fps")) {
            // Vor der Aufnahme wegen Hitze beendet (etwa beim Strombedarf): ein Ergebnis, kein Fehler der App.
            if (end.startsWith(HEAT_STOP)) return Mark.WARN to end
            // Vor dem Start der Aufnahme gestoppt: kein Fehler, der Test ist nur noch offen.
            if (r.optString("ende") == "gestoppt") return Mark.OPEN to "Gestoppt, bevor die Aufnahme lief"
            val reason = endNote(r) ?: file?.optString("fehler")?.ifEmpty { null } ?: "keine auswertbare Aufnahme"
            return Mark.FAIL to reason.replaceFirstChar { it.uppercase() }
        }
        val parts = mutableListOf<String>()
        var mark = Mark.OK
        if (key == "kidauer" && file.has("dauerS")) parts += "${clock(Math.round(file.optDouble("dauerS")))} min"
        val fps = file.optDouble("fps")
        val lost = lostAfterStart(file)
        parts += "${sizeText(file)} mit ${de(fps, 1)} fps"
        parts += lostText(lost)
        // Abnahme: mindestens 30 Bilder pro Sekunde; 1 % Spielraum für die Zeitstempel der Datei.
        if (fps < FPS * 0.99 || lost > 0) mark = Mark.WARN
        if (key != "vergleich") {
            if (ai != null && ai.has("hz")) {
                val hz = ai.optDouble("hz")
                var lo = hz
                var hi = hz
                val log = r.optJSONArray("verlauf")
                if (key == "kidauer" && log != null && log.length() > 1) {
                    lo = Double.MAX_VALUE
                    hi = 0.0
                    for (i in 0 until log.length()) {
                        val w = log.optJSONObject(i)?.optJSONObject("ki")?.optDouble("hz", Double.NaN) ?: Double.NaN
                        if (w.isNaN()) continue
                        lo = min(lo, w)
                        hi = max(hi, w)
                    }
                    if (lo > hi) {
                        lo = hz
                        hi = hz
                    }
                }
                parts += if (de(lo, 1) == de(hi, 1)) "KI ${de(hz, 1)} Durchgänge pro Sekunde" else "KI ${de(lo, 1)} bis ${de(hi, 1)} Durchgänge pro Sekunde"
                if (lo < MIN_LIVE_HZ) mark = Mark.WARN
            } else {
                parts += "KI lief nicht mit"
                mark = Mark.WARN
            }
            blurNote(r.optJSONObject("pipeline"))?.let { (warn, text) ->
                parts += text
                if (warn) mark = Mark.WARN
            }
            val found = ai?.optInt("objektGefundenProzent", -1) ?: -1
            if (found >= 0) {
                parts += "Motiv in $found % der Durchgänge gefunden"
                if (found < MIN_FOUND_PERCENT) mark = Mark.WARN
            }
            ai?.optJSONObject("waermeregelung")?.let { heat ->
                val waited = heat.optDouble("gewartetS", 0.0)
                if (heat.optInt("gebremsteDurchgaenge") > 0) {
                    parts += "Wärmeregelung ließ die KI insgesamt ${clock(waited.roundToInt().toLong())} min warten"
                }
            }
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
                    parts += "Akkustand ${start.optInt("akkuProzent")} → ${stop.optInt("akkuProzent")} %"
                }
            }
            // Heißeste Stufe aus dem Verlauf (alle 30 s) und aus der Wärmeregelung (alle 2 s).
            var hottest = "keine"
            fun consider(stage: String) {
                if (DeviceHeat.stageOrder.indexOf(stage) > DeviceHeat.stageOrder.indexOf(hottest)) hottest = stage
            }
            val log = r.optJSONArray("verlauf") ?: JSONArray()
            for (i in 0 until log.length()) consider(log.optJSONObject(i)?.optJSONObject("waerme")?.optString("stufe").orEmpty())
            ai?.optJSONObject("waermeregelung")?.optJSONObject("sekundenJeStufe")?.keys()?.forEach { consider(it) }
            parts += "höchste Wärmestufe: $hottest"
            if (DeviceHeat.stageOrder.indexOf(hottest) >= 3) mark = Mark.WARN
            r.optJSONObject("leistung")?.optDouble("watt", Double.NaN)?.takeIf { !it.isNaN() }?.let { parts += "im Mittel ${de(it, 1)} W" }
            trendText(r.optJSONArray("verlauf"))?.let { parts += it }
            // Wie kühl die Aufnahme begann: das Abkühlen direkt davor.
            (r.optJSONObject("abkuehlen2") ?: r.optJSONObject("abkuehlen"))?.let { c ->
                val waited = c.optLong("gewartetS", 0L)
                if (c.optBoolean("nichtAbgekuehlt")) {
                    parts += "nach ${clock(waited)} min Abkühlen noch warm gestartet"
                    mark = Mark.WARN
                } else if (waited > 0) {
                    parts += "nach ${clock(waited)} min Abkühlen gestartet"
                }
            }
        }
        endNote(r)?.let {
            parts += it
            mark = Mark.WARN
        }
        r.optJSONObject("galerie")?.let { g ->
            val failed = g.optString("fehler")
            if (failed.isNotEmpty()) {
                parts += failed
                mark = Mark.WARN
            } else {
                parts += "Clip in der Galerie, Album $GALLERY_FOLDER"
            }
        }
        return mark to parts.joinToString(" · ")
    }

    /**
     * Wie gut die Unschärfe zur KI passte: auf wie vielen Bildern der Hintergrund unscharf wurde
     * und wie viele die KI-Masken davor und danach mischten. Gibt (Warnung, Text) zurück oder
     * null ohne Zahlen.
     */
    private fun blurNote(p: JSONObject?): Pair<Boolean, String>? {
        if (p == null) return null
        val frames = p.optInt("mitUnschaerfeWeg")
        if (frames <= 0) {
            val note = p.optJSONArray("hinweise")?.optString(0).orEmpty()
            return true to if (note.isNotEmpty()) "Keine Unschärfe: ${note.take(120)}" else "Keine Unschärfe"
        }
        val blurred = Math.round(p.optInt("mitMotiv") * 100.0 / frames).toInt()
        val withoutMask = Math.round(p.optInt("ohneMaske") * 100.0 / frames).toInt()
        // Wie gemischtProzent bezogen auf die Bilder, die eine KI-Maske hatten.
        val withMaskFrames = frames - p.optInt("ohneMaske")
        val stale = if (withMaskFrames > 0) Math.round(p.optInt("mitAlterMaske") * 100.0 / withMaskFrames).toInt() else 0
        val mixed = p.optInt("gemischtProzent")
        var text = "Hintergrund auf $blurred % der Bilder unscharf, $mixed % zwischen zwei KI-Masken gemischt"
        if (withoutMask > 5) text += ", $withoutMask % ohne KI-Maske"
        if (stale > 5) text += ", $stale % mit einer KI-Maske älter als 0,4 s"
        val gaps = p.optInt("luecken")
        if (gaps > 0) text += ", ${p.optInt("fehlendeBilderGeschaetzt")} Kamerabilder fehlten der Grafik-Pipeline"
        val failures = p.optInt("ausgabeFehler")
        if (failures > 0) text += ", $failures Bilder ließen sich nicht ausgeben"
        return (withoutMask > 5 || stale > 5 || mixed < 80 || gaps > 0 || failures > 0) to text
    }

    /**
     * Die letzten 3 Minuten des Dauertests: KI-Tempo, Leistung und wie schnell der Akku noch wärmer
     * wird. Zeigt, ob sich das Handy eingependelt hat. Null bei weniger als drei Einträgen.
     */
    private fun trendText(log: JSONArray?): String? {
        if (log == null || log.length() < 3) return null
        val entries = (0 until log.length()).mapNotNull { log.optJSONObject(it) }
        val lastS = entries.lastOrNull()?.optDouble("s", Double.NaN) ?: return null
        if (lastS.isNaN()) return null
        val recent = entries.filter { it.optDouble("s", Double.NaN) >= lastS - TREND_S }
        if (recent.size < 3) return null
        val seconds = DoubleArray(recent.size) { recent[it].optDouble("s", Double.NaN) }
        val temps = DoubleArray(recent.size) { recent[it].optJSONObject("waerme")?.optDouble("akkuGrad", Double.NaN) ?: Double.NaN }
        val hz = Trend.mean(DoubleArray(recent.size) { recent[it].optJSONObject("ki")?.optDouble("hz", Double.NaN) ?: Double.NaN })
        val watts = Trend.mean(DoubleArray(recent.size) { recent[it].optJSONObject("leistung")?.optDouble("watt", Double.NaN) ?: Double.NaN })
        val slope = Trend.slopePerMinute(seconds, temps)
        val parts = mutableListOf<String>()
        if (!hz.isNaN()) parts += "KI ${de(hz, 1)}/s"
        if (!watts.isNaN()) parts += "${de(watts, 1)} W"
        if (!slope.isNaN()) parts += "Akku ${if (slope >= 0) "+" else "−"}${de(kotlin.math.abs(slope), 2)} °C/min"
        return if (parts.isEmpty()) null else "letzte 3 min: " + parts.joinToString(", ")
    }

    /**
     * Ein Test, den das System mittendrin beendet hat: wo und wann (aus dem Zwischenstand),
     * welchen Grund Android nennt, wie heiß es zuletzt war und wie viel Strom das Handy zog.
     */
    private fun killedSummary(r: JSONObject, key: String): Pair<Mark, String> {
        val parts = mutableListOf<String>()
        val seconds = r.optDouble("laufS", Double.NaN)
        val where = if (key == "strom") {
            val list = r.optJSONArray("abschnitte")
            val last = list?.optJSONObject(list.length() - 1)?.optString("abschnitt").orEmpty()
            if (last.isEmpty()) "zu Beginn des Strombedarfs" else "beim Strombedarf nach „${POWER_PHASES[last] ?: last}“"
        } else {
            when (r.optString("phase")) {
                "abkuehlen1" -> "beim Abkühlen vor dem Strombedarf"
                "strom" -> "beim Strombedarf"
                "abkuehlen2" -> "beim Abkühlen vor der Aufnahme"
                "countdown" -> "kurz vor der Aufnahme"
                else -> if (seconds.isNaN()) "gleich zu Beginn der Aufnahme" else "nach ${clock(seconds.toLong())} min Aufnahme"
            }
        }
        parts += "Android hat die App $where beendet"
        r.optJSONObject("beendigung")?.let { e ->
            val why = e.optString("grund")
            val text = e.optString("beschreibung")
            parts += "Grund: $why" + if (text.isNotEmpty()) " ($text)" else ""
        }
        // Zuletzt bekannte Wärme: aus dem Verlauf der Aufnahme, sonst aus dem letzten Abschnitt des Strombedarfs.
        val log = r.optJSONArray("verlauf")
        val sections = r.optJSONArray("abschnitte")
        val heat = log?.optJSONObject(log.length() - 1)?.optJSONObject("waerme")
            ?: sections?.optJSONObject(sections.length() - 1)?.optJSONObject("waerme")
        heat?.let { h ->
            parts += "zuletzt Wärmestufe ${h.optString("stufe")}" + if (h.has("akkuGrad")) ", Akku ${de(h.optDouble("akkuGrad"), 1)} °C" else ""
        }
        r.optJSONObject("leistung")?.optDouble("watt", Double.NaN)?.takeIf { !it.isNaN() }?.let { parts += "im Mittel ${de(it, 1)} W" }
        return Mark.FAIL to parts.joinToString(" · ")
    }

    /**
     * Strombedarf als eine Zeile: Grundlast mit Bildschirm, dann was jeder Baustein dazulegt
     * (gegenüber dem Abschnitt davor), 4K mit Unschärfe und KI zusammen, und ob die 4K-Aufnahme
     * am Ende mehr brauchte als am Anfang (Erwärmung).
     */
    private fun powerSummary(): Pair<Mark, String> {
        val r = readTests().optJSONObject("strom") ?: return openRow("strom")
        r.optString("fehler").takeIf { it.isNotEmpty() }?.let { return Mark.WARN to it }
        val end = r.optString("ende")
        if (end == KILLED || end == RUNNING) return killedSummary(r, "strom")
        val phases = r.optJSONArray("abschnitte") ?: return openRow("strom")
        val byKey = HashMap<String, JSONObject>()
        for (i in 0 until phases.length()) phases.optJSONObject(i)?.let { byKey[it.optString("abschnitt")] = it }
        // Nur vollständig gemessene Abschnitte zählen.
        fun watt(key: String): Double =
            byKey[key]?.takeIf { !it.has("fehler") && !it.has("ende") }?.optDouble("watt", Double.NaN) ?: Double.NaN
        fun shaky(key: String) = if (byKey[key]?.optBoolean("schwankt") == true) " (schwankte)" else ""
        fun plus(v: Double) = (if (v >= 0) "+" else "−") + de(kotlin.math.abs(v), 1) + " W"
        val parts = mutableListOf<String>()
        val base = watt("aus")
        if (!base.isNaN()) parts += "Grundlast mit Bildschirm ${de(base, 1)} W${shaky("aus")}"
        // Abschnitt, Bezug, Text
        for ((key, ref, title) in listOf(
            Triple("kamera", "aus", "Kamera"),
            Triple("aufnahme", "kamera", "4K-Aufnahme"),
            Triple("unschaerfe", "aufnahme", "Unschärfe"),
            Triple("kibilder", "unschaerfe", "KI-Bilder"),
            Triple("ki", "kibilder", "KI 5/s"),
            Triple("kiTiefe", "ki", "Tiefe bei jedem Durchgang"),
            Triple("spitze", "ki", "Spitzenleistung statt Dauerleistung"),
        )) {
            val w = watt(key)
            val b = watt(ref)
            if (!w.isNaN() && !b.isNaN()) parts += "$title ${plus(w - b)}${shaky(key)}"
        }
        val ai = watt("ki")
        if (!ai.isNaN()) parts += "4K mit Unschärfe und KI zusammen ${de(ai, 1)} W"
        val rec = watt("aufnahme")
        val rec2 = watt("aufnahme2")
        if (!rec.isNaN() && !rec2.isNaN()) parts += "4K am Ende ${plus(rec2 - rec)}"
        byKey["ki"]?.optDouble("kiProSekunde", Double.NaN)?.takeIf { !it.isNaN() && it < 4.5 }?.let { parts += "KI schaffte nur ${de(it, 1)}/s" }
        val errors = POWER_PHASES.keys.mapNotNull { k ->
            byKey[k]?.optString("fehler")?.takeIf { it.isNotEmpty() }?.let { "${POWER_PHASES[k]}: $it" }
        }
        if (errors.isNotEmpty()) parts += ("Fehler bei " + errors.joinToString("; ")).take(240)
        if (end.startsWith(HEAT_STOP) || end == "gestoppt") parts += end
        if (parts.isEmpty()) return Mark.WARN to "Keine Messwerte: Das Handy meldet keinen Strom"
        val complete = POWER_PHASES.keys.all { !watt(it).isNaN() }
        return (if (complete && errors.isEmpty()) Mark.OK else Mark.WARN) to parts.joinToString(" · ")
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
                "Leistungsstufe und misst Tempo und Genauigkeit. Dann filmt er 30 Sekunden in 4K mit Unschärfe und " +
                "10 Sekunden ohne, zum Vergleich; beide Clips kommen in die Galerie. Beim ersten Mal übersetzt der Chip " +
                "jedes Modell neu; rechne mit etwa zehn Minuten, danach etwa acht.$noMic"
            repeatReason("live") != null -> repeatReason("live")!!
            repeatReason("vergleich") != null -> repeatReason("vergleich")!!
            !tests.has("vergleich") -> "Der Vergleichsclip fehlt noch. Starte den KI-Schnelltest bitte noch einmal und lass die App dabei offen."
            !tests.has("kidauer") -> "Jetzt der KI-Dauertest, ganz ohne Zutun, etwa 25 bis 40 Minuten: Er wartet, bis das Handy " +
                "abgekühlt ist, misst gut 4 Minuten den Strombedarf, kühlt wieder ab und filmt dann 10 Minuten 4K mit " +
                "Unschärfe. Vorher: Ladekabel ab, Akku mindestens 40 %, Energiesparmodus aus, „Nicht stören“ an. Stell das " +
                "Handy so hin, dass die Kamera auf eine Person oder einen Gegenstand in der Mitte zeigt, mit etwas Abstand " +
                "zum Hintergrund, und lass die App offen.$noMic"
            repeatReason("kidauer") != null -> repeatReason("kidauer")!!
            else -> "Alles gemessen. Kopiere den Bericht und füge ihn bei mir im Chat ein. Die Clips liegen in der " +
                "Galerie im Album $GALLERY_FOLDER. In der KI-Vorschau kannst du die Unschärfe weiter ausprobieren."
        }
    }

    /** Warum ein Test noch einmal laufen sollte, oder null. */
    private fun repeatReason(key: String): String? {
        val r = readTests().optJSONObject(key) ?: return null
        val end = r.optString("ende")
        if (end == KILLED || end == RUNNING) {
            return "Android hat die App während des Tests beendet. Kopiere den Bericht und füge ihn bei mir im Chat ein; der Grund steht darin."
        }
        if (end.startsWith(HEAT_STOP)) {
            return "Die App hat den Test wegen Hitze selbst beendet. Das ist schon ein Ergebnis: Kopiere den Bericht und füge ihn bei mir im Chat ein. " +
                "Lass das Handy danach ein paar Minuten abkühlen."
        }
        if (recordingSummary(key, r).first == Mark.FAIL) {
            return "Eine Aufnahme hat nicht geklappt (siehe Ergebnis). Kopiere den Bericht und füge ihn bei mir im Chat ein."
        }
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
        hintText.text = hintOverride ?: hintAfter ?: defaultHint(quickDone)
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

        // Bühne: das Kamerabild, wie es aufgenommen wird; Antippen wählt das Motiv
        previewBox = FrameLayout(this)
        val cameraPanel = panel()
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.BLACK)
        }
        cameraPanel.addView(previewView, FrameLayout.LayoutParams(MATCH, MATCH))
        tapOverlay = TapOverlay(this)
        cameraPanel.addView(tapOverlay, FrameLayout.LayoutParams(MATCH, MATCH))
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
        stage = StageLayout(this, cameraPanel)
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
            text = "KI-Dauertest"
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
