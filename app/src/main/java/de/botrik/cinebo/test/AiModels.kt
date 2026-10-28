package de.botrik.cinebo.test

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Process
import android.os.SystemClock
import android.system.Os
import android.system.OsConstants
import com.qualcomm.qti.QnnDelegate
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.InterpreterApi
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.TimeUnit

/** Leistungsstufe, mit der QNN den KI-Chip taktet. */
internal enum class PerfMode(val key: String, val title: String, val htp: QnnDelegate.Options.HtpPerformanceMode) {
    SUSTAINED("dauer", "Dauerleistung", QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE),
    HIGH("hoch", "hohe Leistung", QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_HIGH_PERFORMANCE),
    BURST("spitze", "Spitzenleistung", QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST),
}

/** Wie gründlich QNN ein Modell beim ersten Laden für den Chip übersetzt. */
internal enum class OptLevel(val key: String, val title: String, val htp: QnnDelegate.Options.HtpOptimizationStrategy) {
    STANDARD("o2", "Standard", QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_INFERENCE),
    O3("o3", "O3", QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_INFERENCE_O3),
}

/**
 * Einstellung des KI-Chips: Leistungsstufe, Übersetzung und ob die App den Takt selbst festhält
 * ([manualVote]: QNN regelt dann nicht nach jedem Lauf zurück, siehe [NpuModel.vote]).
 */
internal data class ChipSetting(val perf: PerfMode, val opt: OptLevel, val manualVote: Boolean = false) {
    val key: String get() = perf.key + "-" + opt.key + if (manualVote) "-fest" else ""
    val title: String get() = "${perf.title.replaceFirstChar { it.uppercase() }}, ${opt.title}" + if (manualVote) ", Takt festgehalten" else ""

    /**
     * Für live (Vorschau und Aufnahmen): Dauerleistung mit derselben Übersetzung, ohne
     * festgehaltenen Takt. Die Spitzenleistung ist rund 18 % schneller, ließ das Handy in 0.3.15
     * aber nach 3–4 Minuten Dauertest überhitzen.
     */
    fun forLive(): ChipSetting = ChipSetting(PerfMode.SUSTAINED, opt)

    companion object {
        /** So lief Stufe 2. */
        val DEFAULT = ChipSetting(PerfMode.SUSTAINED, OptLevel.STANDARD)

        /** Was der Leistungstest misst; danach noch die beste mit festgehaltenem Takt. */
        val CANDIDATES = OptLevel.entries.flatMap { opt -> PerfMode.entries.map { ChipSetting(it, opt) } }

        fun fromKey(key: String?): ChipSetting? {
            if (key.isNullOrEmpty()) return null
            val parts = key.split('-')
            val perf = PerfMode.entries.firstOrNull { it.key == parts.getOrNull(0) } ?: return null
            val opt = OptLevel.entries.firstOrNull { it.key == parts.getOrNull(1) } ?: return null
            return ChipSetting(perf, opt, parts.getOrNull(2) == "fest")
        }
    }
}

/** Wie ein Modell auf dem KI-Chip rechnen soll. */
internal data class NpuConfig(val fp16: Boolean, val chip: ChipSetting = ChipSetting.DEFAULT)

internal fun directBuffer(bytes: Int): ByteBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())

/**
 * Ein TFLite-Modell aus den App-Daten. Mit [NpuConfig] rechnet es über Qualcomms
 * QNN-Delegate auf dem KI-Chip (Hexagon), ohne auf der CPU (XNNPACK).
 * Nur von einem Thread aus benutzen.
 */
internal class NpuModel private constructor(
    val asset: String,
    val npu: NpuConfig?,
    private val interpreter: Interpreter,
    private val delegate: QnnDelegate?,
    val loadMs: Double,
    val fromCache: Boolean,
    val log: JSONArray,
) : AutoCloseable {

    private val inputNames = List(interpreter.inputTensorCount) { interpreter.getInputTensor(it).name() }
    private val outputNames = List(interpreter.outputTensorCount) { interpreter.getOutputTensor(it).name() }

    val inputCount: Int get() = inputNames.size

    /** Rechenzeit des letzten Laufs im Modell selbst, ohne das Kopieren der Ein- und Ausgaben. */
    var lastNativeMs = 0.0
        private set

    fun inputIndex(name: String): Int = inputNames.indexOf(name).also {
        require(it >= 0) { "$asset hat keinen Eingang $name, nur $inputNames" }
    }

    fun outputIndex(name: String): Int = outputNames.indexOf(name).also {
        require(it >= 0) { "$asset hat keinen Ausgang $name, nur $outputNames" }
    }

    fun newInput(i: Int): ByteBuffer = directBuffer(interpreter.getInputTensor(i).numBytes())

    fun newOutput(i: Int): ByteBuffer = directBuffer(interpreter.getOutputTensor(i).numBytes())

    fun inputBytes(i: Int): Int = interpreter.getInputTensor(i).numBytes()

    fun outputFloats(i: Int): Int = interpreter.getOutputTensor(i).numElements()

    val outputCount: Int get() = outputNames.size

    fun outputShape(i: Int): IntArray = interpreter.getOutputTensor(i).shape()

    fun inputShape(i: Int): IntArray = interpreter.getInputTensor(i).shape()

    /** Erster Ausgang mit genau dieser Form; die Namen vergibt der Umwandler, die Form legen wir fest. */
    fun outputIndexByShape(vararg shape: Int): Int = (0 until outputCount).firstOrNull { outputShape(it).contentEquals(shape) }
        ?: throw IllegalArgumentException("$asset hat keinen Ausgang der Form ${shape.joinToString("x")}")

    /** Skala und Nullpunkt eines quantisierten Eingangs (UINT8): Wert = (q - Nullpunkt) · Skala. */
    fun inputQuant(i: Int): Pair<Float, Int> = interpreter.getInputTensor(i).quantizationParams().let { it.scale to it.zeroPoint }

    fun outputQuant(i: Int): Pair<Float, Int> = interpreter.getOutputTensor(i).quantizationParams().let { it.scale to it.zeroPoint }

    fun isQuantizedInput(i: Int): Boolean = interpreter.getInputTensor(i).dataType() == org.tensorflow.lite.DataType.UINT8

    /**
     * Hält den Takt des KI-Chips fest, bis [release] kommt. Nur mit [ChipSetting.manualVote]
     * nötig; sonst regelt QNN selbst und das hier tut nichts.
     */
    fun vote() {
        if (npu?.chip?.manualVote != true || voted) return
        delegate?.performanceVote()
        voted = true
    }

    fun release() {
        if (!voted) return
        voted = false
        delegate?.performanceRelease()
    }

    private var voted = false

    /**
     * Rechnet einmal und gibt die Zeit in ms zurück, mit Kopieren. Eingänge sind direkte
     * ByteBuffer in Speicher-Reihenfolge des Geräts, je Eingang einer in Modell-Reihenfolge.
     * Ausgänge, die nicht in [outputs] stehen, liest man danach mit [output].
     */
    fun run(inputs: Array<Any>, outputs: Map<Int, Any> = emptyMap()): Double {
        val start = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        val ms = (System.nanoTime() - start) / 1e6
        lastNativeMs = interpreter.lastNativeInferenceDurationNanoseconds?.let { it / 1e6 } ?: ms
        for (o in outputs.values) (o as? ByteBuffer)?.rewind()
        return ms
    }

    /** Sicht auf einen Ausgang ohne Kopie. Gilt nur bis zum nächsten Lauf dieses Modells. */
    fun output(i: Int): ByteBuffer = interpreter.getOutputTensor(i).asReadOnlyBuffer()

    fun describe(): JSONObject {
        val o = JSONObject()
        o.put("datei", asset)
        o.put("wo", if (npu == null) "CPU" else "KI-Chip")
        if (npu != null) o.put("genauigkeit", if (npu.fp16) "fp16" else "quantisiert")
        if (npu != null) o.put("einstellung", npu.chip.key)
        o.put("ladenMs", r1(loadMs))
        if (npu != null) o.put("ausZwischenspeicher", fromCache)
        val ins = JSONArray()
        for (i in inputNames.indices) {
            val t = interpreter.getInputTensor(i)
            ins.put("${t.name()} ${t.shape().joinToString("x")} ${t.dataType()}")
        }
        o.put("eingaenge", ins)
        val outs = JSONArray()
        for (i in outputNames.indices) {
            val t = interpreter.getOutputTensor(i)
            outs.put("${t.name()} ${t.shape().joinToString("x")} ${t.dataType()}")
        }
        o.put("ausgaenge", outs)
        if (log.length() > 0) o.put("protokoll", log)
        LogProbe.delegation(log)?.let { o.put("aufDemChip", it) }
        return o
    }

    override fun close() {
        release()
        interpreter.close()
        delegate?.close()
    }

    companion object {

        fun load(context: Context, asset: String, npu: NpuConfig?, threads: Int = 4): NpuModel {
            val wallStart = System.currentTimeMillis()
            val start = SystemClock.elapsedRealtimeNanos()
            val model = map(File(File(context.filesDir, ModelStore.FOLDER), asset))
            val options = Interpreter.Options()
            options.setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_APPLICATION_ONLY)
            options.setUseNNAPI(false)
            options.setUseXNNPACK(true)
            options.setNumThreads(threads)
            var delegate: QnnDelegate? = null
            val before = if (npu != null) Npu.cacheDir(context).list()?.toSet().orEmpty() else emptySet()
            if (npu != null) {
                delegate = Npu.delegate(context, npu, Npu.token(context, asset, npu))
                options.addDelegate(delegate)
            }
            val interpreter = try {
                Interpreter(model, options).also { it.allocateTensors() }
            } catch (e: Throwable) {
                delegate?.close()
                throw e
            }
            val ms = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
            // Aus dem Zwischenspeicher, wenn QNN dabei keine neue übersetzte Datei angelegt hat.
            val added = if (npu != null) Npu.cacheDir(context).list()?.filter { it !in before }.orEmpty() else emptyList()
            val cached = npu != null && added.none { it.startsWith("qnn_binary") }
            val log = LogProbe.since(wallStart)
            return NpuModel(asset, npu, interpreter, delegate, ms, cached, log)
        }

        /** Die Modelle liegen nach dem Laden im App-Speicher (siehe [ModelStore]). */
        private fun map(file: File): MappedByteBuffer {
            if (!file.exists()) throw IllegalStateException("Modell fehlt: ${file.name}. Die KI-Modelle sind noch nicht heruntergeladen.")
            FileInputStream(file).use { stream ->
                return stream.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
            }
        }
    }
}

/**
 * Zugang zum KI-Chip über Qualcomms QNN-Delegate. In der APK steckt nur der kleine Delegate;
 * die große QNN-Laufzeit (Laufzeit, Übersetzer, Stub und Skel für den Hexagon-Chip) holt
 * [ModelStore] direkt aus Qualcomms Paket in den App-Speicher. Vor dem ersten Gebrauch lädt
 * [loadLibraries] sie, damit QNN sie unter ihrem Namen findet.
 */
internal object Npu {

    /**
     * Vorab geladen, damit QNN sie in jedem Fall findet: libQnnSystem.so und den Stub laden der
     * Delegate und libQnnHtp.so womöglich nur unter ihrem Namen, und den kennt der Linker nur für
     * schon geladene Bibliotheken oder solche aus der APK. Die große libQnnHtpPrepare.so
     * (Übersetzer für den Chip) holt libQnnHtp.so selbst aus ihrem eigenen Ordner, und nur solange
     * sie sie braucht; vorab geladen bliebe sie dauerhaft im Speicher. Der Skel läuft auf dem Chip
     * und kommt über [QnnDelegate.Options.setSkelLibraryDir] dorthin.
     */
    private val LIBRARIES = listOf(
        "libQnnSystem.so", "libQnnHtpV81Stub.so", "libQnnHtpV79Stub.so", "libQnnHtp.so"
    )

    /** Nur Rückfall, falls sich der Chip als Hexagon V79 meldet; ein Fehler damit hält nichts auf. */
    private const val FALLBACK_STUB = "libQnnHtpV79Stub.so"

    @Volatile private var librariesLoaded = false
    @Volatile private var fallbackError: String? = null

    private fun libDir(context: Context): File = File(context.filesDir, ModelStore.FOLDER)

    /** Lädt die QNN-Bibliotheken aus dem App-Speicher, einmal je Prozess. Wirft bei Fehlern. */
    @Synchronized
    fun loadLibraries(context: Context) {
        if (librariesLoaded) return
        val dir = libDir(context)
        for (name in LIBRARIES) {
            val f = File(dir, name)
            val optional = name == FALLBACK_STUB
            if (!f.isFile) {
                if (optional) continue
                throw IllegalStateException("QNN-Bibliothek fehlt: $name. Die KI-Modelle sind noch nicht vollständig heruntergeladen.")
            }
            try {
                f.setReadOnly() // die geprüfte Datei soll niemand mehr unbemerkt ändern
                System.load(f.absolutePath)
            } catch (e: Throwable) {
                if (!optional) throw e
                fallbackError = e.toString().take(300)
            }
        }
        librariesLoaded = true
    }

    /**
     * Fertig übersetzte Modelle, ein Ordner je App-Version. Im App-Speicher statt im Cache-Ordner,
     * den Android jederzeit leeren darf: sonst übersetzt QNN unbemerkt neu. QNN legt dort zu
     * jedem Modell eine kleine Datei mit unserem Namen und die große qnn_binary_….bin ohne ihn ab;
     * deshalb räumt [cleanCache] nur ganze Ordner früherer Versionen weg.
     */
    fun cacheDir(context: Context): File = File(cacheRoot(context), "v" + versionCode(context)).apply { mkdirs() }

    private fun cacheRoot(context: Context): File = File(context.filesDir, "qnn-cache")

    private fun versionCode(context: Context): Long = try {
        context.packageManager.getPackageInfo(context.packageName, PackageManager.PackageInfoFlags.of(0)).longVersionCode
    } catch (_: Exception) {
        0L
    }

    /**
     * Name, unter dem QNN das fertig übersetzte Modell zwischenspeichert. Er enthält die
     * App-Version, damit eine neue Test-App nie ein altes, anders gebautes Modell lädt, und die
     * Übersetzungsstufe. Die Leistungsstufe ändert nichts an der Übersetzung und fehlt deshalb.
     */
    fun token(context: Context, asset: String, cfg: NpuConfig): String =
        asset.substringBeforeLast('.') + "-v" + versionCode(context) +
            (if (cfg.fp16) "-fp16" else "-q") + "-" + cfg.chip.opt.key + "-"

    /** Löscht die Zwischenspeicher früherer App-Versionen und den alten Ort aus Stufe 2. */
    fun cleanCache(context: Context) {
        val mine = cacheDir(context)
        cacheRoot(context).listFiles()?.forEach { if (it != mine) it.deleteRecursively() }
        File(context.cacheDir, "qnn").deleteRecursively()
    }

    fun delegate(context: Context, cfg: NpuConfig, token: String): QnnDelegate {
        loadLibraries(context)
        val o = QnnDelegate.Options()
        o.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
        o.setLibraryPath(File(libDir(context), "libQnnHtp.so").absolutePath)
        o.setSkelLibraryDir(libDir(context).absolutePath)
        o.setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_WARN)
        o.setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
        o.setHtpPerformanceMode(cfg.chip.perf.htp)
        o.setHtpOptimizationStrategy(cfg.chip.opt.htp)
        o.setHtpPerfCtrlStrategy(
            if (cfg.chip.manualVote) QnnDelegate.Options.HtpPerfCtrlStrategy.HTP_PERF_CTRL_MANUAL
            else QnnDelegate.Options.HtpPerfCtrlStrategy.HTP_PERF_CTRL_AUTO
        )
        o.setHtpPrecision(
            if (cfg.fp16) QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16
            else QnnDelegate.Options.HtpPrecision.HTP_PRECISION_QUANTIZED
        )
        o.setCacheDir(cacheDir(context).absolutePath)
        o.setModelToken(token)
        return QnnDelegate(o)
    }

    /** Was über den KI-Chip und die Bibliotheken bekannt ist. Wirft nie. */
    fun info(context: Context): JSONObject {
        val o = JSONObject()
        o.put("geraet", "${Build.MANUFACTURER} ${Build.MODEL}")
        o.put("chip", "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}")
        o.put("android", "${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})")
        o.put("build", Build.FINGERPRINT)
        o.put("seitengroesseBytes", Os.sysconf(OsConstants._SC_PAGESIZE))
        try {
            val memory = ActivityManager.MemoryInfo()
            context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
            o.put("arbeitsspeicherGB", r1(memory.totalMem / 1e9))
        } catch (_: Exception) {
        }
        o.put("qnnSdk", BuildConfig.QNN_VERSION)
        try {
            loadLibraries(context)
            // checkCapability lädt die JNI-Bibliothek des Delegates; getVersion tut das nicht und
            // scheitert davor (Stufe 2b: UnsatisfiedLinkError). Darum erst die Fähigkeiten.
            o.put("htpFp16", QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16))
            o.put("htpQuantisiert", QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED))
            o.put("dsp", QnnDelegate.checkCapability(QnnDelegate.Capability.DSP_RUNTIME))
        } catch (e: Throwable) {
            o.put("fehler", e.toString())
        }
        // Nur eine Angabe für den Bericht: Scheitert sie, ist der Chip deshalb nicht unerreichbar.
        try {
            o.put("delegateApi", QnnDelegate.getVersion().joinToString("."))
        } catch (e: Throwable) {
            o.put("delegateApiFehler", e.toString().take(200))
        }
        val libs = JSONArray()
        File(context.applicationInfo.nativeLibraryDir).listFiles()?.sortedBy { it.name }?.forEach {
            libs.put("${it.name} ${r1(it.length() / 1e6)} MB")
        }
        libDir(context).listFiles()?.filter { it.name.endsWith(".so") }?.sortedBy { it.name }?.forEach {
            libs.put("${it.name} ${r1(it.length() / 1e6)} MB (nachgeladen)")
        }
        o.put("bibliotheken", libs)
        fallbackError?.let { o.put("rueckfallStubFehler", it) }
        return o
    }
}

/** Liest Meldungen dieser App aus dem System-Log, etwa wie viel eines Modells auf dem Chip läuft. */
internal object LogProbe {

    private val KEYWORDS = Regex("(?i)qnn|htp|delegat|partition|replacing|tflite|hexagon|fastrpc|cdsp|skel")
    private val REPLACING = Regex("Replacing (\\d+) out of (\\d+) node\\(s\\) with delegate \\((\\w+)\\) node, yielding (\\d+) partition")

    /** Info-Zeilen der FastRPC-Schicht: zahlreich, aber für den Bericht ohne Wert. */
    private val NOISE = Regex("^\\S+\\s+\\d+\\s+\\d+ I .*fastrpc", RegexOption.IGNORE_CASE)

    /**
     * Passende Zeilen seit [sinceWallMs] (Wanduhr in ms). Höchstens [max] Zeilen; reicht der Platz
     * nicht, bleiben die letzten, denn dort steht, wie viel eines Modells auf dem Chip läuft.
     */
    fun since(sinceWallMs: Long, max: Int = 60): JSONArray {
        val kept = ArrayDeque<String>()
        var dropped = 0
        var process: java.lang.Process? = null
        try {
            process = ProcessBuilder("logcat", "-d", "-v", "epoch", "--pid=${Process.myPid()}")
                .redirectErrorStream(true)
                .start()
            process.inputStream.bufferedReader().useLines { lines ->
                for (line in lines) {
                    val t = line.trimStart().substringBefore(' ').toDoubleOrNull() ?: continue
                    // Ohne Vorlauf: Zeilen davor gehören zum vorher geladenen Modell (1 ms für Rundung).
                    if (t * 1000 < sinceWallMs - 1) continue
                    if (!KEYWORDS.containsMatchIn(line) || NOISE.containsMatchIn(line.trim())) continue
                    kept.addLast(line.trim().take(300))
                    if (kept.size > max) {
                        kept.removeFirst()
                        dropped++
                    }
                }
            }
            process.waitFor(2, TimeUnit.SECONDS)
        } catch (e: Exception) {
            kept.addLast("Log nicht lesbar: ${e.message}")
        } finally {
            process?.destroy()
        }
        val out = JSONArray()
        if (dropped > 0) out.put("($dropped ältere Zeilen ausgelassen)")
        kept.forEach { out.put(it) }
        return out
    }

    /**
     * Fasst die TFLite-Zeile „Replacing x out of y node(s) …“ zusammen. Es zählt die LETZTE
     * QNN-Zeile: Die eigene schreibt TFLite erst am Ende des Ladens, und so bleibt das Ergebnis
     * richtig, selbst wenn eine Zeile des vorher geladenen Modells mit hineinrutscht.
     */
    fun delegation(log: JSONArray): JSONObject? {
        var result: JSONObject? = null
        for (i in 0 until log.length()) {
            val m = REPLACING.find(log.optString(i)) ?: continue
            val (done, total, name, parts) = m.destructured
            if (!name.contains("Qnn", ignoreCase = true)) continue
            result = JSONObject()
                .put("schritte", done.toInt())
                .put("vonSchritten", total.toInt())
                .put("aufCpu", total.toInt() - done.toInt())
                .put("teile", parts.toInt())
        }
        return result
    }
}

/** Kennzahlen einer Reihe von Zeiten in ms. */
internal class Timing(samples: DoubleArray) {
    val runs = samples.size
    private val sorted = samples.sortedArray()
    val median: Double = if (runs == 0) 0.0 else quantile(0.5)
    val p90: Double = if (runs == 0) 0.0 else quantile(0.9)
    val min: Double = sorted.firstOrNull() ?: 0.0
    val max: Double = sorted.lastOrNull() ?: 0.0

    private fun quantile(q: Double): Double {
        val pos = q * (runs - 1)
        val lo = pos.toInt()
        val hi = minOf(lo + 1, runs - 1)
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
    }

    fun toJson(): JSONObject = JSONObject()
        .put("medianMs", r2(median))
        .put("p90Ms", r2(p90))
        .put("minMs", r2(min))
        .put("maxMs", r2(max))
        .put("laeufe", runs)
}
