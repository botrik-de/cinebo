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

/** Wie ein Modell auf dem KI-Chip rechnen soll. */
internal data class NpuConfig(val fp16: Boolean, val burst: Boolean = false)

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
        interpreter.close()
        delegate?.close()
    }

    companion object {
        const val FOLDER = "modelle"

        fun load(context: Context, asset: String, npu: NpuConfig?, threads: Int = 4): NpuModel {
            val wallStart = System.currentTimeMillis()
            val start = SystemClock.elapsedRealtimeNanos()
            val model = map(context, "$FOLDER/$asset")
            val options = Interpreter.Options()
            options.setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_APPLICATION_ONLY)
            options.setUseNNAPI(false)
            options.setUseXNNPACK(true)
            options.setNumThreads(threads)
            var delegate: QnnDelegate? = null
            var cached = false
            if (npu != null) {
                val token = Npu.token(context, asset, npu)
                cached = Npu.cacheDir(context).listFiles()?.any { it.name.contains(token) } == true
                delegate = Npu.delegate(context, npu, token)
                options.addDelegate(delegate)
            }
            val interpreter = try {
                Interpreter(model, options).also { it.allocateTensors() }
            } catch (e: Throwable) {
                delegate?.close()
                throw e
            }
            val ms = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
            val log = LogProbe.since(wallStart)
            return NpuModel(asset, npu, interpreter, delegate, ms, cached, log)
        }

        private fun map(context: Context, path: String): MappedByteBuffer {
            context.assets.openFd(path).use { fd ->
                FileInputStream(fd.fileDescriptor).use { stream ->
                    return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
        }
    }
}

/** Zugang zum KI-Chip über Qualcomms QNN-Delegate. */
internal object Npu {

    fun cacheDir(context: Context): File = File(context.cacheDir, "qnn").apply { mkdirs() }

    private fun versionCode(context: Context): Long = try {
        context.packageManager.getPackageInfo(context.packageName, PackageManager.PackageInfoFlags.of(0)).longVersionCode
    } catch (_: Exception) {
        0L
    }

    /**
     * Name, unter dem QNN das fertig übersetzte Modell zwischenspeichert. Er enthält die
     * App-Version, damit eine neue Test-App nie ein altes, anders gebautes Modell lädt.
     */
    fun token(context: Context, asset: String, cfg: NpuConfig): String =
        asset.substringBeforeLast('.') + "-v" + versionCode(context) +
            (if (cfg.fp16) "-fp16" else "-q") + (if (cfg.burst) "-burst" else "-dauer")

    /** Löscht Zwischenspeicher früherer App-Versionen. */
    fun cleanCache(context: Context) {
        val mine = "-v" + versionCode(context) + "-"
        cacheDir(context).listFiles()?.forEach { if (!it.name.contains(mine)) it.deleteRecursively() }
    }

    fun delegate(context: Context, cfg: NpuConfig, token: String): QnnDelegate {
        val o = QnnDelegate.Options()
        o.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
        o.setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
        o.setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_WARN)
        o.setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
        o.setHtpPerformanceMode(
            if (cfg.burst) QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST
            else QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE
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
        try {
            o.put("qnnVersion", QnnDelegate.getVersion().joinToString("."))
            o.put("htpFp16", QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16))
            o.put("htpQuantisiert", QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED))
            o.put("dsp", QnnDelegate.checkCapability(QnnDelegate.Capability.DSP_RUNTIME))
        } catch (e: Throwable) {
            o.put("fehler", e.toString())
        }
        val libs = JSONArray()
        File(context.applicationInfo.nativeLibraryDir).listFiles()?.sortedBy { it.name }?.forEach {
            libs.put("${it.name} ${r1(it.length() / 1e6)} MB")
        }
        o.put("bibliotheken", libs)
        return o
    }
}

/** Liest Meldungen dieser App aus dem System-Log, etwa wie viel eines Modells auf dem Chip läuft. */
internal object LogProbe {

    private val KEYWORDS = Regex("(?i)qnn|htp|delegat|partition|replacing|tflite|hexagon|fastrpc|cdsp|skel")
    private val REPLACING = Regex("Replacing (\\d+) out of (\\d+) node\\(s\\) with delegate \\((\\w+)\\) node, yielding (\\d+) partition")

    /** Passende Zeilen seit [sinceWallMs] (Wanduhr in ms). */
    fun since(sinceWallMs: Long, max: Int = 40): JSONArray {
        val out = JSONArray()
        var process: java.lang.Process? = null
        try {
            process = ProcessBuilder("logcat", "-d", "-v", "epoch", "--pid=${Process.myPid()}")
                .redirectErrorStream(true)
                .start()
            process.inputStream.bufferedReader().useLines { lines ->
                for (line in lines) {
                    if (out.length() >= max) continue // weiterlesen, damit logcat nicht hängt
                    val t = line.trimStart().substringBefore(' ').toDoubleOrNull() ?: continue
                    if (t * 1000 < sinceWallMs - 50) continue
                    if (KEYWORDS.containsMatchIn(line)) out.put(line.trim().take(300))
                }
            }
            process.waitFor(2, TimeUnit.SECONDS)
        } catch (e: Exception) {
            out.put("Log nicht lesbar: ${e.message}")
        } finally {
            process?.destroy()
        }
        return out
    }

    /** Fasst die TFLite-Zeile „Replacing x out of y node(s) …“ zusammen, falls sie da ist. */
    fun delegation(log: JSONArray): JSONObject? {
        for (i in 0 until log.length()) {
            val m = REPLACING.find(log.optString(i)) ?: continue
            val (done, total, name, parts) = m.destructured
            if (!name.contains("Qnn", ignoreCase = true)) continue
            return JSONObject()
                .put("schritte", done.toInt())
                .put("vonSchritten", total.toInt())
                .put("teile", parts.toInt())
        }
        return null
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
