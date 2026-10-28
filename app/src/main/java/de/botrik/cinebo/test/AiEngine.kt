package de.botrik.cinebo.test

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Debug
import android.os.SystemClock
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.util.EnumMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import java.util.zip.ZipFile
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Alle Modelle der Test-App. */
internal enum class ModelId(val asset: String, val title: String, val fp16: Boolean) {
    DEPTH("depth_anything_v2.tflite", "Tiefe · Depth Anything V2 Small", true),
    ENC("edgetam_encoder.tflite", "EdgeTAM · Encoder", true),
    DEC("edgetam_decoder.tflite", "EdgeTAM · Decoder", true),
    DEC_META("edgetam_decoder_meta.tflite", "EdgeTAM · Decoder wie im Original (eigener Export)", true),
    ATTN("edgetam_memory_attention.tflite", "EdgeTAM · Memory-Attention", true),
    ATTN2("edgetam_memory_attention_v2.tflite", "EdgeTAM · Memory-Attention, chipfreundlich exportiert", true),
    MEMENC("edgetam_memory_encoder.tflite", "EdgeTAM · Memory-Encoder (eigener Export)", true),
    MEMPREP("edgetam_memory_prep.tflite", "EdgeTAM · Vorbereitung für Qualcomms Memory-Encoder", true),
    MEMENC_QC("edgetam_memory_encoder_qualcomm.tflite", "EdgeTAM · Memory-Encoder von Qualcomm", true),
    ENC_Q("edgetam_encoder_w8a8.tflite", "EdgeTAM · Encoder quantisiert", false),
    DEC_Q("edgetam_decoder_w8a8.tflite", "EdgeTAM · Decoder quantisiert", false),
    MEMENC_Q("edgetam_memory_encoder_w8a8.tflite", "EdgeTAM · Memory-Encoder quantisiert", false);

    val key: String get() = name.lowercase()

    companion object {
        /** Was der Leistungstest in jeder Einstellung misst: alles, was in einer Verfolgung laufen kann. */
        val MATRIX = listOf(DEPTH, ENC, ENC_Q, DEC, DEC_META, ATTN, ATTN2, MEMENC, MEMPREP, MEMENC_QC)
    }
}

/**
 * Lädt die Modelle auf den KI-Chip, vergleicht die Einstellungen des Chips, misst die Modelle
 * einzeln und als Ketten, prüft die Genauigkeit gegen die CPU und gegen Metas Original und lässt
 * Tiefe und Verfolgung auf den Kamerabildern laufen. Alles läuft auf einem eigenen Thread („ki“);
 * die Methoden hier nur über [dispatcher] aufrufen.
 */
internal class AiEngine(private val context: Context) : AiDemand {

    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "ki").apply { priority = Thread.NORM_PRIORITY + 2 }
    }
    val dispatcher: CoroutineDispatcher = executor.asCoroutineDispatcher()

    /** Beendet alles: stoppt die Live-Schleife, gibt die Modelle frei und den Thread. */
    fun shutdown() {
        stopLive = true
        executor.execute { closeAll() }
        executor.shutdown()
    }

    private val models = EnumMap<ModelId, NpuModel>(ModelId::class.java)
    private var consts: EdgeTamConstants? = null
    private var tracker: Tracker? = null
    private var cacheCleaned = false

    /** Einstellung des KI-Chips für alle geladenen Modelle. */
    var chip: ChipSetting = ChipSetting.DEFAULT
        private set

    /** Welche Memory-Attention live läuft; nach der Einzelmessung die schnellere. */
    var liveAttn = ModelId.ATTN
        private set

    /** Live mit Qualcomms Memory-Encoder (samt Vorbereitung) statt unserem; nach der Einzelmessung der schnellere. */
    var liveQcMemory = false
        private set

    /**
     * Live mit unserem Decoder, der wie Metas Videoverfolgung rechnet (Qualcomms weicht davon ab).
     * Läuft er nicht, springt Qualcomms ein.
     */
    var liveMetaDecoder = true
        private set

    @Volatile var liveRunning = false
        private set
    @Volatile private var stopLive = false
    private val pendingTap = AtomicReference<Tap?>(null)

    /** Ein Tipp ins Bild: (u, v) von 0 bis 1 im aufrechten Bild; [rank] > 0 heißt gleich noch einmal an derselben Stelle. */
    private class Tap(val u: Float, val v: Float, val rank: Int)

    // Letzter Tipp (Hauptthread), um ein erneutes Tippen an derselben Stelle zu erkennen.
    private var lastTapU = -1f
    private var lastTapV = -1f
    private var lastTapAt = 0L
    private var lastTapRank = 0

    // Takt für die Grafik-Pipeline (siehe AiDemand); schreibt nur der KI-Thread.
    @Volatile private var nextTake = 0L
    @Volatile private var aiWaiting = false
    @Volatile private var wantDepth = true
    override val nextTakeNs: Long get() = nextTake
    override val waiting: Boolean get() = aiWaiting
    override val depthNeeded: Boolean get() = wantDepth || pendingTap.get() != null

    /** Wie oft die Verfolgung neu angesetzt hat; die Pipeline mischt keine Masken verschiedener Motive. */
    private var generation = 0

    /** Bekommt nach jedem Durchgang auf einem Kamerabild die Maske (Grafik-Pipeline); sonst null. */
    @Volatile var maskSink: MaskSink? = null

    /** Wärmeregelung der Live-Schleife; ihr Ergebnis über [heatJson]. */
    private val heat = HeatGovernor(context)
    private val maskBytes = ByteArray(MASK_SIZE * MASK_SIZE)

    /** Wie die Wärmeregelung im letzten Live-Lauf gebremst hat (über [dispatcher] aufrufen). */
    fun heatJson(): JSONObject = heat.toJson()

    /**
     * Stopp-Knopf während einer Messung: Leistungstest, Einzelmessung, Genauigkeit und Prüfvideo
     * hören nach dem laufenden Schritt auf und geben zurück, was sie bis dahin haben.
     */
    @Volatile var cancelWork = false
        private set

    /** Vor jedem Test aufrufen (auf dem Hauptthread). */
    fun beginWork() {
        cancelWork = false
    }

    fun requestCancel() {
        cancelWork = true
    }

    /** Ein Abschnitt einer Messung: Ein Fehler darin steht unter [key] im Bericht, die anderen laufen weiter. */
    private inline fun section(out: JSONObject, key: String, block: () -> Unit) {
        if (cancelWork) return
        try {
            block()
        } catch (e: Exception) {
            out.put(key, JSONObject().put("fehler", e.toString().take(300)))
        }
    }

    /** Qualcomms quantisierter Encoder samt Umrechnung, einmal je geladenem Modell (die Puffer sind groß). */
    private var quantEncoder: QuantEncoder? = null
    private var quantEncoderModel: NpuModel? = null

    private fun quantEncoderFor(m: NpuModel): QuantEncoder =
        quantEncoder?.takeIf { quantEncoderModel === m } ?: QuantEncoder(m).also {
            quantEncoder = it
            quantEncoderModel = m
        }

    val stats = LiveStats()

    fun loaded(id: ModelId): Boolean = models.containsKey(id)

    /** Bereit für live: Tiefe geladen und eine volle Verfolgung aus geladenen Teilen gebaut. */
    fun liveReady(): Boolean = models.containsKey(ModelId.DEPTH) && tracker != null

    /**
     * Die Modelle für live, mit Rückfall: beide Decoder und immer unser Memory-Encoder. Fällt
     * unser Decoder oder Qualcomms Memory-Encoder aus, springt der andere ein.
     */
    fun liveSet(): List<ModelId> = listOf(ModelId.DEPTH, ModelId.ENC, ModelId.DEC, ModelId.DEC_META, liveAttn, ModelId.MEMENC) +
        if (liveQcMemory) listOf(ModelId.MEMPREP, ModelId.MEMENC_QC) else emptyList()

    /**
     * Übernimmt, was Leistungstest und Einzelmessung gewählt haben (steht im Bericht), damit
     * Dauertest und Vorschau auch nach einem Neustart dasselbe nehmen. Eine andere Einstellung
     * des Chips heißt: alle Modelle neu laden.
     */
    fun configureLive(setting: ChipSetting?, attn: ModelId?, qcMemory: Boolean?, metaDecoder: Boolean?) {
        if (setting != null && setting != chip) {
            unload(ModelId.entries)
            chip = setting
        }
        if (attn == ModelId.ATTN || attn == ModelId.ATTN2) liveAttn = attn
        if (qcMemory != null) liveQcMemory = qcMemory
        if (metaDecoder != null) liveMetaDecoder = metaDecoder
        buildTracker()
    }

    // ------------------------------------------------------------ Laden

    /**
     * Lädt die angegebenen Modelle in der aktuellen Einstellung auf den KI-Chip, soweit noch
     * nicht geladen. Ein Fehler bei einem Modell hält die anderen nicht auf; er steht im Ergebnis.
     */
    fun load(ids: List<ModelId>, progress: (done: Int, total: Int, id: ModelId) -> Unit): JSONObject {
        val out = JSONObject().put("einstellung", chip.key)
        try {
            Npu.loadLibraries(context)
        } catch (e: Throwable) {
            // Ohne QNN-Bibliotheken kann kein Modell auf den Chip; der Grund steht im Bericht.
            return out.put("fehler", "QNN-Bibliotheken: $e".take(600))
        }
        if (!cacheCleaned) {
            Npu.cleanCache(context)
            cacheCleaned = true
        }
        val start = SystemClock.elapsedRealtime()
        val todo = ids.filter { !models.containsKey(it) }
        todo.forEachIndexed { i, id ->
            progress(i, todo.size, id)
            try {
                val m = NpuModel.load(context, id.asset, NpuConfig(id.fp16, chip))
                models[id] = m
                out.put(id.key, m.describe())
            } catch (e: Throwable) {
                out.put(id.key, JSONObject().put("datei", id.asset).put("fehler", e.toString().take(600))
                    .put("protokoll", LogProbe.since(System.currentTimeMillis() - 30_000)))
            }
        }
        progress(todo.size, todo.size, todo.lastOrNull() ?: ModelId.DEPTH)
        if (consts == null) consts = try {
            EdgeTamConstants.load(File(context.filesDir, ModelStore.FOLDER))
        } catch (e: Exception) {
            out.put("konstantenFehler", e.toString().take(300))
            null
        }
        buildTracker()
        trackerError?.let { out.put("verfolgungFehler", it) }
        out.put("gesamtMs", SystemClock.elapsedRealtime() - start)
        val cache = JSONArray()
        Npu.cacheDir(context).listFiles()?.sortedBy { it.name }?.forEach { cache.put("${it.name} ${r1(it.length() / 1e6)} MB") }
        out.put("zwischenspeicher", cache)
        return out
    }

    fun unload(ids: Collection<ModelId>) {
        for (id in ids) models.remove(id)?.close()
        if (quantEncoderModel != null && quantEncoderModel !== models[ModelId.ENC_Q]) {
            quantEncoder = null
            quantEncoderModel = null
        }
        buildTracker()
    }

    fun unloadExtras() = unload(ModelId.entries.filter { it !in liveSet() })

    fun closeAll() {
        unload(ModelId.entries)
    }

    /** Warum sich zuletzt keine Verfolgung bauen ließ, obwohl die Teile geladen waren (für den Bericht). */
    private var trackerError: String? = null

    /**
     * Die Verfolgung für live aus den geladenen Teilen: wie gewählt, sonst mit dem anderen
     * Memory-Encoder, dem anderen Decoder und zuletzt wie in Stufe 2 (Einbettung vom Encoder).
     * Welche es wurde, steht im Bericht (Name der Kette). Null, wenn nichts davon geht.
     */
    private fun buildTracker() {
        trackerError = null
        val other = !liveMetaDecoder
        val candidates = listOf(
            Triple(liveMetaDecoder, liveQcMemory, true),
            Triple(liveMetaDecoder, !liveQcMemory, true),
            Triple(other, liveQcMemory, true),
            Triple(other, !liveQcMemory, true),
            Triple(false, false, false),
        )
        tracker = null
        for ((meta, qcMemory, appPrompt) in candidates) {
            val t = try {
                buildFullTracker(liveAttn, qcMemory, appPrompt, meta)
            } catch (e: IllegalArgumentException) {
                if (trackerError == null) trackerError = e.message?.take(300)
                null
            }
            if (t != null) {
                tracker = t
                return
            }
        }
    }

    /** Eingebauter Decoder: unserer wie im Original oder Qualcomms; null, wenn er nicht geladen ist oder nicht passt. */
    private fun decoderOf(meta: Boolean, from: Map<ModelId, NpuModel> = models): MaskDecoder? = try {
        if (meta) from[ModelId.DEC_META]?.let { MetaDecoder(it) } else from[ModelId.DEC]?.let { QualcommDecoder(it) }
    } catch (e: IllegalArgumentException) {
        null
    }

    /** Volle Verfolgung mit Speicher aus den geladenen Teilen, oder null, wenn etwas fehlt. */
    private fun buildFullTracker(attnId: ModelId, qcMemory: Boolean, appPrompt: Boolean, metaDecoder: Boolean): EdgeTamTracker? {
        val enc = models[ModelId.ENC] ?: return null
        val decoder = decoderOf(metaDecoder) ?: return null
        val attn = models[attnId] ?: return null
        val c = consts ?: return null
        val writer: MemoryWriter = if (qcMemory) {
            QualcommMemoryWriter(models[ModelId.MEMPREP] ?: return null, models[ModelId.MEMENC_QC] ?: return null)
        } else {
            OwnMemoryWriter(models[ModelId.MEMENC] ?: return null)
        }
        val prompt = if (appPrompt) c.prompt ?: return null else null
        if (prompt == null && decoder.points != 2) return null // die Einbettung des Encoders passt nur zu Qualcomms Decoder
        return EdgeTamTracker(
            FloatEncoder(enc), decoder, attn, writer, c.noObjPtr, prompt,
            enc.newInput(enc.inputIndex("unnorm_coords")), enc.newInput(enc.inputIndex("labels"))
        )
    }

    private fun buildFastTracker(quantized: Boolean, metaDecoder: Boolean): FastTracker? {
        val decoder = decoderOf(metaDecoder) ?: return null
        val prompt = consts?.prompt ?: return null
        val encModel = models[if (quantized) ModelId.ENC_Q else ModelId.ENC] ?: return null
        val encoder: ImageEncoder = if (quantized) quantEncoderFor(encModel) else FloatEncoder(encModel)
        // Die Punkte des Encoders zählen hier nicht; float-Puffer in der Form des fp16-Encoders reichen.
        return FastTracker(encoder, decoder, prompt, directBuffer(16), directBuffer(8))
    }

    // ------------------------------------------------------------ Leistungstest

    /**
     * Misst jedes Modell aus [ModelId.MATRIX] in jeder Einstellung des KI-Chips (3 Läufe Einlaufen,
     * dann 10) und wählt die Einstellung, mit der die volle Verfolgung samt Tiefe am schnellsten ist
     * (läuft sie nirgends, zählt die Schnellverfolgung, sonst die Tiefe). Bei weniger als 3 %
     * Unterschied gewinnt die kühlere Stufe. Gemessen wird Modell für Modell durch alle
     * Einstellungen: Erwärmt sich das Handy dabei, trifft das alle Einstellungen eines Modells
     * gleich. Zuletzt laufen die beste und dieselbe mit festgehaltenem Takt im Wechsel; der
     * festgehaltene Takt gewinnt nur, wenn er mehr als 3 % schneller ist. Danach sind alle Modelle
     * entladen; sie laden in der gewählten Einstellung neu.
     */
    fun perfMatrix(frame: FrameSlot, progress: (String, Int, Int) -> Unit): JSONObject {
        unload(ModelId.entries)
        val start = SystemClock.elapsedRealtime()
        val settings = ChipSetting.CANDIDATES
        val total = (settings.size + 2) * ModelId.MATRIX.size
        var done = 0
        val heat = JSONArray()

        class Series(val setting: ChipSetting, val pass: Int) {
            val medians = EnumMap<ModelId, Double>(ModelId::class.java)
            val details = JSONObject()
            fun estimates() = chainEstimates(medians)
            fun row(): JSONObject = JSONObject().put("einstellung", setting.key).put("titel", setting.title)
                .put("durchgang", pass).put("modelle", details)
                .also { r -> estimates().forEach { (k, v) -> r.put(k, r2(v)) } }
        }

        fun measure(series: Series, id: ModelId) {
            // Nach dem Stopp nichts mehr laden; die Anzeige bleibt bei „Stoppt …“.
            if (cancelWork) return
            val s = series.setting
            progress("${s.title} · ${id.title}", done++, total)
            var m: NpuModel? = null
            try {
                m = NpuModel.load(context, id.asset, NpuConfig(id.fp16, s))
                val inputs = benchInputs(id, m, frame)
                m.vote()
                val t = time(m, runs = MATRIX_RUNS) { m.run(inputs) }.getJSONObject("imModell")
                val median = t.getDouble("medianMs")
                series.medians[id] = median
                series.details.put(id.key, JSONObject()
                    .put("medianMs", median)
                    .put("p90Ms", t.getDouble("p90Ms"))
                    .put("ladenMs", r1(m.loadMs))
                    .put("ausZwischenspeicher", m.fromCache))
            } catch (e: Throwable) {
                series.details.put(id.key, JSONObject().put("fehler", e.toString().take(300)))
            } finally {
                m?.close()
            }
        }

        val first = settings.map { Series(it, 1) }
        for (id in ModelId.MATRIX) {
            if (cancelWork) break
            heat.put(DeviceHeat.read(context).put("vor", id.key))
            for (series in first) measure(series, id)
        }
        if (cancelWork) {
            // Unvollständig: keine Wahl, die Einstellung bleibt.
            val rows = JSONArray()
            first.forEach { rows.put(it.row()) }
            return JSONObject().put("einstellungen", rows).put("waerme", heat).put("abgebrochen", true)
                .put("gesamtS", r1((SystemClock.elapsedRealtime() - start) / 1000.0))
        }

        // Maßstab: die volle Verfolgung, wenn sie in mindestens einer Einstellung ganz lief.
        val scale = listOf("vollMs", "schnellMs").firstOrNull { k -> first.any { it.estimates()[k] != null } }
        fun cost(series: Series): Double =
            (if (scale != null) series.estimates()[scale] else series.medians[ModelId.DEPTH]) ?: Double.NaN
        val ran = first.filter { !cost(it).isNaN() }
        var best: Series? = ran.minByOrNull { cost(it) }
        if (best != null) {
            val bestMs = cost(best)
            best = ran.filter { cost(it) <= bestMs * 1.03 }
                .minWith(compareBy<Series> { it.setting.perf.ordinal }.thenBy { cost(it) })
        }
        val what = when (scale) {
            "vollMs" -> "volle Verfolgung mit Tiefe"
            "schnellMs" -> "Schnellverfolgung mit Tiefe (die volle lief in keiner Einstellung)"
            else -> "Tiefe (die Verfolgung lief in keiner Einstellung)"
        }
        var chosen = best?.setting
        var reason = "$what am schnellsten"
        val rows = JSONArray()
        first.forEach { rows.put(it.row()) }
        if (best != null) {
            // Die beste noch einmal, im Wechsel mit festgehaltenem Takt, damit beide unter gleichen Bedingungen laufen.
            val again = Series(best.setting, 2)
            val held = Series(best.setting.copy(manualVote = true), 2)
            for (id in ModelId.MATRIX) {
                if (cancelWork) break
                measure(again, id)
                measure(held, id)
            }
            rows.put(again.row())
            rows.put(held.row())
            val a = cost(again)
            val h = cost(held)
            if (!cancelWork && !a.isNaN() && !h.isNaN() && h < a * 0.97) {
                chosen = held.setting
                reason = "mit festgehaltenem Takt mehr als 3 % schneller"
            }
        }
        heat.put(DeviceHeat.read(context).put("vor", "ende"))
        val out = JSONObject().put("einstellungen", rows).put("waerme", heat)
            .put("gesamtS", r1((SystemClock.elapsedRealtime() - start) / 1000.0))
        if (chosen != null) {
            out.put("gewaehlt", chosen.key).put("gewaehltTitel", chosen.title).put("grund", reason)
            chip = chosen
        } else {
            out.put("fehler", "Kein Modell lief in irgendeiner Einstellung; es bleibt ${chip.title}.")
        }
        if (cancelWork) {
            // Gestoppt im zweiten Durchgang: Die Wahl aus dem ersten gilt, der Vergleich mit festgehaltenem Takt fehlt.
            out.put("abgebrochen", true)
        } else {
            progress("fertig", total, total)
        }
        return out
    }

    /**
     * Geschätzte Zeiten je Bild aus Einzelzeiten: voll (mit Speicher) und schnell (ohne), jeweils
     * mit Tiefe und mit dem Decoder, der live liefe (unserer wie im Original, sonst Qualcomms).
     */
    private fun chainEstimates(m: Map<ModelId, Double>): Map<String, Double> {
        val out = LinkedHashMap<String, Double>()
        val depth = m[ModelId.DEPTH] ?: return out
        val enc = m[ModelId.ENC]
        val dec = m[ModelId.DEC_META] ?: m[ModelId.DEC]
        val attn = listOfNotNull(m[ModelId.ATTN], m[ModelId.ATTN2]).minOrNull()
        val qcMem = if (m[ModelId.MEMPREP] != null && m[ModelId.MEMENC_QC] != null) m.getValue(ModelId.MEMPREP) + m.getValue(ModelId.MEMENC_QC) else null
        val mem = listOfNotNull(m[ModelId.MEMENC], qcMem).minOrNull()
        if (enc != null && dec != null && attn != null && mem != null) out["vollMs"] = depth + enc + attn + dec + mem
        if (enc != null && dec != null) out["schnellMs"] = depth + enc + dec
        val encQ = m[ModelId.ENC_Q]
        if (encQ != null && dec != null) out["schnellQuantisiertMs"] = depth + encQ + dec
        return out
    }

    /** Eingänge für eine Zeitmessung: echte Bilder für Tiefe und Encoder, sonst Nullen (die Rechenzeit hängt nicht vom Inhalt ab). */
    private fun benchInputs(id: ModelId, m: NpuModel, frame: FrameSlot): Array<Any> = when (id) {
        ModelId.DEPTH -> arrayOf(frame.depth)
        ModelId.ENC -> {
            val inputs = arrayOfNulls<Any>(3)
            inputs[m.inputIndex("image")] = frame.enc
            inputs[m.inputIndex("unnorm_coords")] = directBuffer(16).apply { putFloat(0.5f).putFloat(0.5f).putFloat(0f).putFloat(0f).rewind() }
            inputs[m.inputIndex("labels")] = directBuffer(8).apply { putFloat(1f).putFloat(-1f).rewind() }
            @Suppress("UNCHECKED_CAST")
            inputs as Array<Any>
        }
        else -> Array(m.inputCount) { m.newInput(it) }
    }

    // ------------------------------------------------------------ Einzelmessung

    /**
     * Misst jedes geladene Modell einzeln (3 Läufe Einlaufen, dann 20) und danach ganze Ketten
     * je Bild: wie in Stufe 2, die beste volle Verfolgung und die Schnellverfolgung. Wählt die
     * schnellere Memory-Attention und den schnelleren Memory-Encoder für live.
     */
    fun bench(frame: FrameSlot, progress: (String) -> Unit): JSONObject {
        // Wird gestoppt, bleibt die bisherige Wahl für live (wie im Leistungstest).
        val beforeAttn = liveAttn
        val beforeQcMemory = liveQcMemory
        val beforeMetaDecoder = liveMetaDecoder
        val out = JSONObject().put("einstellung", chip.key)
        val perModel = JSONObject()
        for (id in ModelId.entries) {
            val m = models[id] ?: continue
            if (cancelWork) break
            progress(id.title)
            section(perModel, id.key) {
                val inputs = benchInputs(id, m, frame)
                try {
                    m.vote()
                    perModel.put(id.key, time(m) { m.run(inputs) })
                } finally {
                    m.release()
                }
            }
        }
        out.put("modelle", perModel)

        // Schnellere Memory-Attention für live: Beide haben dieselben Ein- und Ausgänge, es zählt die Zeit im Modell.
        fun med(id: ModelId) = perModel.optJSONObject(id.key)?.optJSONObject("imModell")?.optDouble("medianMs", Double.NaN) ?: Double.NaN
        val a1 = med(ModelId.ATTN)
        val a2 = med(ModelId.ATTN2)
        liveAttn = if (!a2.isNaN() && (a1.isNaN() || a2 < a1)) ModelId.ATTN2 else ModelId.ATTN

        // Schnellerer Memory-Encoder für live: so gemessen, wie die Kette ihn sieht, mit dem
        // Weiterreichen der Puffer (Qualcomms Weg hat ein Modell mehr und reicht 8 MB weiter).
        val memory = JSONObject()
        val writers = listOfNotNull(
            models[ModelId.MEMENC]?.let { "eigener" to { OwnMemoryWriter(it) } },
            models[ModelId.MEMPREP]?.let { p -> models[ModelId.MEMENC_QC]?.let { q -> "qualcomm" to { QualcommMemoryWriter(p, q) } } },
        )
        for ((key, make) in writers) {
            if (cancelWork) break
            progress(if (key == "eigener") "unseren Memory-Encoder in der Kette" else "Qualcomms Memory-Encoder in der Kette")
            section(memory, key) { memory.put(key, writerTiming(make())) }
        }
        out.put("speicherEncoderKette", memory)
        fun wall(key: String) = memory.optJSONObject(key)?.optJSONObject("mitKopieren")?.optDouble("medianMs", Double.NaN) ?: Double.NaN
        val own = wall("eigener")
        val qc = wall("qualcomm")
        liveQcMemory = !qc.isNaN() && (own.isNaN() || qc < own)

        // Ketten je Bild, jeweils mit der Tiefe davor. „voll“ ist die volle Verfolgung mit unserem
        // Decoder wie im Original, „vollQualcomm“ dieselbe mit Qualcomms Decoder.
        val chains = JSONObject()
        val depth = models[ModelId.DEPTH]
        val candidates = listOf(
            "stufe2" to { buildFullTracker(ModelId.ATTN, qcMemory = false, appPrompt = false, metaDecoder = false) },
            "voll" to { buildFullTracker(liveAttn, liveQcMemory, appPrompt = true, metaDecoder = true) },
            "vollQualcomm" to { buildFullTracker(liveAttn, liveQcMemory, appPrompt = true, metaDecoder = false) },
            "schnell" to { buildFastTracker(quantized = false, metaDecoder = true) },
            "schnellQuantisiert" to { buildFastTracker(quantized = true, metaDecoder = true) },
        )
        if (depth != null) {
            for ((key, make) in candidates) {
                if (cancelWork) break
                val t = try {
                    make()
                } catch (e: Exception) {
                    chains.put(key, JSONObject().put("fehler", e.toString().take(300)))
                    null
                } ?: continue
                progress("Kette ${t.name}")
                chains.put(key, try {
                    chain(t, depth, frame)
                } catch (e: Exception) {
                    JSONObject().put("name", t.name).put("fehler", e.toString().take(300))
                })
            }
        }
        out.put("ketten", chains)
        if (cancelWork) {
            // Unvollständig: keine Wahl für live und nichts davon im Bericht, damit auch ein
            // Neustart die bisherige Wahl behält.
            liveAttn = beforeAttn
            liveQcMemory = beforeQcMemory
            liveMetaDecoder = beforeMetaDecoder
            buildTracker()
            return out.put("abgebrochen", true)
        }
        // Live vorerst: unser Decoder wie im Original, wenn seine Kette lief; nach dem Prüfvideo
        // entscheidet die Genauigkeit noch einmal (chooseDecoderAfterVideo).
        val metaRan = chains.optJSONObject("voll")?.let { !it.has("fehler") } == true
        liveMetaDecoder = metaRan
        buildTracker()
        out.put("liveMemoryAttention", liveAttn.key)
        out.put("liveMemoryEncoder", if (liveQcMemory) "qualcomm" else "eigener")
        out.put("liveDecoder", if (liveMetaDecoder) "meta" else "qualcomm")
        // Wie in Stufe 2: „jeBild“ ist die volle Verfolgung, die live läuft.
        chains.optJSONObject(liveChainKey(liveMetaDecoder))?.let { out.put("jeBild", it) }
        return out
    }

    /**
     * Nach dem Prüfvideo: Unser Decoder bleibt live, solange er auf dem Chip mindestens so genau
     * trifft wie Qualcomms (1 Prozentpunkt Spielraum); sonst Qualcomms. Lief nur einer der beiden
     * durch, weil der andere scheiterte, nimmt live diesen. Null, wenn keiner durchlief oder der
     * Vergleich fehlt, weil das Prüfvideo gestoppt wurde (dann bleibt die bisherige Wahl).
     */
    fun chooseDecoderAfterVideo(video: JSONObject): String? {
        val meta = video.optJSONObject("voll")?.optDouble("maskeGleichMittel", Double.NaN) ?: Double.NaN
        val qualcomm = video.optJSONObject("vollQualcomm")?.optDouble("maskeGleichMittel", Double.NaN) ?: Double.NaN
        val stopped = video.optBoolean("abgebrochen")
        val useMeta = when {
            meta.isNaN() && qualcomm.isNaN() -> return null
            (meta.isNaN() || qualcomm.isNaN()) && stopped -> return null
            meta.isNaN() -> false
            qualcomm.isNaN() -> true
            else -> meta + 0.01 >= qualcomm
        }
        if (useMeta != liveMetaDecoder) {
            liveMetaDecoder = useMeta
            buildTracker()
        }
        return if (useMeta) "meta" else "qualcomm"
    }

    /** Zeit eines Memory-Encoders, wie die Kette sie sieht: Wanduhr mit dem Weiterreichen der Puffer, dazu die Zeit im Modell. */
    private fun writerTiming(writer: MemoryWriter): JSONObject {
        val emb = directBuffer(64 * 64 * EdgeTamTracker.D * 4)
        val mask = directBuffer(MASK_SIZE * MASK_SIZE * 4)
        val result = FloatArray(EdgeTamTracker.TOK_MEM * EdgeTamTracker.MEM_DIM)
        val used = models.values.toList()
        try {
            used.forEach { it.vote() }
            repeat(WARMUP) { writer.write(emb, mask, startFrame = false, out = result) }
            val wall = DoubleArray(RUNS)
            val native = DoubleArray(RUNS)
            for (i in 0 until RUNS) {
                val s = System.nanoTime()
                writer.write(emb, mask, startFrame = false, out = result)
                wall[i] = (System.nanoTime() - s) / 1e6
                native[i] = writer.lastMs
            }
            return JSONObject().put("mitKopieren", Timing(wall).toJson()).put("imModell", Timing(native).toJson())
        } finally {
            used.forEach { it.release() }
        }
    }

    private fun chain(t: Tracker, depth: NpuModel, frame: FrameSlot): JSONObject {
        val used = liveModelsOf(t)
        try {
            used.forEach { it.vote() }
            depth.vote()
            t.start(frame.enc, 0.5f, 0.5f)
            repeat(WARMUP) {
                depth.run(arrayOf(frame.depth))
                t.step(frame.enc)
            }
            val series = Array(8) { DoubleArray(RUNS) }
            for (i in 0 until RUNS) {
                val s = System.nanoTime()
                series[1][i] = depth.run(arrayOf(frame.depth))
                t.step(frame.enc)
                series[0][i] = (System.nanoTime() - s) / 1e6
                series[2][i] = t.totalMs
                series[3][i] = t.encMs
                series[4][i] = t.attnMs
                series[5][i] = t.decMs
                series[6][i] = t.memMs
                series[7][i] = t.convertMs
            }
            val names = listOf("beide", "tiefe", "verfolgung", "encoder", "speicherAttention", "decoder", "speicherEncoder", "umrechnen")
            val o = JSONObject().put("name", t.name)
            names.forEachIndexed { k, n -> if (series[k].any { it > 0.0 }) o.put(n, Timing(series[k]).toJson()) }
            o.put("objektGefunden", t.present)
            t.reset()
            return o
        } finally {
            depth.release()
            used.forEach { it.release() }
        }
    }

    /** Die Modelle, die eine Kette nutzt, damit ein festgehaltener Takt für alle gilt. */
    private fun liveModelsOf(@Suppress("UNUSED_PARAMETER") t: Tracker): List<NpuModel> = models.values.toList()

    private fun time(m: NpuModel, runs: Int = RUNS, run: () -> Double): JSONObject {
        repeat(WARMUP) { run() }
        val total = DoubleArray(runs)
        val native = DoubleArray(runs)
        for (i in 0 until runs) {
            total[i] = run()
            native[i] = m.lastNativeMs
        }
        return JSONObject()
            .put("imModell", Timing(native).toJson())
            .put("mitKopieren", Timing(total).toJson())
    }

    // ------------------------------------------------------------ Genauigkeit

    /**
     * Rechnet dasselbe Bild auf der CPU (volle Genauigkeit) und vergleicht. Jedes Modell
     * bekommt dabei genau dieselben Eingänge; zuletzt läuft die ganze Verfolgung doppelt.
     * Jeder Abschnitt ist für sich abgesichert: Ein Fehler steht unter seinem Namen im Bericht.
     */
    fun accuracy(frame: FrameSlot, progress: (String) -> Unit): JSONObject {
        val out = JSONObject()
        val cpu = EnumMap<ModelId, NpuModel>(ModelId::class.java)
        try {
            val loadErrors = JSONObject()
            for (id in listOf(ModelId.DEPTH, ModelId.ENC, ModelId.DEC, ModelId.DEC_META, ModelId.ATTN, ModelId.ATTN2, ModelId.MEMENC, ModelId.MEMPREP, ModelId.MEMENC_QC)) {
                if (!models.containsKey(id) || cancelWork) continue
                progress("Lädt ${id.title} für die CPU")
                try {
                    cpu[id] = NpuModel.load(context, id.asset, null, threads = 4)
                } catch (e: Exception) {
                    loadErrors.put(id.key, e.toString().take(300))
                }
            }
            if (loadErrors.length() > 0) out.put("cpuLadefehler", loadErrors)

            section(out, "tiefe") {
                val dn = models[ModelId.DEPTH]
                val dc = cpu[ModelId.DEPTH]
                if (dn != null && dc != null) {
                    progress("Vergleicht die Tiefe")
                    val cpuMs = dc.run(arrayOf(frame.depth))
                    dn.run(arrayOf(frame.depth))
                    out.put("tiefe", compare(dn.output(0), dc.output(0)).put("cpuMs", r1(cpuMs)))
                }
            }

            // Encoder: Seine Ausgänge auf der CPU sind die Eingänge aller weiteren Vergleiche.
            val ec = cpu[ModelId.ENC] ?: return out
            val en = models[ModelId.ENC] ?: return out
            val center = directBuffer(16).apply { putFloat(0.5f).putFloat(0.5f).putFloat(0f).putFloat(0f).rewind() }
            val labels = directBuffer(8).apply { putFloat(1f).putFloat(-1f).rewind() }
            var encoderOk = false
            section(out, "encoder") {
                progress("Vergleicht den Encoder")
                val encIn = arrayOfNulls<Any>(3)
                encIn[ec.inputIndex("image")] = frame.enc
                encIn[ec.inputIndex("unnorm_coords")] = center
                encIn[ec.inputIndex("labels")] = labels
                @Suppress("UNCHECKED_CAST")
                val encCpuMs = ec.run(encIn as Array<Any>)
                en.run(encIn)
                val eo = JSONObject()
                for (name in listOf("image_embeddings", "high_res_features1", "high_res_features2", "pix_feat")) {
                    eo.put(name, compare(en.output(en.outputIndex(name)), ec.output(ec.outputIndex(name))))
                }
                eo.put("cpuMs", r1(encCpuMs))
                // Quantisierter Encoder gegen die CPU (Ausgänge zurück in float umgerechnet).
                models[ModelId.ENC_Q]?.let { q ->
                    section(eo, "quantisiert") {
                        val qe = quantEncoderFor(q)
                        qe.run(frame.enc, center, labels)
                        eo.put("quantisiert", JSONObject()
                            .put("image_embeddings", compare(qe.embeddings, ec.output(ec.outputIndex("image_embeddings"))))
                            .put("high_res_features1", compare(qe.hr1, ec.output(ec.outputIndex("high_res_features1"))))
                            .put("high_res_features2", compare(qe.hr2, ec.output(ec.outputIndex("high_res_features2"))))
                            .put("pix_feat", compare(qe.pixFeat, ec.output(ec.outputIndex("pix_feat"))))
                            .put("umrechnenMs", r1(qe.convertMs)))
                    }
                }
                out.put("encoder", eo)
                encoderOk = true
            }
            if (!encoderOk) return out
            val pix = ec.output(ec.outputIndex("pix_feat"))
            val hr1 = ec.output(ec.outputIndex("high_res_features1"))
            val hr2 = ec.output(ec.outputIndex("high_res_features2"))
            val emb = ec.output(ec.outputIndex("image_embeddings"))

            // Eingabe-Einbettung der App gegen die des Encoders (CPU) und gegen Metas Proben.
            section(out, "einbettung") {
                consts?.prompt?.let { p ->
                    val mine = FloatArray(3 * PromptEncoder.D)
                    p.embed(floatArrayOf(0.5f, 0f), floatArrayOf(0.5f, 0f), intArrayOf(1, -1), mine)
                    val po = compare(buffer(mine), ec.output(ec.outputIndex("sparse_embedding")))
                    consts?.promptSamples?.let { po.put("metaProben", p.check(it)) }
                    out.put("einbettung", po)
                }
            }

            // Qualcomms Decoder: Maske und Zeiger der CPU sind danach die Eingänge für die Memory-Teile.
            var maskC: FloatArray? = null
            var ptr: FloatArray? = null
            section(out, "decoder") {
                val dcd = cpu[ModelId.DEC]
                val dnd = models[ModelId.DEC]
                if (dcd != null && dnd != null) {
                    progress("Vergleicht den Decoder")
                    val decIn = arrayOfNulls<Any>(4)
                    decIn[dcd.inputIndex("image_embeddings")] = pix
                    decIn[dcd.inputIndex("high_res_features1")] = hr1
                    decIn[dcd.inputIndex("high_res_features2")] = hr2
                    decIn[dcd.inputIndex("sparse_embedding")] = ec.output(ec.outputIndex("sparse_embedding"))
                    @Suppress("UNCHECKED_CAST")
                    val decCpuMs = dcd.run(decIn as Array<Any>)
                    dnd.run(decIn)
                    val mc = floats(dcd.output(dcd.outputIndex("masks")))
                    val mn = floats(dnd.output(dnd.outputIndex("masks")))
                    out.put(
                        "decoder", JSONObject()
                            .put("maskeGleich", r2(iou(mn, mc)))
                            .put("maskeCpuAnteil", r4(mc.count { it > 0f } / mc.size.toDouble()))
                            .put("maskeChipAnteil", r4(mn.count { it > 0f } / mn.size.toDouble()))
                            .put("maskenLogits", compare(dnd.output(dnd.outputIndex("masks")), dcd.output(dcd.outputIndex("masks"))))
                            .put("scoreCpu", num2(dcd.output(dcd.outputIndex("scores")).getFloat(0).toDouble()))
                            .put("scoreChip", num2(dnd.output(dnd.outputIndex("scores")).getFloat(0).toDouble()))
                            .put("zeiger", compare(dnd.output(dnd.outputIndex("obj_ptr")), dcd.output(dcd.outputIndex("obj_ptr"))))
                            .put("objektWertCpu", num2(dcd.output(dcd.outputIndex("object_score_logits")).getFloat(0).toDouble()))
                            .put("objektWertChip", num2(dnd.output(dnd.outputIndex("object_score_logits")).getFloat(0).toDouble()))
                            .put("cpuMs", r1(decCpuMs))
                    )
                    maskC = mc
                    ptr = floats(dcd.output(dcd.outputIndex("obj_ptr")))
                }
            }

            // Unser Decoder wie im Original, mit der Einbettung der App (ein Punkt in der Mitte).
            section(out, "decoderWieMeta") {
                val metaCpu = decoderOf(meta = true, from = cpu) as? MetaDecoder
                val metaChip = decoderOf(meta = true) as? MetaDecoder
                val prompt = consts?.prompt
                if (metaCpu != null && metaChip != null && prompt != null) {
                    progress("Vergleicht den Decoder wie im Original")
                    val sparse = metaCpu.newSparse()
                    prompt.write(floatArrayOf(0.5f), floatArrayOf(0.5f), intArrayOf(1), FloatArray(2 * PromptEncoder.D), sparse)
                    val outC = directBuffer(MASK_SIZE * MASK_SIZE * 4)
                    val outN = directBuffer(MASK_SIZE * MASK_SIZE * 4)
                    val ptrC = FloatArray(PromptEncoder.D)
                    val ptrN = FloatArray(PromptEncoder.D)
                    val scoreC = metaCpu.run(pix, hr1, hr2, sparse, outC, ptrC)
                    val scoreN = metaChip.run(pix, hr1, hr2, sparse, outN, ptrN)
                    val mC = floats(outC)
                    val mN = floats(outN)
                    out.put(
                        "decoderWieMeta", JSONObject()
                            .put("maskeGleich", r2(iou(mN, mC)))
                            .put("maskeCpuAnteil", r4(mC.count { it > 0f } / mC.size.toDouble()))
                            .put("maskeChipAnteil", r4(mN.count { it > 0f } / mN.size.toDouble()))
                            .put("wahlCpu", metaCpu.lastChoice)
                            .put("wahlChip", metaChip.lastChoice)
                            .put("maskenLogits", compare(outN, outC))
                            .put("zeiger", compare(buffer(ptrN), buffer(ptrC)))
                            .put("objektWertCpu", num2(scoreC.toDouble()))
                            .put("objektWertChip", num2(scoreN.toDouble()))
                            .put("cpuMs", r1(metaCpu.lastMs))
                    )
                }
            }

            // Memory-Encoder: unserer auf Chip und CPU, Qualcomms samt Vorbereitung gegen unseren.
            var memRef: FloatArray? = null
            section(out, "speicherEncoder") {
                val mask = maskC
                val mc = cpu[ModelId.MEMENC]
                if (mask != null && mc != null) {
                    progress("Vergleicht den Memory-Encoder")
                    val maskBuf = buffer(mask)
                    val so = JSONObject()
                    val ref = FloatArray(EdgeTamTracker.TOK_MEM * EdgeTamTracker.MEM_DIM)
                    OwnMemoryWriter(mc).write(emb, maskBuf, startFrame = false, out = ref)
                    memRef = ref
                    val refBuf = buffer(ref)
                    models[ModelId.MEMENC]?.let { mn ->
                        section(so, "eigener") {
                            val chipOut = FloatArray(ref.size)
                            OwnMemoryWriter(mn).write(emb, maskBuf, startFrame = false, out = chipOut)
                            so.put("eigener", compare(buffer(chipOut), refBuf))
                        }
                    }
                    val pc = cpu[ModelId.MEMPREP]
                    val qc = cpu[ModelId.MEMENC_QC]
                    if (pc != null && qc != null) {
                        section(so, "qualcommCpuGegenEigenen") {
                            val qcOut = FloatArray(ref.size)
                            QualcommMemoryWriter(pc, qc).write(emb, maskBuf, startFrame = false, out = qcOut)
                            so.put("qualcommCpuGegenEigenen", compare(buffer(qcOut), refBuf))
                            val start1 = FloatArray(ref.size)
                            val start2 = FloatArray(ref.size)
                            OwnMemoryWriter(mc).write(emb, maskBuf, startFrame = true, out = start1)
                            QualcommMemoryWriter(pc, qc).write(emb, maskBuf, startFrame = true, out = start2)
                            so.put("qualcommCpuGegenEigenenStartbild", compare(buffer(start2), buffer(start1)))
                        }
                    }
                    val pn = models[ModelId.MEMPREP]
                    val qn = models[ModelId.MEMENC_QC]
                    if (pn != null && qn != null) {
                        section(so, "qualcomm") {
                            val chipOut = FloatArray(ref.size)
                            QualcommMemoryWriter(pn, qn).write(emb, maskBuf, startFrame = false, out = chipOut)
                            so.put("qualcomm", compare(buffer(chipOut), refBuf))
                        }
                    }
                    out.put("speicherEncoder", so)
                }
            }

            // Memory-Attention mit einem echten Speicher: Startbild in Platz 0 und 6.
            section(out, "speicherAttention") {
                val ac = cpu[ModelId.ATTN]
                val mem = memRef
                val p = ptr
                if (ac != null && mem != null && p != null) {
                    progress("Vergleicht die Memory-Attention")
                    val bank = directBuffer(EdgeTamTracker.N_MEM * EdgeTamTracker.TOK_MEM * EdgeTamTracker.MEM_DIM * 4)
                    val slot = EdgeTamTracker.TOK_MEM * EdgeTamTracker.MEM_DIM
                    bank.asFloatBuffer().apply { put(mem); position(6 * slot); put(mem) }
                    val ptrs = directBuffer(EdgeTamTracker.N_PTR * EdgeTamTracker.D * 4)
                    ptrs.asFloatBuffer().apply { put(p); put(p) }
                    val bias = directBuffer(EdgeTamTracker.N_K * 4)
                    bias.asFloatBuffer().apply {
                        for (i in 0 until EdgeTamTracker.N_K) {
                            val inSlot = i < EdgeTamTracker.N_MEM * EdgeTamTracker.TOK_MEM
                            val valid = if (inSlot) {
                                val s = i / EdgeTamTracker.TOK_MEM
                                s == 0 || s == 6
                            } else {
                                (i - EdgeTamTracker.N_MEM * EdgeTamTracker.TOK_MEM) / EdgeTamTracker.TOK_PTR < 2
                            }
                            put(if (valid) 0f else EdgeTamTracker.MASK)
                        }
                    }
                    val ao = JSONObject()
                    val attnCpuMs = ac.run(arrayOf(emb, bank, ptrs, bias))
                    val ref = floats(ac.output(0))
                    ao.put("cpuMs", r1(attnCpuMs))
                    models[ModelId.ATTN]?.let { an ->
                        section(ao, "chip") {
                            an.run(arrayOf(emb, bank, ptrs, bias))
                            ao.put("chip", compare(an.output(0), buffer(ref)))
                        }
                    }
                    cpu[ModelId.ATTN2]?.let { a2 ->
                        section(ao, "neuCpuGegenAlt") {
                            a2.run(arrayOf(emb, bank, ptrs, bias))
                            ao.put("neuCpuGegenAlt", compare(a2.output(0), buffer(ref)))
                        }
                    }
                    models[ModelId.ATTN2]?.let { a2 ->
                        section(ao, "neuChip") {
                            a2.run(arrayOf(emb, bank, ptrs, bias))
                            ao.put("neuChip", compare(a2.output(0), buffer(ref)))
                        }
                    }
                    out.put("speicherAttention", ao)
                }
            }

            // Ganze Verfolgung doppelt: dieselbe Kette wie live, einmal auf der CPU, einmal auf dem Chip.
            // Startbild in der Mitte, dann zwei Folgebilder.
            section(out, "verfolgung") {
                val c = consts
                val attnCpu = cpu[liveAttn]
                val writerCpu: MemoryWriter? = if (liveQcMemory) {
                    val p = cpu[ModelId.MEMPREP]
                    val q = cpu[ModelId.MEMENC_QC]
                    if (p != null && q != null) QualcommMemoryWriter(p, q) else null
                } else {
                    cpu[ModelId.MEMENC]?.let { OwnMemoryWriter(it) }
                }
                val useMeta = liveMetaDecoder && cpu.containsKey(ModelId.DEC_META) && models.containsKey(ModelId.DEC_META)
                val decoderCpu: MaskDecoder? = decoderOf(meta = useMeta, from = cpu)
                val prompt = c?.prompt
                if (c != null && prompt != null && attnCpu != null && writerCpu != null && decoderCpu != null) {
                    val tn = buildFullTracker(liveAttn, liveQcMemory, appPrompt = true, metaDecoder = useMeta)
                    if (tn != null) {
                        progress("Vergleicht die ganze Verfolgung")
                        val tc = EdgeTamTracker(FloatEncoder(ec), decoderCpu, attnCpu, writerCpu, c.noObjPtr, prompt, directBuffer(16), directBuffer(8))
                        try {
                            tc.start(frame.enc, 0.5f, 0.5f)
                            val startC = tc.mask().copyOf()
                            tn.start(frame.enc, 0.5f, 0.5f)
                            val startN = tn.mask().copyOf()
                            repeat(2) {
                                tc.step(frame.enc)
                                tn.step(frame.enc)
                            }
                            val followC = tc.mask()
                            val followN = tn.mask()
                            out.put(
                                "verfolgung", JSONObject()
                                    .put("kette", tn.name)
                                    .put("startMaskeGleich", r2(iou(startN, startC)))
                                    .put("folgeMaskeGleich", r2(iou(followN, followC)))
                                    .put("objektCpu", tc.present)
                                    .put("objektChip", tn.present)
                                    .put("maskeCpuAnteil", r4(followC.count { it > 0f } / followC.size.toDouble()))
                                    .put("maskeChipAnteil", r4(followN.count { it > 0f } / followN.size.toDouble()))
                                    .put("cpuMsJeBild", r1(tc.totalMs))
                            )
                        } finally {
                            tn.reset()
                            tc.reset()
                        }
                    }
                }
            }
        } finally {
            // Auch nach einem frühen Ende (etwa ohne Encoder auf der CPU) steht ein Stopp im Bericht.
            if (cancelWork) out.put("abgebrochen", true)
            for (m in cpu.values) m.close()
        }
        return out
    }

    /**
     * Prüfvideo aus unserem Export: Metas Original hat darin einen Ball verfolgt. Jede Kette
     * verfolgt ihn hier noch einmal auf dem Chip; verglichen wird Bild für Bild mit Metas Maske.
     */
    fun videoCheck(progress: (String) -> Unit): JSONObject {
        val out = JSONObject()
        val meta = consts?.testVideo ?: return out.put("fehler", "Kein Prüfvideo in diesem KI-Satz")
        val file = File(File(context.filesDir, ModelStore.FOLDER), meta.getString("datei"))
        if (!file.isFile) return out.put("fehler", "Prüfvideo fehlt: ${file.name}")
        val frames = meta.getInt("bilder")
        val startU = meta.getJSONArray("start").getDouble(0).toFloat()
        val startV = meta.getJSONArray("start").getDouble(1).toFloat()
        // Was dieselbe Kette im Export auf der CPU erreichte (gleiche Modelle, volle Genauigkeit):
        // die Messlatte für den Chip. Bei den vollen Ketten je Memory-Encoder; es zählt der, der live läuft.
        val cpuChains = meta.optJSONObject("cpuKetten")
        fun expectation(key: String): JSONObject? {
            val c = cpuChains?.optJSONObject(key) ?: return null
            return if (key == "voll" || key == "vollQualcomm") c.optJSONObject(if (liveQcMemory) "qualcomm" else "eigener") else c
        }
        val slot = FrameSlot()
        val chains = listOf(
            "stufe2" to { buildFullTracker(ModelId.ATTN, qcMemory = false, appPrompt = false, metaDecoder = false) },
            "voll" to { buildFullTracker(liveAttn, liveQcMemory, appPrompt = true, metaDecoder = true) },
            "vollQualcomm" to { buildFullTracker(liveAttn, liveQcMemory, appPrompt = true, metaDecoder = false) },
            "schnell" to { buildFastTracker(quantized = false, metaDecoder = true) },
            "schnellQuantisiert" to { buildFastTracker(quantized = true, metaDecoder = true) },
        )
        // Puffer für das Einlesen der Bilder, einmal für alle Ketten.
        val pixels = IntArray(FrameSlot.ENC * FrameSlot.ENC)
        val row = FloatArray(FrameSlot.ENC * 3)
        ZipFile(file).use { zip ->
            val truth = Array(frames) { i -> readMask(zip, "maske_%02d.png".format(i)) }
            for ((key, make) in chains) {
                if (cancelWork) {
                    out.put("abgebrochen", true)
                    break
                }
                val t = try {
                    make()
                } catch (e: Exception) {
                    out.put(key, JSONObject().put("fehler", e.toString().take(300)))
                    null
                } ?: continue
                progress("Prüfvideo · ${t.name}")
                val ious = JSONArray()
                val found = JSONArray()
                var sum = 0.0
                var worst = 1.0
                try {
                    for (i in 0 until frames) {
                        readFrame(zip, "bild_%02d.png".format(i), slot, pixels, row)
                        if (i == 0) t.start(slot.enc, startU, startV) else t.step(slot.enc)
                        val v = iou(t.mask(), truth[i])
                        ious.put(r2(v))
                        found.put(t.present)
                        sum += v
                        worst = min(worst, v)
                    }
                    val result = JSONObject().put("name", t.name).put("maskeGleichMittel", r2(sum / frames))
                        .put("maskeGleichSchlechtestes", r2(worst)).put("jeBild", ious).put("objektGefunden", found)
                    expectation(key)?.let { c ->
                        val mean = c.optDouble("maskeGleichMittel", Double.NaN)
                        if (!mean.isNaN()) result.put("cpuNachbauMittel", r2(mean))
                        c.optJSONArray("jeBild")?.let { result.put("cpuNachbauJeBild", it) }
                    }
                    out.put(key, result)
                } catch (e: Exception) {
                    out.put(key, JSONObject().put("name", t.name).put("fehler", e.toString().take(300)).put("jeBild", ious))
                } finally {
                    t.reset()
                }
            }
        }
        out.put("bilder", frames)
        return out
    }

    /** Liest ein Bild des Prüfvideos als RGB 0 bis 1 in [slot]; [px] und [row] sind wiederverwendete Puffer. */
    private fun readFrame(zip: ZipFile, name: String, slot: FrameSlot, px: IntArray, row: FloatArray) {
        val bmp = decode(zip, name)
        try {
            require(bmp.width == FrameSlot.ENC && bmp.height == FrameSlot.ENC) { "$name: ${bmp.width}×${bmp.height}" }
            bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            val fb = slot.enc.duplicate().order(slot.enc.order()).apply { rewind() }.asFloatBuffer()
            for (y in 0 until bmp.height) {
                for (x in 0 until bmp.width) {
                    val p = px[y * bmp.width + x]
                    row[x * 3] = ((p ushr 16) and 0xFF) / 255f
                    row[x * 3 + 1] = ((p ushr 8) and 0xFF) / 255f
                    row[x * 3 + 2] = (p and 0xFF) / 255f
                }
                fb.put(row)
            }
            slot.enc.rewind()
        } finally {
            bmp.recycle()
        }
    }

    private fun readMask(zip: ZipFile, name: String): FloatArray {
        val bmp = decode(zip, name)
        try {
            require(bmp.width == MASK_SIZE && bmp.height == MASK_SIZE) { "$name: ${bmp.width}×${bmp.height}" }
            val px = IntArray(bmp.width * bmp.height)
            bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            return FloatArray(px.size) { if ((px[it] and 0xFF) >= 128) 1f else -1f }
        } finally {
            bmp.recycle()
        }
    }

    private fun decode(zip: ZipFile, name: String): Bitmap {
        val entry = zip.getEntry(name) ?: throw IllegalStateException("Prüfvideo: $name fehlt")
        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inScaled = false
        }
        return zip.getInputStream(entry).use { BitmapFactory.decodeStream(it, null, opts) }
            ?: throw IllegalStateException("Prüfvideo: $name nicht lesbar")
    }

    private fun buffer(values: FloatArray): ByteBuffer = directBuffer(values.size * 4).apply { asFloatBuffer().put(values); rewind() }

    private fun floats(b: ByteBuffer): FloatArray {
        val fb = b.duplicate().order(b.order()).apply { rewind() }.asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    /** Abweichung des Chips von der CPU: relativ (L2), größte einzelne und Korrelation. */
    private fun compare(npu: ByteBuffer, cpu: ByteBuffer): JSONObject {
        val a = floats(npu)
        val b = floats(cpu)
        val n = min(a.size, b.size)
        var diff2 = 0.0
        var ref2 = 0.0
        var maxDiff = 0.0
        var lo = Double.MAX_VALUE
        var hi = -Double.MAX_VALUE
        var sa = 0.0
        var sb = 0.0
        var saa = 0.0
        var sbb = 0.0
        var sab = 0.0
        var nonFinite = 0
        var count = 0
        for (i in 0 until n) {
            val x = a[i].toDouble()
            val y = b[i].toDouble()
            if (!x.isFinite()) {
                nonFinite++
                continue
            }
            count++
            val d = x - y
            diff2 += d * d
            ref2 += y * y
            maxDiff = max(maxDiff, kotlin.math.abs(d))
            lo = min(lo, y)
            hi = max(hi, y)
            sa += x
            sb += y
            saa += x * x
            sbb += y * y
            sab += x * y
        }
        val o = JSONObject()
        if (count == 0) return o.put("ungueltigeWerte", nonFinite)
        val corr = run {
            val cov = sab / count - (sa / count) * (sb / count)
            val va = saa / count - (sa / count) * (sa / count)
            val vb = sbb / count - (sb / count) * (sb / count)
            if (va > 0 && vb > 0) cov / sqrt(va * vb) else Double.NaN
        }
        o.put("abweichungProzent", r2(100 * sqrt(diff2 / max(ref2, 1e-30))))
        o.put("groessteAbweichung", r4(maxDiff))
        o.put("wertebereich", "${r2(lo)} bis ${r2(hi)}")
        if (!corr.isNaN()) o.put("korrelation", Math.round(corr * 10000) / 10000.0)
        if (a.size != b.size) o.put("groessenUnterschied", "${a.size} gegen ${b.size}")
        if (nonFinite > 0) o.put("ungueltigeWerte", nonFinite)
        return o
    }

    private fun iou(a: FloatArray, b: FloatArray): Double {
        var inter = 0
        var union = 0
        for (i in a.indices) {
            val x = a[i] > 0f
            val y = b[i] > 0f
            if (x && y) inter++
            if (x || y) union++
        }
        return if (union == 0) 1.0 else inter / union.toDouble()
    }

    // ------------------------------------------------------------ Live

    /**
     * Neues Ziel für die Verfolgung, (u, v) von 0 bis 1 im aufrechten Kamerabild (Hauptthread).
     * Tippt man binnen 1,5 s noch einmal auf dieselbe Stelle (höchstens 3 % daneben), nimmt die KI
     * die nächstkleinere glaubwürdige Maske ([SubjectPick.choose] mit Rang), reihum.
     */
    fun track(u: Float, v: Float) {
        val now = SystemClock.elapsedRealtime()
        val again = now - lastTapAt <= RETAP_MS && kotlin.math.abs(u - lastTapU) <= RETAP_DISTANCE && kotlin.math.abs(v - lastTapV) <= RETAP_DISTANCE
        lastTapRank = if (again) lastTapRank + 1 else 0
        lastTapU = u
        lastTapV = v
        lastTapAt = now
        pendingTap.set(Tap(u, v, lastTapRank))
    }

    /** Vor jedem Start von [runLive] aufrufen (auf dem Hauptthread), damit ein frühes Stopp nicht verloren geht. */
    fun beginLive() {
        stopLive = false
        pendingTap.set(null)
        lastTapAt = 0L
        lastTapRank = 0
    }

    fun requestStop() {
        stopLive = true
    }

    /** Name der Verfolgung, die live läuft, für den Bericht. */
    fun liveTrackerName(): String = tracker?.name ?: "keine"

    /**
     * Feste Taktung für den Strombedarf: höchstens alle [intervalMs] ein Durchgang, Tiefe jedes
     * [depthEvery]. Mal (0 = nie, außer beim Start), ohne Wärmeregelung. Gilt ab dem nächsten
     * Durchgang, auch mitten in [runLive].
     */
    data class FixedLive(val intervalMs: Long, val depthEvery: Int)

    /** Null = normal (Wärmeregelung); sonst feste Taktung, siehe [FixedLive]. */
    @Volatile var fixedLive: FixedLive? = null

    /** Aktuelle Wärmestufe laut Wärmeregelung (0 keine bis 6 Abschaltung). */
    val heatStatus: Int get() = heat.status

    /** Aktuelle Stufe der Wärmeregelung (folgt der Wärmestufe nach oben sofort, nach unten verzögert). */
    val heatLevel: Int get() = heat.level

    /**
     * Die letzten Starts der Verfolgung (Antippen oder Bildmitte) mit allen Masken, der Wahl und
     * einem kleinen Bild (gewählte Maske eingefärbt), für den Bericht. Das Bild entsteht auf einem
     * eigenen Thread; deshalb ist alles hier unter [startsLock].
     */
    private var starts = JSONArray()
    private val startsLock = Any()
    private var liveStartedAt = 0L

    /** Kopie der Starts des letzten Laufs. */
    fun startsJson(): JSONArray = synchronized(startsLock) { JSONArray(starts.toString()) }

    // Puffer für die Wahl des ganzen Objekts beim Start (nur der KI-Thread).
    private val depthOut = FloatArray(FrameSlot.DEPTH * FrameSlot.DEPTH)
    private val depthGrid = FloatArray(MASK_SIZE * MASK_SIZE)

    /**
     * Läuft, bis [requestStop] kommt: nimmt das jeweils neueste Kamerabild aus der Grafik-Pipeline,
     * verfolgt das Motiv, gibt die Maske an [maskSink] und rechnet die Tiefe (jedes 3. Mal, bei
     * Hitze seltener). Ein Start (erstes Bild oder Antippen) braucht ein Bild mit Tiefe: Mit ihr
     * wählt [SubjectPick] die Maske, die das ganze Objekt trifft. Fehlt sie, verfolgt die KI für
     * einen Durchgang weiter das alte Motiv, und die Pipeline legt dem nächsten Bild die Tiefe bei.
     */
    fun runLive(pool: FramePool) {
        val depth = models[ModelId.DEPTH] ?: error("Tiefenmodell nicht geladen")
        val t = tracker ?: error("EdgeTAM nicht geladen")
        val used = liveSet().mapNotNull { models[it] }
        liveRunning = true
        t.reset()
        heat.reset()
        synchronized(startsLock) { starts = JSONArray() }
        liveStartedAt = SystemClock.elapsedRealtime()
        nextTake = 0L
        wantDepth = true
        var sinceDepth = 1_000
        var lastStart = 0L
        var expectedMs = Double.NaN
        try {
            used.forEach { it.vote() }
            while (!stopLive) {
                // Wärmeregelung: Wird das Handy heiß, rechnet die KI seltener (siehe HeatGovernor).
                // Der Strombedarf misst mit fester Taktung.
                val fixed = fixedLive
                val gap = fixed?.intervalMs ?: heat.minIntervalMs()
                if (gap > 0 && lastStart > 0) {
                    val until = lastStart + gap
                    var now = SystemClock.elapsedRealtime()
                    if (now < until) {
                        nextTake = System.nanoTime() + (until - now) * 1_000_000L
                        heat.noteWait(until - now)
                        while (!stopLive && now < until) {
                            Thread.sleep(min(until - now, 50L))
                            now = SystemClock.elapsedRealtime()
                        }
                        if (stopLive) break
                    }
                }
                // Die Pipeline sieht an aiWaiting, dass die KI auf ein Bild wartet, und liefert eins.
                aiWaiting = true
                val slot = try {
                    pool.take(100)
                } finally {
                    aiWaiting = false
                } ?: continue
                try {
                    val start = System.nanoTime()
                    val cpuStart = Debug.threadCpuTimeNanos()
                    lastStart = SystemClock.elapsedRealtime()
                    // Startbild: neu angetippt oder der erste Durchgang (Mitte des Bildes). Es
                    // braucht die Tiefe; ohne sie bleibt der Tipp liegen (siehe oben).
                    val tap = pendingTap.get()
                    val wantStart = tap != null || !t.started
                    val startNow = wantStart && slot.hasDepth
                    if (!t.started && !startNow) {
                        wantDepth = true
                        continue
                    }
                    // Wärmeregelung: Bei Hitze fällt zuerst die Tiefe weg (live nur fürs Antippen
                    // gebraucht). Gleich zu Beginn entscheiden, auch fürs nächste Bild: Die Pipeline
                    // bereitet es schon während dieses Durchgangs vor und spart sich das
                    // Tiefenbild, wenn es nicht gebraucht wird.
                    val every = fixed?.depthEvery ?: heat.depthEvery()
                    val runDepth = startNow || (slot.hasDepth && every > 0 && sinceDepth + 1 >= every)
                    sinceDepth = if (runDepth) 0 else min(sinceDepth + 1, 1_000)
                    wantDepth = (every > 0 && sinceDepth + 1 >= every) || (wantStart && !startNow)
                    // Wann die KI voraussichtlich das nächste Bild nimmt: Danach richtet die
                    // Grafik-Pipeline das nächste Bild ein, damit es frisch ist.
                    nextTake = if (expectedMs.isNaN()) 0L else start + (max(expectedMs, gap.toDouble()) * 1e6).toLong()
                    // Wie alt das Kamerabild ist, wenn die KI es aufnimmt. Je nach Gerät zählt die
                    // Kamera ab Systemstart mit oder ohne Tiefschlaf.
                    val age = frameAgeMs(slot.timestampNs)
                    var depthNative = Double.NaN
                    if (runDepth) {
                        depth.run(arrayOf(slot.depth))
                        depthNative = depth.lastNativeMs
                    } else {
                        heat.noteDepthSkipped()
                    }
                    if (startNow) {
                        // Nur diesen Tipp abholen; kam inzwischen ein neuer, bleibt er für den nächsten Durchgang.
                        if (tap != null) pendingTap.compareAndSet(tap, null)
                        generation++
                        val u = tap?.u ?: 0.5f
                        val v = tap?.v ?: 0.5f
                        val rank = tap?.rank ?: 0
                        depth.output(0).asFloatBuffer().get(depthOut)
                        val c = slot.depthContent
                        SubjectPick.depthGrid(depthOut, FrameSlot.DEPTH, c.left, c.top, c.right, c.bottom, depthGrid)
                        t.start(slot.enc, u, v) { masks, ious, n ->
                            val t0 = System.nanoTime()
                            val cands = SubjectPick.evaluate(masks, n, ious, depthGrid, u, v)
                            val k = SubjectPick.choose(cands, rank)
                            noteStart(slot, masks, u, v, tap != null, rank, cands, k, SubjectPick.bestIou(cands), (System.nanoTime() - t0) / 1e6)
                            k
                        }
                    } else {
                        t.step(slot.enc)
                    }
                    // Die Maske gehört zu genau diesem Kamerabild (Zeitstempel); die Pipeline legt
                    // sie auf dasselbe Bild, wenn sie es verzögert ausgibt (siehe BokehProcessor).
                    val masks = maskSink
                    if (masks != null) {
                        PipelineMath.maskProbabilities(t.mask(), maskBytes)
                        masks.onMask(slot.timestampNs, maskBytes, t.present, generation)
                    }
                    val aiMs = (System.nanoTime() - start) / 1e6
                    val cpuEnd = Debug.threadCpuTimeNanos()
                    val cpuMs = if (cpuStart >= 0 && cpuEnd >= cpuStart) (cpuEnd - cpuStart) / 1e6 else Double.NaN
                    expectedMs = if (expectedMs.isNaN()) aiMs else expectedMs * 0.8 + aiMs * 0.2
                    // Das Ergebnis ist fertig, wenn das Bild so alt ist wie beim Aufnehmen plus die Rechenzeit.
                    stats.add(
                        aiMs = aiMs, depthMs = depthNative, encMs = t.encMs,
                        attnMs = if (startNow) Double.NaN else t.attnMs, decMs = t.decMs,
                        memMs = t.memMs, trackMs = t.totalMs, prepMs = slot.prepMs,
                        ageMs = age, resultAgeMs = age + aiMs, cpuMs = cpuMs,
                        present = t.present, restart = tap != null && startNow
                    )
                } finally {
                    pool.release(slot)
                }
            }
        } finally {
            used.forEach { it.release() }
            nextTake = 0L
            heat.finish()
            liveRunning = false
        }
    }

    /**
     * Ein Start der Verfolgung für den Bericht: alle Masken mit Merkmalen, die Wahl, was Meta
     * genommen hätte, und ein kleines Bild des KI-Eingangs mit der gewählten Maske (eingefärbt)
     * und dem Tipp (weißes Kreuz). Die Bildpunkte entstehen hier (schnell), das JPEG auf einem
     * eigenen Thread, damit die KI nicht wartet.
     */
    private fun noteStart(
        slot: FrameSlot, masks: FloatArray, u: Float, v: Float, tapped: Boolean, rank: Int,
        cands: List<SubjectPick.Candidate>, chosen: Int, meta: Int, ms: Double,
    ) {
        val list = JSONArray()
        for (c in cands) {
            list.put(JSONObject().put("flaeche", r4(c.area)).put("iouSchaetzung", r4(c.iou.toDouble()))
                .put("kontrast", num2(c.contrast)).put("hinten", r4(c.behind)).put("spanne", num2(c.spread))
                .put("tippAbstand", num2(c.tapDistance)).put("verschachtelt", r4(c.nested)).put("glaubwuerdig", c.plausible))
        }
        val o = JSONObject()
            .put("s", r1((SystemClock.elapsedRealtime() - liveStartedAt) / 1000.0))
            .put("angetippt", tapped).put("u", r4(u.toDouble())).put("v", r4(v.toDouble()))
            .put("erneutGetippt", rank).put("masken", list).put("wahl", chosen).put("wieMeta", meta).put("wahlMs", r1(ms))
        synchronized(startsLock) {
            if (starts.length() >= MAX_STARTS) starts.remove(0)
            starts.put(o)
        }
        try {
            val size = FrameSlot.THUMBNAIL
            val pixels = slot.thumbnailPixels(size)
            val plane = chosen * MASK_SIZE * MASK_SIZE
            for (y in 0 until size) {
                for (x in 0 until size) {
                    val mx = (x * MASK_SIZE / size).coerceIn(0, MASK_SIZE - 1)
                    val my = (y * MASK_SIZE / size).coerceIn(0, MASK_SIZE - 1)
                    if (masks[plane + my * MASK_SIZE + mx] > 0f) pixels[y * size + x] = tint(pixels[y * size + x])
                }
            }
            val cx = (u * size).toInt().coerceIn(0, size - 1)
            val cy = (v * size).toInt().coerceIn(0, size - 1)
            for (d in -3..3) {
                if (cx + d in 0 until size) pixels[cy * size + cx + d] = -1
                if (cy + d in 0 until size) pixels[(cy + d) * size + cx] = -1
            }
            Thread({
                try {
                    val jpeg = FrameSlot.jpegBase64(pixels, size)
                    synchronized(startsLock) { o.put("bildJpegBase64", jpeg) }
                } catch (_: Throwable) {
                }
            }, "start-bild").start()
        } catch (_: Throwable) {
        }
    }

    /** Bildpunkt halb mit Magenta mischen (gewählte Maske im Bild des Berichts). */
    private fun tint(argb: Int): Int {
        val r = (((argb shr 16) and 0xFF) + 255) / 2
        val g = ((argb shr 8) and 0xFF) / 2
        val b = ((argb and 0xFF) + 200) / 2
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

    private fun frameAgeMs(timestampNs: Long): Double {
        val realtime = (SystemClock.elapsedRealtimeNanos() - timestampNs) / 1e6
        if (realtime in 0.0..2000.0) return realtime
        val monotonic = (System.nanoTime() - timestampNs) / 1e6
        return if (monotonic in 0.0..2000.0) monotonic else Double.NaN
    }

    companion object {
        const val WARMUP = 3
        const val RUNS = 20
        const val MATRIX_RUNS = 10

        /** So viele Starts der Verfolgung behält der Bericht (die letzten). */
        const val MAX_STARTS = 12

        /** Erneutes Tippen: binnen dieser Zeit (ms) und höchstens so weit daneben (Bruchteil des Bildes). */
        const val RETAP_MS = 1500L
        const val RETAP_DISTANCE = 0.03f

        /** Die volle Kette, die live läuft: „voll“ mit unserem Decoder wie im Original, sonst „vollQualcomm“. */
        fun liveChainKey(metaDecoder: Boolean): String = if (metaDecoder) "voll" else "vollQualcomm"
    }
}

internal fun r4(v: Double): Double = Math.round(v * 10000.0) / 10000.0

/** Zahl für den Bericht, auf zwei Stellen; „ungültig“, wenn der Chip NaN oder Unendlich lieferte (das kann JSON nicht). */
internal fun num2(v: Double): Any = if (v.isFinite()) r2(v) else "ungültig"

/** Zeiten und Zähler der Live-Läufe, in Abschnitten (etwa alle 30 s) und insgesamt. */
internal class LiveStats {
    private class Series {
        var values = DoubleArray(256)
        var size = 0
        fun add(v: Double) {
            if (v.isNaN()) return
            if (size == values.size) values = values.copyOf(size * 2)
            values[size++] = v
        }
        fun timing() = Timing(values.copyOf(size))
        fun clear() {
            size = 0
        }
    }

    /** „kiThreadCpu“: Rechenzeit des KI-Threads je Durchgang; nahe „ki“ hieße, er wartet aktiv auf den KI-Chip. */
    private val names = listOf(
        "ki", "tiefe", "encoder", "speicherAttention", "decoder", "speicherEncoder", "verfolgung",
        "vorbereitung", "alterBild", "alterErgebnis", "kiThreadCpu"
    )
    private val window = names.associateWith { Series() }
    private val total = names.associateWith { Series() }
    private var windowStart = SystemClock.elapsedRealtime()
    private var totalStart = windowStart
    private var windowCycles = 0
    private var totalCycles = 0
    private var windowPresent = 0
    private var totalPresent = 0
    private var restarts = 0

    @Synchronized
    fun reset() {
        window.values.forEach { it.clear() }
        total.values.forEach { it.clear() }
        windowStart = SystemClock.elapsedRealtime()
        totalStart = windowStart
        windowCycles = 0
        totalCycles = 0
        windowPresent = 0
        totalPresent = 0
        restarts = 0
    }

    @Synchronized
    fun add(
        aiMs: Double, depthMs: Double, encMs: Double, attnMs: Double, decMs: Double, memMs: Double,
        trackMs: Double, prepMs: Double, ageMs: Double, resultAgeMs: Double, cpuMs: Double,
        present: Boolean, restart: Boolean
    ) {
        val values = listOf(aiMs, depthMs, encMs, attnMs, decMs, memMs, trackMs, prepMs, ageMs, resultAgeMs, cpuMs)
        for ((i, name) in names.withIndex()) {
            window.getValue(name).add(values[i])
            total.getValue(name).add(values[i])
        }
        windowCycles++
        totalCycles++
        if (present) {
            windowPresent++
            totalPresent++
        }
        if (restart) restarts++
    }

    /** Alle Bilder seit dem letzten Zurücksetzen. */
    @Synchronized
    fun cycles(): Int = totalCycles

    /** Bilder je Sekunde im laufenden Abschnitt. */
    @Synchronized
    fun currentHz(): Double {
        val s = (SystemClock.elapsedRealtime() - windowStart) / 1000.0
        return if (s > 0.3) windowCycles / s else 0.0
    }

    /** Abschnitt abschließen und neuen beginnen. */
    @Synchronized
    fun closeWindow(): JSONObject {
        val o = summary(window, windowCycles, windowPresent, windowStart)
        window.values.forEach { it.clear() }
        windowCycles = 0
        windowPresent = 0
        windowStart = SystemClock.elapsedRealtime()
        return o
    }

    @Synchronized
    fun totalJson(): JSONObject = summary(total, totalCycles, totalPresent, totalStart)
        .put("neuAngetippt", restarts)

    private fun summary(series: Map<String, Series>, cycles: Int, present: Int, since: Long): JSONObject {
        val seconds = (SystemClock.elapsedRealtime() - since) / 1000.0
        val o = JSONObject()
        o.put("bilder", cycles)
        o.put("sekunden", r1(seconds))
        if (seconds > 0) o.put("hz", r1(cycles / seconds))
        if (cycles > 0) o.put("objektGefundenProzent", Math.round(present * 100.0 / cycles))
        val ms = JSONObject()
        for ((name, s) in series) {
            if (s.size == 0) continue
            val t = s.timing()
            ms.put(name, JSONObject().put("median", r1(t.median)).put("p90", r1(t.p90)).put("max", r1(t.max)))
        }
        o.put("ms", ms)
        return o
    }
}
