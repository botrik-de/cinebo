package de.botrik.cinebo.test

import android.content.Context
import android.os.SystemClock
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.util.EnumMap
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Alle Modelle der Test-App. [live] heißt: läuft bei Live-Test, Dauertest und Vorschau mit. */
internal enum class ModelId(val asset: String, val title: String, val fp16: Boolean, val live: Boolean) {
    DEPTH("depth_anything_v2.tflite", "Tiefe · Depth Anything V2 Small", true, true),
    ENC("edgetam_encoder.tflite", "EdgeTAM · Encoder", true, true),
    DEC("edgetam_decoder.tflite", "EdgeTAM · Decoder", true, true),
    ATTN("edgetam_memory_attention.tflite", "EdgeTAM · Memory-Attention", true, true),
    ATTN4("edgetam_memory_attention_q4.tflite", "EdgeTAM · Memory-Attention in 4 Blöcken", true, false),
    MEMENC("edgetam_memory_encoder.tflite", "EdgeTAM · Memory-Encoder", true, true),
    MEMENC_QC("edgetam_memory_encoder_qualcomm.tflite", "EdgeTAM · Memory-Encoder von Qualcomm", true, false),
    ENC_Q("edgetam_encoder_w8a8.tflite", "EdgeTAM · Encoder quantisiert", false, false),
    DEC_Q("edgetam_decoder_w8a8.tflite", "EdgeTAM · Decoder quantisiert", false, false),
    MEMENC_Q("edgetam_memory_encoder_w8a8.tflite", "EdgeTAM · Memory-Encoder quantisiert", false, false);

    val key: String get() = name.lowercase()
}

/** Was die Vorschau während der KI-Läufe zeigt. Wird auf dem KI-Thread aufgerufen. */
internal interface LiveSink {
    fun onPictures(depth: IntArray, mask: IntArray, present: Boolean)
}

/**
 * Lädt die Modelle auf den KI-Chip, misst sie einzeln, prüft ihre Genauigkeit gegen die
 * CPU und lässt beide zusammen auf den Kamerabildern laufen. Alles läuft auf einem
 * eigenen Thread („ki“); die Methoden hier nur über [dispatcher] aufrufen.
 */
internal class AiEngine(private val context: Context) {

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
    private var noObjPtr: FloatArray? = null
    private var tracker: EdgeTamTracker? = null
    private var cacheCleaned = false

    /** Welche Memory-Attention live läuft; nach der Einzelmessung die schnellere. */
    var liveAttn = ModelId.ATTN
        private set

    /** Ein Standbild für den Fall, dass die Kamera neben 4K keine Bilder für die KI liefert. */
    private var still: FrameSlot? = null

    @Volatile var liveRunning = false
        private set
    @Volatile private var stopLive = false
    @Volatile private var pendingTap: FloatArray? = null

    val stats = LiveStats()
    private val depthPainter = DepthPainter(DEPTH_W, DEPTH_H)
    private val depthPixels = Array(2) { IntArray(DEPTH_W * DEPTH_H) }
    private val maskPixels = Array(2) { IntArray(MaskPainter.SIZE * MaskPainter.SIZE) }
    private var flip = 0

    fun loaded(id: ModelId): Boolean = models.containsKey(id)

    fun liveReady(): Boolean = liveSet().all { models.containsKey(it) } && tracker != null

    private fun liveSet(): List<ModelId> = listOf(ModelId.DEPTH, ModelId.ENC, ModelId.DEC, liveAttn, ModelId.MEMENC)

    // ------------------------------------------------------------ Laden

    /**
     * Lädt die angegebenen Modelle auf den KI-Chip, soweit noch nicht geladen. Ein Fehler bei
     * einem Modell hält die anderen nicht auf; er steht im Ergebnis.
     */
    fun load(ids: List<ModelId>, progress: (done: Int, total: Int, id: ModelId) -> Unit): JSONObject {
        if (!cacheCleaned) {
            Npu.cleanCache(context)
            cacheCleaned = true
        }
        val out = JSONObject()
        val start = SystemClock.elapsedRealtime()
        val todo = ids.filter { !models.containsKey(it) }
        todo.forEachIndexed { i, id ->
            progress(i, todo.size, id)
            try {
                val m = NpuModel.load(context, id.asset, NpuConfig(fp16 = id.fp16))
                models[id] = m
                out.put(id.key, m.describe())
            } catch (e: Throwable) {
                out.put(id.key, JSONObject().put("datei", id.asset).put("fehler", e.toString().take(600))
                    .put("protokoll", LogProbe.since(System.currentTimeMillis() - 30_000)))
            }
        }
        progress(todo.size, todo.size, todo.lastOrNull() ?: ModelId.DEPTH)
        if (noObjPtr == null) noObjPtr = readNoObjPtr()
        buildTracker()
        out.put("gesamtMs", SystemClock.elapsedRealtime() - start)
        val cache = JSONArray()
        Npu.cacheDir(context).listFiles()?.sortedBy { it.name }?.forEach { cache.put("${it.name} ${r1(it.length() / 1e6)} MB") }
        out.put("zwischenspeicher", cache)
        return out
    }

    fun unload(ids: Collection<ModelId>) {
        for (id in ids) models.remove(id)?.close()
        if (liveSet().any { !models.containsKey(it) }) tracker = null
    }

    fun unloadExtras() = unload(ModelId.entries.filter { it !in liveSet() })

    fun closeAll() {
        unload(ModelId.entries)
        still = null
    }

    private fun readNoObjPtr(): FloatArray {
        val text = context.assets.open("${NpuModel.FOLDER}/edgetam_konstanten.json").bufferedReader().use { it.readText() }
        val list = JSONObject(text).getJSONArray("noObjPtr")
        return FloatArray(list.length()) { list.getDouble(it).toFloat() }
    }

    private fun buildTracker() {
        val enc = models[ModelId.ENC]
        val dec = models[ModelId.DEC]
        val attn = models[liveAttn]
        val mem = models[ModelId.MEMENC]
        val ptr = noObjPtr
        tracker = if (enc != null && dec != null && attn != null && mem != null && ptr != null) {
            EdgeTamTracker(enc, dec, attn, mem, ptr)
        } else {
            null
        }
    }

    // ------------------------------------------------------------ Einzelmessung

    /**
     * Misst jedes geladene Modell einzeln auf dem KI-Chip (3 Läufe Einlaufen, dann 20) und
     * danach den ganzen Ablauf je Bild. Wählt die schnellere Memory-Attention für live.
     */
    fun bench(frame: FrameSlot, progress: (String) -> Unit): JSONObject {
        still = FrameSlot().also { it.copyFrom(frame) }
        val out = JSONObject()
        val perModel = JSONObject()
        val enc = models[ModelId.ENC]
        val dec = models[ModelId.DEC]
        val center = directBuffer(16).apply { putFloat(0.5f).putFloat(0.5f).putFloat(0f).putFloat(0f).rewind() }
        val labels = directBuffer(8).apply { putFloat(1f).putFloat(-1f).rewind() }

        models[ModelId.DEPTH]?.let { m ->
            progress(ModelId.DEPTH.title)
            perModel.put(ModelId.DEPTH.key, time(m) { m.run(arrayOf(frame.depth)) })
        }
        if (enc != null) {
            progress(ModelId.ENC.title)
            val inputs = arrayOfNulls<Any>(3)
            inputs[enc.inputIndex("image")] = frame.enc
            inputs[enc.inputIndex("unnorm_coords")] = center
            inputs[enc.inputIndex("labels")] = labels
            @Suppress("UNCHECKED_CAST")
            perModel.put(ModelId.ENC.key, time(enc) { enc.run(inputs as Array<Any>) })
        }
        if (enc != null && dec != null) {
            progress(ModelId.DEC.title)
            val inputs = arrayOfNulls<Any>(4)
            inputs[dec.inputIndex("image_embeddings")] = enc.output(enc.outputIndex("pix_feat"))
            inputs[dec.inputIndex("high_res_features1")] = enc.output(enc.outputIndex("high_res_features1"))
            inputs[dec.inputIndex("high_res_features2")] = enc.output(enc.outputIndex("high_res_features2"))
            inputs[dec.inputIndex("sparse_embedding")] = enc.output(enc.outputIndex("sparse_embedding"))
            @Suppress("UNCHECKED_CAST")
            perModel.put(ModelId.DEC.key, time(dec) { dec.run(inputs as Array<Any>) })
        }
        val emb = enc?.output(enc.outputIndex("image_embeddings"))
        for (id in listOf(ModelId.ATTN, ModelId.ATTN4)) {
            val m = models[id] ?: continue
            if (emb == null) continue
            progress(id.title)
            val bank = m.newInput(1)
            val ptrs = m.newInput(2)
            val bias = m.newInput(3) // alles 0: alle Plätze gültig, rechnet voll
            perModel.put(id.key, time(m) { m.run(arrayOf(emb, bank, ptrs, bias)) })
        }
        models[ModelId.MEMENC]?.let { m ->
            if (emb == null) return@let
            progress(ModelId.MEMENC.title)
            val mask = m.newInput(1)
            val cond = m.newInput(2)
            perModel.put(ModelId.MEMENC.key, time(m) { m.run(arrayOf(emb, mask, cond)) })
        }
        // Nur zum Vergleich, mit Nullen gefüllt: Rechenzeit hängt nicht vom Inhalt ab.
        for (id in listOf(ModelId.MEMENC_QC, ModelId.ENC_Q, ModelId.DEC_Q, ModelId.MEMENC_Q)) {
            val m = models[id] ?: continue
            progress(id.title)
            val inputs: Array<Any> = Array(m.inputCount) { m.newInput(it) }
            perModel.put(id.key, time(m) { m.run(inputs) })
        }
        out.put("modelle", perModel)

        // Schnellere Memory-Attention für live wählen.
        val a1 = perModel.optJSONObject(ModelId.ATTN.key)?.optJSONObject("imModell")?.optDouble("medianMs", Double.NaN) ?: Double.NaN
        val a4 = perModel.optJSONObject(ModelId.ATTN4.key)?.optJSONObject("imModell")?.optDouble("medianMs", Double.NaN) ?: Double.NaN
        val chosen = if (!a4.isNaN() && (a1.isNaN() || a4 < a1 * 0.97)) ModelId.ATTN4 else ModelId.ATTN
        if (chosen != liveAttn && models.containsKey(chosen)) {
            liveAttn = chosen
            buildTracker()
        }
        out.put("liveMemoryAttention", liveAttn.key)

        // Ganzer Ablauf je Bild: Tiefe, dann Verfolgung mit Speicher.
        val t = tracker
        val depth = models[ModelId.DEPTH]
        if (t != null && depth != null) {
            progress("Tiefe und Verfolgung zusammen")
            t.start(frame.enc, 0.5f, 0.5f)
            repeat(WARMUP) {
                depth.run(arrayOf(frame.depth))
                t.step(frame.enc)
            }
            val total = DoubleArray(RUNS)
            val depthMs = DoubleArray(RUNS)
            val trackMs = DoubleArray(RUNS)
            val parts = Array(4) { DoubleArray(RUNS) }
            for (i in 0 until RUNS) {
                val s = System.nanoTime()
                depthMs[i] = depth.run(arrayOf(frame.depth))
                t.step(frame.enc)
                total[i] = (System.nanoTime() - s) / 1e6
                trackMs[i] = t.totalMs
                parts[0][i] = t.encMs
                parts[1][i] = t.attnMs
                parts[2][i] = t.decMs
                parts[3][i] = t.memMs
            }
            out.put(
                "jeBild", JSONObject()
                    .put("beide", Timing(total).toJson())
                    .put("tiefe", Timing(depthMs).toJson())
                    .put("verfolgung", Timing(trackMs).toJson())
                    .put("encoder", Timing(parts[0]).toJson())
                    .put("speicherAttention", Timing(parts[1]).toJson())
                    .put("decoder", Timing(parts[2]).toJson())
                    .put("speicherEncoder", Timing(parts[3]).toJson())
                    .put("objektGefunden", t.present)
            )
            t.reset()
        }
        return out
    }

    private fun time(m: NpuModel, run: () -> Double): JSONObject {
        repeat(WARMUP) { run() }
        val total = DoubleArray(RUNS)
        val native = DoubleArray(RUNS)
        for (i in 0 until RUNS) {
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
     */
    fun accuracy(frame: FrameSlot, progress: (String) -> Unit): JSONObject {
        val out = JSONObject()
        val cpu = EnumMap<ModelId, NpuModel>(ModelId::class.java)
        try {
            for (id in listOf(ModelId.DEPTH, ModelId.ENC, ModelId.DEC, ModelId.ATTN, ModelId.MEMENC)) {
                if (!models.containsKey(id)) continue
                progress("Lädt ${id.title} für die CPU")
                cpu[id] = NpuModel.load(context, id.asset, null, threads = 4)
            }
            // Tiefe
            val dn = models[ModelId.DEPTH]
            val dc = cpu[ModelId.DEPTH]
            if (dn != null && dc != null) {
                progress("Vergleicht die Tiefe")
                val cpuMs = dc.run(arrayOf(frame.depth))
                dn.run(arrayOf(frame.depth))
                val r = compare(dn.output(0), dc.output(0))
                r.put("cpuMs", r1(cpuMs))
                out.put("tiefe", r)
            }
            // EdgeTAM, Modell für Modell mit den Eingängen der CPU
            val ec = cpu[ModelId.ENC]
            val en = models[ModelId.ENC]
            val dcd = cpu[ModelId.DEC]
            val dnd = models[ModelId.DEC]
            val mc = cpu[ModelId.MEMENC]
            val mn = models[ModelId.MEMENC]
            val ac = cpu[ModelId.ATTN]
            val an = models[ModelId.ATTN]
            if (ec != null && en != null && dcd != null && dnd != null && mc != null && mn != null && ac != null && an != null) {
                progress("Vergleicht EdgeTAM")
                val center = directBuffer(16).apply { putFloat(0.5f).putFloat(0.5f).putFloat(0f).putFloat(0f).rewind() }
                val labels = directBuffer(8).apply { putFloat(1f).putFloat(-1f).rewind() }
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
                out.put("encoder", eo)

                val decIn = arrayOfNulls<Any>(4)
                decIn[dcd.inputIndex("image_embeddings")] = ec.output(ec.outputIndex("pix_feat"))
                decIn[dcd.inputIndex("high_res_features1")] = ec.output(ec.outputIndex("high_res_features1"))
                decIn[dcd.inputIndex("high_res_features2")] = ec.output(ec.outputIndex("high_res_features2"))
                decIn[dcd.inputIndex("sparse_embedding")] = ec.output(ec.outputIndex("sparse_embedding"))
                @Suppress("UNCHECKED_CAST")
                val decCpuMs = dcd.run(decIn as Array<Any>)
                dnd.run(decIn)
                val maskC = floats(dcd.output(dcd.outputIndex("masks")))
                val maskN = floats(dnd.output(dnd.outputIndex("masks")))
                out.put(
                    "decoder", JSONObject()
                        .put("maskeGleich", r2(iou(maskN, maskC)))
                        .put("maskeCpuAnteil", r2(maskC.count { it > 0f } / maskC.size.toDouble()))
                        .put("zeiger", compare(dnd.output(dnd.outputIndex("obj_ptr")), dcd.output(dcd.outputIndex("obj_ptr"))))
                        .put("objektWertCpu", r2(dcd.output(dcd.outputIndex("object_score_logits")).getFloat(0).toDouble()))
                        .put("objektWertChip", r2(dnd.output(dnd.outputIndex("object_score_logits")).getFloat(0).toDouble()))
                        .put("cpuMs", r1(decCpuMs))
                )

                val emb = ec.output(ec.outputIndex("image_embeddings"))
                val maskBuf = directBuffer(maskC.size * 4).apply { asFloatBuffer().put(maskC); rewind() }
                val cond = directBuffer(4).apply { putFloat(1f).rewind() }
                val memCpuMs = mc.run(arrayOf(emb, maskBuf, cond))
                mn.run(arrayOf(emb, maskBuf, cond))
                val memC = floats(mc.output(0))
                out.put("speicherEncoder", compare(mn.output(0), mc.output(0)).put("cpuMs", r1(memCpuMs)))

                // Memory-Attention mit einem echten Speicher: Startbild in Platz 0 und 6.
                val bank = directBuffer(EdgeTamTracker.N_MEM * EdgeTamTracker.TOK_MEM * EdgeTamTracker.MEM_DIM * 4)
                val slot = EdgeTamTracker.TOK_MEM * EdgeTamTracker.MEM_DIM
                bank.asFloatBuffer().apply { put(memC); position(6 * slot); put(memC) }
                val ptrs = directBuffer(EdgeTamTracker.N_PTR * EdgeTamTracker.D * 4)
                val ptr = floats(dcd.output(dcd.outputIndex("obj_ptr")))
                ptrs.asFloatBuffer().apply { put(ptr); put(ptr) }
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
                val attnCpuMs = ac.run(arrayOf(emb, bank, ptrs, bias))
                an.run(arrayOf(emb, bank, ptrs, bias))
                out.put("speicherAttention", compare(an.output(0), ac.output(0)).put("cpuMs", r1(attnCpuMs)))

                // Ganze Verfolgung doppelt: Startbild in der Mitte, dann zwei Folgebilder.
                progress("Vergleicht die ganze Verfolgung")
                val ptrNo = noObjPtr
                if (ptrNo != null) {
                    val tc = EdgeTamTracker(ec, dcd, ac, mc, ptrNo)
                    val tn = tracker ?: EdgeTamTracker(en, dnd, an, mn, ptrNo)
                    tc.start(frame.enc, 0.5f, 0.5f)
                    val startC = tc.mask().copyOf()
                    tn.start(frame.enc, 0.5f, 0.5f)
                    val startN = tn.mask().copyOf()
                    repeat(2) {
                        tc.step(frame.enc)
                        tn.step(frame.enc)
                    }
                    out.put(
                        "verfolgung", JSONObject()
                            .put("startMaskeGleich", r2(iou(startN, startC)))
                            .put("folgeMaskeGleich", r2(iou(tn.mask(), tc.mask())))
                            .put("objektCpu", tc.present)
                            .put("objektChip", tn.present)
                            .put("maskeAnteil", r2(tc.mask().count { it > 0f } / startC.size.toDouble()))
                            .put("cpuMsJeBild", r1(tc.totalMs))
                    )
                    tn.reset()
                }
            }
        } finally {
            for (m in cpu.values) m.close()
        }
        return out
    }

    private fun floats(b: ByteBuffer): FloatArray {
        val fb = b.asFloatBuffer()
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
        for (i in 0 until n) {
            val x = a[i].toDouble()
            val y = b[i].toDouble()
            if (!x.isFinite()) {
                nonFinite++
                continue
            }
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
        val corr = run {
            val cov = sab / n - (sa / n) * (sb / n)
            val va = saa / n - (sa / n) * (sa / n)
            val vb = sbb / n - (sb / n) * (sb / n)
            if (va > 0 && vb > 0) cov / sqrt(va * vb) else Double.NaN
        }
        val o = JSONObject()
        o.put("abweichungProzent", r2(100 * sqrt(diff2 / max(ref2, 1e-30))))
        o.put("groessteAbweichung", r2(maxDiff))
        o.put("wertebereich", "${r2(lo)} bis ${r2(hi)}")
        if (!corr.isNaN()) o.put("korrelation", Math.round(corr * 10000) / 10000.0)
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

    /** Neues Ziel für die Verfolgung, (u, v) von 0 bis 1 im aufrechten Kamerabild. */
    fun track(u: Float, v: Float) {
        pendingTap = floatArrayOf(u, v)
    }

    /** Vor jedem Start von [runLive] aufrufen (auf dem Hauptthread), damit ein frühes Stopp nicht verloren geht. */
    fun beginLive() {
        stopLive = false
    }

    fun requestStop() {
        stopLive = true
    }

    /**
     * Läuft, bis [requestStop] kommt: nimmt das jeweils neueste Kamerabild, rechnet die Tiefe
     * und verfolgt das Objekt. Kommen länger keine Kamerabilder und gibt es ein Standbild,
     * rechnet sie darauf weiter (das steht dann im Bericht).
     */
    fun runLive(pool: FramePool, sink: LiveSink?, allowStill: Boolean) {
        val depth = models[ModelId.DEPTH] ?: error("Tiefenmodell nicht geladen")
        val t = tracker ?: error("EdgeTAM nicht geladen")
        liveRunning = true
        t.reset()
        depthPainter.reset()
        var lastFrame = SystemClock.elapsedRealtime()
        try {
            while (!stopLive) {
                val taken = pool.take(100)
                val slot: FrameSlot
                val fromPool: Boolean
                if (taken != null) {
                    slot = taken
                    fromPool = true
                    lastFrame = SystemClock.elapsedRealtime()
                } else {
                    val s = still
                    if (!allowStill || s == null || SystemClock.elapsedRealtime() - lastFrame < 600) continue
                    slot = s
                    fromPool = false
                }
                try {
                    val start = System.nanoTime()
                    depth.run(arrayOf(slot.depth))
                    val depthNative = depth.lastNativeMs
                    val tap = pendingTap
                    if (tap != null) pendingTap = null
                    if (tap != null || !t.started) {
                        t.start(slot.enc, tap?.get(0) ?: 0.5f, tap?.get(1) ?: 0.5f)
                    } else {
                        t.step(slot.enc)
                    }
                    val aiMs = (System.nanoTime() - start) / 1e6
                    val ageMs = (SystemClock.elapsedRealtimeNanos() - slot.timestampNs) / 1e6
                    stats.add(
                        aiMs = aiMs, depthMs = depthNative, encMs = t.encMs, attnMs = t.attnMs, decMs = t.decMs,
                        memMs = t.memMs, trackMs = t.totalMs, prepMs = slot.prepMs,
                        ageMs = if (fromPool && ageMs > 0 && ageMs < 2000) ageMs else Double.NaN,
                        present = t.present, restart = tap != null, still = !fromPool
                    )
                    if (sink != null) {
                        flip = 1 - flip
                        depthPainter.paint(depth.output(0), slot.depthContent, depthPixels[flip])
                        MaskPainter.paint(t.mask(), maskPixels[flip], MASK_COLOR)
                        sink.onPictures(depthPixels[flip], maskPixels[flip], t.present)
                    }
                } finally {
                    if (fromPool) pool.release(slot)
                }
                if (!fromPool) Thread.sleep(25) // Standbild: im Takt der Kamera bleiben
            }
        } finally {
            liveRunning = false
        }
    }

    companion object {
        const val WARMUP = 3
        const val RUNS = 20
        const val DEPTH_W = 144
        const val DEPTH_H = 256
        val MASK_COLOR = Palette.amber
    }
}

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

    private val names = listOf("ki", "tiefe", "encoder", "speicherAttention", "decoder", "speicherEncoder", "verfolgung", "vorbereitung", "alterBild")
    private val window = names.associateWith { Series() }
    private val total = names.associateWith { Series() }
    private var windowStart = SystemClock.elapsedRealtime()
    private var totalStart = windowStart
    private var windowCycles = 0
    private var totalCycles = 0
    private var windowPresent = 0
    private var totalPresent = 0
    private var restarts = 0
    private var stillCycles = 0

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
        stillCycles = 0
    }

    @Synchronized
    fun add(
        aiMs: Double, depthMs: Double, encMs: Double, attnMs: Double, decMs: Double, memMs: Double,
        trackMs: Double, prepMs: Double, ageMs: Double, present: Boolean, restart: Boolean, still: Boolean
    ) {
        val values = listOf(aiMs, depthMs, encMs, if (restart) Double.NaN else attnMs, decMs, memMs, trackMs, prepMs, ageMs)
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
        if (still) stillCycles++
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
        .put("aufStandbild", stillCycles)

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
