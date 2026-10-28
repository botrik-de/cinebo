package de.botrik.cinebo.test

import org.json.JSONObject
import java.nio.ByteBuffer

/** Gemeinsame Sicht auf beide Verfolgungen, für Messung, Vorschau und Dauertest. */
internal interface Tracker {
    val name: String
    val started: Boolean
    val present: Boolean
    val objectScore: Float

    /** Zeiten des letzten Bildes in ms (im Modell gerechnet) und insgesamt. */
    val encMs: Double
    val attnMs: Double
    val decMs: Double
    val memMs: Double
    val convertMs: Double
    val totalMs: Double

    /** Ob ein Start unter mehreren Masken wählen kann (sonst nimmt er die einzige des Decoders). */
    val canChoose: Boolean

    /**
     * Startbild: Objekt an Stelle (u, v), beides 0 bis 1 im aufrechten Kamerabild. Mit [chooser]
     * wählt die App unter den Masken des Decoders (sonst wie im Original die beste Schätzung).
     */
    fun start(image: ByteBuffer, u: Float, v: Float, chooser: StartChooser? = null)
    fun step(image: ByteBuffer)

    /** Maske des letzten Bildes (256 × 256 Logits, größer 0 heißt Objekt). */
    fun mask(): FloatArray
    fun reset()
}

/** Wählt beim Startbild unter den Masken des Decoders (siehe [SubjectPick]). */
internal fun interface StartChooser {
    /** [masks]: [count] Masken zu je 256 · 256 Logits hintereinander, [ious] ihre Schätzungen; gibt den Index zurück. */
    fun choose(masks: FloatArray, ious: FloatArray, count: Int): Int
}

/**
 * Verfolgt ein angetipptes Objekt mit EdgeTAM, Bild für Bild, mit Speicher.
 *
 * Der Encoder stammt von Qualcomm, die Memory-Attention ist unser Export
 * (tools/export/edgetam_export.py). Den Decoder gibt es als Qualcomms oder als unseren, der wie
 * Metas Videoverfolgung rechnet ([MaskDecoder]); den Memory-Encoder als unseren Export oder als
 * Qualcomms mit unserer Vorbereitung ([MemoryWriter]). Der Speicher hält wie im Original das
 * Startbild und die sechs letzten Bilder, dazu bis zu 16 Objekt-Zeiger. Ist [prompt] gesetzt,
 * rechnet die App die Eingabe-Einbettung selbst, sonst kommt sie aus dem Encoder (wie in Stufe 2,
 * nur mit Qualcomms Decoder). Nur vom KI-Thread benutzen.
 */
internal class EdgeTamTracker(
    private val encoder: ImageEncoder,
    private val decoder: MaskDecoder,
    private val attn: NpuModel,
    private val memory: MemoryWriter,
    private val noObjPtr: FloatArray,
    private val prompt: PromptEncoder?,
    private val coords: ByteBuffer,
    private val labels: ByteBuffer,
) : Tracker {
    private val maskBuf = directBuffer(MASK_SIZE * MASK_SIZE * 4)
    private val sparseBuf = decoder.newSparse()
    private val sparseScratch = FloatArray((decoder.points + 1) * D)
    private val promptU = FloatArray(decoder.points)
    private val promptV = FloatArray(decoder.points)
    private val promptLabels = IntArray(decoder.points)
    private val bank = attn.newInput(1)
    private val ptrs = attn.newInput(2)
    private val bias = attn.newInput(3)

    private val maskFloats = FloatArray(MASK_SIZE * MASK_SIZE)
    private val hidden = FloatArray(maskFloats.size) { -1024f }
    private val zeroBias = FloatArray(TOK_MEM) { 0f }
    private val maskedBias = FloatArray(TOK_MEM) { MASK }

    private var condMem: FloatArray? = null
    private val condPtr = FloatArray(D)
    private val recentMem = ArrayDeque<FloatArray>()
    private val recentPtr = ArrayDeque<FloatArray>()

    override val name: String = "voll · Encoder ${encoder.name} · Decoder ${decoder.name} · " +
        "${attn.asset.removePrefix("edgetam_").removeSuffix(".tflite")} · Memory-Encoder ${memory.name}" +
        if (prompt != null) " · Einbettung App" else " · Einbettung Encoder"

    init {
        require(bank.capacity() == N_MEM * TOK_MEM * MEM_DIM * 4) { "Memory-Attention hat eine andere Speichergröße" }
        require(ptrs.capacity() == N_PTR * D * 4 && bias.capacity() == N_K * 4)
        require(noObjPtr.size == D)
        require(prompt != null || (encoder.sparse != null && decoder.points == 2)) {
            "Ohne Einbettung aus der App braucht es die des Encoders, und die passt nur zu Qualcomms Decoder"
        }
    }

    override var encMs = 0.0; private set
    override var attnMs = 0.0; private set
    override var decMs = 0.0; private set
    override var memMs = 0.0; private set
    override var convertMs = 0.0; private set
    override var totalMs = 0.0; private set
    override var present = false; private set
    override var objectScore = 0f; private set
    var steps = 0L; private set

    override val started: Boolean get() = condMem != null
    override val canChoose: Boolean get() = decoder.candidates > 1

    /** Masken und Schätzungen des Startbilds für [StartChooser]; erst beim ersten Bedarf angelegt. */
    private var candMasks: FloatArray? = null
    private val candIous = FloatArray(3)

    override fun start(image: ByteBuffer, u: Float, v: Float, chooser: StartChooser?) {
        val t = System.nanoTime()
        point(u, v, positive = true)
        encoder.run(image, coords, labels)
        encMs = encoder.lastMs
        convertMs = encoder.convertMs
        writePrompt(u, v, 1)
        // Beim Startbild geht pix_feat (mit no_mem_embed) direkt in den Decoder.
        decode(encoder.pixFeat)
        // Die App wählt die Maske selbst (ganzes Objekt statt angetippter Fläche); Maske und Zeiger
        // des Startbilds und damit der Speicher kommen dann von dieser Maske.
        val n = decoder.candidates
        if (chooser != null && present && n > 1) {
            val masks = candMasks ?: FloatArray(n * MASK_SIZE * MASK_SIZE).also { candMasks = it }
            decoder.candidateMasks(masks, candIous)
            val k = chooser.choose(masks, candIous, n)
            if (k in 0 until n && k != decoder.lastChoice) decoder.select(k, maskBuf, currentPtr)
        }
        attnMs = 0.0
        val mem = encodeMemory(encoder.embeddings, startFrame = true)
        condMem = mem
        System.arraycopy(currentPtr, 0, condPtr, 0, D)
        recentMem.clear()
        recentPtr.clear()
        steps = 1
        totalMs = (System.nanoTime() - t) / 1e6
    }

    /** Folgebild: findet das Objekt mit Hilfe des Speichers wieder. */
    override fun step(image: ByteBuffer) {
        check(started) { "Erst ein Objekt antippen" }
        val t = System.nanoTime()
        point(0f, 0f, positive = false)
        encoder.run(image, coords, labels)
        encMs = encoder.lastMs
        convertMs = encoder.convertMs
        writePrompt(0f, 0f, -1)
        val emb = encoder.embeddings
        fillMemory()
        attn.run(arrayOf(emb, bank, ptrs, bias))
        attnMs = attn.lastNativeMs
        decode(attn.output(0))
        val mem = encodeMemory(encoder.embeddings, startFrame = false)
        recentMem.addLast(mem)
        while (recentMem.size > N_MEM - 1) recentMem.removeFirst()
        recentPtr.addLast(currentPtr.copyOf())
        while (recentPtr.size > N_PTR - 1) recentPtr.removeFirst()
        steps++
        totalMs = (System.nanoTime() - t) / 1e6
    }

    override fun mask(): FloatArray {
        maskBuf.rewind()
        maskBuf.asFloatBuffer().get(maskFloats)
        return maskFloats
    }

    override fun reset() {
        condMem = null
        recentMem.clear()
        recentPtr.clear()
        present = false
        steps = 0
    }

    fun timesJson(): JSONObject = JSONObject()
        .put("encoderMs", r2(encMs))
        .put("speicherAttentionMs", r2(attnMs))
        .put("decoderMs", r2(decMs))
        .put("speicherEncoderMs", r2(memMs))
        .put("gesamtMs", r2(totalMs))

    // ------------------------------------------------------------------

    private val currentPtr = FloatArray(D)

    private fun point(u: Float, v: Float, positive: Boolean) {
        coords.rewind()
        coords.putFloat(u.coerceIn(0f, 1f)).putFloat(v.coerceIn(0f, 1f)).putFloat(0f).putFloat(0f)
        coords.rewind()
        labels.rewind()
        labels.putFloat(if (positive) 1f else -1f).putFloat(-1f)
        labels.rewind()
    }

    /**
     * Eingabe-Einbettung der App, falls [prompt] gesetzt: der Punkt (Label 1 beim Tippen, -1 =
     * kein Punkt im Folgebild), so viele Leerpunkte, wie der Decoder Punkte erwartet, und der
     * angehängte Leerpunkt; wie in Metas Videoverfolgung.
     */
    private fun writePrompt(u: Float, v: Float, label: Int) {
        val p = prompt ?: return
        promptU.fill(0f)
        promptV.fill(0f)
        promptLabels.fill(-1)
        promptU[0] = u
        promptV[0] = v
        promptLabels[0] = label
        p.write(promptU, promptV, promptLabels, sparseScratch, sparseBuf)
    }

    /** Decoder; danach wie im Original: Ist kein Objekt da, Maske leer und Zeiger neutral. */
    private fun decode(embeddings: ByteBuffer) {
        val sparse = if (prompt != null) sparseBuf else requireNotNull(encoder.sparse)
        objectScore = decoder.run(embeddings, encoder.hr1, encoder.hr2, sparse, maskBuf, currentPtr)
        decMs = decoder.lastMs
        present = objectScore > 0f
        if (!present) {
            System.arraycopy(noObjPtr, 0, currentPtr, 0, D)
            maskBuf.rewind()
            maskBuf.asFloatBuffer().put(hidden)
            maskBuf.rewind()
        }
    }

    private fun encodeMemory(embeddings: ByteBuffer, startFrame: Boolean): FloatArray {
        val out = FloatArray(TOK_MEM * MEM_DIM)
        memory.write(embeddings, maskBuf, startFrame, out)
        memMs = memory.lastMs
        maskBuf.rewind()
        return out
    }

    /**
     * Füllt die festen Eingänge der Memory-Attention. Platz 0 ist das Startbild, Platz 7 - n
     * das Bild vor n Schritten. Leere Plätze und Zeiger bekommen -10000 und zählen nicht.
     */
    private fun fillMemory() {
        val mem = condMem ?: return
        val bankF = bank.asFloatBuffer()
        val biasF = bias.asFloatBuffer()
        bankF.put(mem)
        biasF.put(zeroBias)
        val n = recentMem.size
        for (slot in 1 until N_MEM) {
            val back = N_MEM - slot // so viele Schritte zurück
            val index = n - back
            bankF.position(slot * TOK_MEM * MEM_DIM)
            biasF.position(slot * TOK_MEM)
            if (index >= 0) {
                bankF.put(recentMem[index])
                biasF.put(zeroBias)
            } else {
                biasF.put(maskedBias)
            }
        }
        val ptrF = ptrs.asFloatBuffer()
        ptrF.put(condPtr)
        val np = recentPtr.size
        biasF.position(N_MEM * TOK_MEM)
        for (k in 0 until TOK_PTR) biasF.put(0f)
        for (j in 1 until N_PTR) {
            val index = np - j // neuester zuerst
            ptrF.position(j * D)
            val valid = index >= 0
            if (valid) ptrF.put(recentPtr[index])
            for (k in 0 until TOK_PTR) biasF.put(if (valid) 0f else MASK)
        }
        bank.rewind()
        ptrs.rewind()
        bias.rewind()
    }

    companion object {
        const val N_MEM = 7
        const val TOK_MEM = 512
        const val N_PTR = 16
        const val TOK_PTR = 4
        const val MEM_DIM = 64
        const val D = 256
        const val N_K = N_MEM * TOK_MEM + N_PTR * TOK_PTR
        const val MASK = -10000f
    }
}

/**
 * Schnellverfolgung: In jedem Bild nur Encoder und Decoder, ohne Speicher. Das Objekt findet
 * sie über einen Punkt in der Mitte seiner Maske im Bild davor (der Maskenpunkt, der dem
 * Schwerpunkt am nächsten liegt), so wie ein neuer Tipp. Ein Rahmen um die Maske taugt dafür
 * nicht: Er wächst von Bild zu Bild, bis er den Hintergrund mitnimmt. Mit dem Decoder wie im
 * Original kam sie auf der CPU in einem echten Video auf 97 % Übereinstimmung mit Metas
 * Verfolgung, mit Qualcomms Decoder nur auf 91 %. Schnell, aber ohne Gedächtnis: Bei Verdeckung
 * oder zwei ähnlichen Motiven kann sie springen.
 */
internal class FastTracker(
    private val encoder: ImageEncoder,
    private val decoder: MaskDecoder,
    private val prompt: PromptEncoder,
    private val coords: ByteBuffer,
    private val labels: ByteBuffer,
) : Tracker {
    private val maskBuf = directBuffer(MASK_SIZE * MASK_SIZE * 4)
    private val sparseBuf = decoder.newSparse()
    private val sparseScratch = FloatArray((decoder.points + 1) * PromptEncoder.D)
    private val promptU = FloatArray(decoder.points)
    private val promptV = FloatArray(decoder.points)
    private val promptLabels = IntArray(decoder.points)
    private val ptrScratch = FloatArray(PromptEncoder.D)
    private val maskFloats = FloatArray(MASK_SIZE * MASK_SIZE)
    private val hidden = FloatArray(maskFloats.size) { -1024f }

    /** Punkt (u, v) im quadratischen Bild, 0 bis 1: Mitte der Maske im Bild davor. */
    private val point = FloatArray(2)

    override val name: String = "schnell · Encoder ${encoder.name} · Decoder ${decoder.name} · Punkt aus dem Bild davor"
    override var started = false; private set
    override val canChoose: Boolean get() = decoder.candidates > 1
    override var present = false; private set
    override var objectScore = 0f; private set
    override var encMs = 0.0; private set
    override val attnMs = 0.0
    override var decMs = 0.0; private set
    override val memMs = 0.0
    override var convertMs = 0.0; private set
    override var totalMs = 0.0; private set

    override fun start(image: ByteBuffer, u: Float, v: Float, chooser: StartChooser?) {
        val t = System.nanoTime()
        encode(image)
        writePrompt(u, v)
        decode()
        val n = decoder.candidates
        if (chooser != null && present && n > 1) {
            val masks = FloatArray(n * MASK_SIZE * MASK_SIZE)
            val ious = FloatArray(n)
            decoder.candidateMasks(masks, ious)
            val k = chooser.choose(masks, ious, n)
            if (k in 0 until n && k != decoder.lastChoice) decoder.select(k, maskBuf, ptrScratch)
        }
        started = true
        // Fehlt die Maske, sucht der nächste Schritt wieder am Tipp-Punkt.
        if (!updatePoint()) {
            point[0] = u.coerceIn(0f, 1f)
            point[1] = v.coerceIn(0f, 1f)
        }
        totalMs = (System.nanoTime() - t) / 1e6
    }

    override fun step(image: ByteBuffer) {
        check(started) { "Erst ein Objekt antippen" }
        val t = System.nanoTime()
        encode(image)
        writePrompt(point[0], point[1])
        decode()
        updatePoint() // ohne Objekt bleibt der alte Punkt
        totalMs = (System.nanoTime() - t) / 1e6
    }

    override fun mask(): FloatArray {
        maskBuf.rewind()
        maskBuf.asFloatBuffer().get(maskFloats)
        return maskFloats
    }

    override fun reset() {
        started = false
        present = false
    }

    private fun encode(image: ByteBuffer) {
        coords.rewind()
        repeat(4) { coords.putFloat(0f) }
        coords.rewind()
        labels.rewind()
        labels.putFloat(-1f).putFloat(-1f)
        labels.rewind()
        encoder.run(image, coords, labels)
        encMs = encoder.lastMs
        convertMs = encoder.convertMs
    }

    /** Ein positiver Punkt, dann Leerpunkte bis zur Zahl, die der Decoder erwartet (der letzte Leerpunkt kommt dazu). */
    private fun writePrompt(u: Float, v: Float) {
        promptU.fill(0f)
        promptV.fill(0f)
        promptLabels.fill(-1)
        promptU[0] = u
        promptV[0] = v
        promptLabels[0] = 1
        prompt.write(promptU, promptV, promptLabels, sparseScratch, sparseBuf)
    }

    private fun decode() {
        objectScore = decoder.run(encoder.pixFeat, encoder.hr1, encoder.hr2, sparseBuf, maskBuf, ptrScratch)
        decMs = decoder.lastMs
        present = objectScore > 0f
        if (!present) {
            maskBuf.rewind()
            maskBuf.asFloatBuffer().put(hidden)
            maskBuf.rewind()
        }
    }

    /** Neuer Punkt: der Maskenpunkt nächst dem Schwerpunkt der Maske; false, wenn die Maske leer ist. */
    private fun updatePoint(): Boolean {
        if (!present) return false
        val found = maskCenter(mask(), MASK_SIZE) ?: return false
        point[0] = found[0]
        point[1] = found[1]
        return true
    }
}

/**
 * Mitte einer Maske (n × n Logits, größer 0 heißt Objekt) als (u, v) von 0 bis 1: der
 * Maskenpunkt, der dem Schwerpunkt am nächsten liegt, damit er auch bei einer gebogenen Form
 * auf dem Objekt liegt. Bei gleichem Abstand zählt der erste Punkt zeilenweise. Null, wenn die
 * Maske leer ist. (Gleiche Rechnung wie mask_center im Export.)
 */
internal fun maskCenter(m: FloatArray, n: Int): FloatArray? {
    var count = 0L
    var sx = 0L
    var sy = 0L
    for (y in 0 until n) {
        val row = y * n
        for (x in 0 until n) {
            if (m[row + x] > 0f) {
                count++
                sx += x
                sy += y
            }
        }
    }
    if (count == 0L) return null
    val cx = sx.toDouble() / count
    val cy = sy.toDouble() / count
    var best = Double.MAX_VALUE
    var bx = 0
    var by = 0
    for (y in 0 until n) {
        val row = y * n
        for (x in 0 until n) {
            if (m[row + x] > 0f) {
                val d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                if (d < best) {
                    best = d
                    bx = x
                    by = y
                }
            }
        }
    }
    return floatArrayOf((bx + 0.5f) / n, (by + 0.5f) / n)
}
