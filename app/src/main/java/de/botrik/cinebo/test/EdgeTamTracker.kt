package de.botrik.cinebo.test

import org.json.JSONObject
import java.nio.ByteBuffer

/**
 * Verfolgt ein angetipptes Objekt mit EdgeTAM, Bild für Bild.
 *
 * Encoder und Decoder stammen von Qualcomm, Memory-Attention und Memory-Encoder sind
 * unser eigener Export (tools/export/edgetam_export.py). Der Speicher hält wie im
 * Original das Startbild und die sechs letzten Bilder, dazu bis zu 16 Objekt-Zeiger.
 * Nur vom KI-Thread benutzen.
 */
internal class EdgeTamTracker(
    private val enc: NpuModel,
    private val dec: NpuModel,
    private val attn: NpuModel,
    private val memEnc: NpuModel,
    private val noObjPtr: FloatArray,
) {
    private val inImage = enc.inputIndex("image")
    private val inCoords = enc.inputIndex("unnorm_coords")
    private val inLabels = enc.inputIndex("labels")
    private val outEmb = enc.outputIndex("image_embeddings")
    private val outHr1 = enc.outputIndex("high_res_features1")
    private val outHr2 = enc.outputIndex("high_res_features2")
    private val outSparse = enc.outputIndex("sparse_embedding")
    private val outPix = enc.outputIndex("pix_feat")
    private val decEmb = dec.inputIndex("image_embeddings")
    private val decHr1 = dec.inputIndex("high_res_features1")
    private val decHr2 = dec.inputIndex("high_res_features2")
    private val decSparse = dec.inputIndex("sparse_embedding")
    private val decMasks = dec.outputIndex("masks")
    private val decPtr = dec.outputIndex("obj_ptr")
    private val decLogits = dec.outputIndex("object_score_logits")

    private val coords = enc.newInput(inCoords)
    private val labels = enc.newInput(inLabels)
    private val maskBuf = dec.newOutput(decMasks)
    private val ptrBuf = dec.newOutput(decPtr)
    private val logitBuf = dec.newOutput(decLogits)
    private val bank = attn.newInput(1)
    private val ptrs = attn.newInput(2)
    private val bias = attn.newInput(3)
    private val cond = memEnc.newInput(2)

    private val maskFloats = FloatArray(MaskPainter.SIZE * MaskPainter.SIZE)
    private val hidden = FloatArray(maskFloats.size) { -1024f }
    private val zeroBias = FloatArray(TOK_MEM) { 0f }
    private val maskedBias = FloatArray(TOK_MEM) { MASK }

    private var condMem: FloatArray? = null
    private val condPtr = FloatArray(D)
    private val recentMem = ArrayDeque<FloatArray>()
    private val recentPtr = ArrayDeque<FloatArray>()

    init {
        require(enc.inputBytes(inImage) == FrameSlot.ENC * FrameSlot.ENC * 3 * 4) { "Encoder erwartet ein anderes Bild" }
        require(bank.capacity() == N_MEM * TOK_MEM * MEM_DIM * 4) { "Memory-Attention hat eine andere Speichergröße" }
        require(ptrs.capacity() == N_PTR * D * 4 && bias.capacity() == N_K * 4)
        require(memEnc.outputFloats(0) == TOK_MEM * MEM_DIM) { "Memory-Encoder liefert eine andere Größe" }
        require(noObjPtr.size == D)
    }

    /** Zeiten des letzten Bildes in ms (im Modell gerechnet) und insgesamt. */
    var encMs = 0.0; private set
    var attnMs = 0.0; private set
    var decMs = 0.0; private set
    var memMs = 0.0; private set
    var totalMs = 0.0; private set

    /** Hat EdgeTAM das Objekt im letzten Bild gefunden? */
    var present = false; private set
    var objectScore = 0f; private set
    var steps = 0L; private set

    val started: Boolean get() = condMem != null

    /** Startbild: Objekt an Stelle (u, v), beides 0 bis 1 im aufrechten Kamerabild. */
    fun start(image: ByteBuffer, u: Float, v: Float) {
        val t = System.nanoTime()
        point(u, v, positive = true)
        enc.run(inputs(enc, inImage to image, inCoords to coords, inLabels to labels))
        encMs = enc.lastNativeMs
        // Beim Startbild geht pix_feat (mit no_mem_embed) direkt in den Decoder.
        decode(enc.output(outPix))
        attnMs = 0.0
        val mem = encodeMemory(enc.output(outEmb), startFrame = true)
        condMem = mem
        System.arraycopy(currentPtr, 0, condPtr, 0, D)
        recentMem.clear()
        recentPtr.clear()
        steps = 1
        totalMs = (System.nanoTime() - t) / 1e6
    }

    /** Folgebild: findet das Objekt mit Hilfe des Speichers wieder. */
    fun step(image: ByteBuffer) {
        check(started) { "Erst ein Objekt antippen" }
        val t = System.nanoTime()
        point(0f, 0f, positive = false)
        enc.run(inputs(enc, inImage to image, inCoords to coords, inLabels to labels))
        encMs = enc.lastNativeMs
        val emb = enc.output(outEmb)
        fillMemory()
        attn.run(arrayOf(emb, bank, ptrs, bias))
        attnMs = attn.lastNativeMs
        decode(attn.output(0))
        val mem = encodeMemory(emb, startFrame = false)
        recentMem.addLast(mem)
        while (recentMem.size > N_MEM - 1) recentMem.removeFirst()
        recentPtr.addLast(currentPtr.copyOf())
        while (recentPtr.size > N_PTR - 1) recentPtr.removeFirst()
        steps++
        totalMs = (System.nanoTime() - t) / 1e6
    }

    /** Maske des letzten Bildes (256 × 256 Logits, größer 0 heißt Objekt). */
    fun mask(): FloatArray {
        maskBuf.rewind()
        maskBuf.asFloatBuffer().get(maskFloats)
        return maskFloats
    }

    /** Speicher (für die Genauigkeitsprüfung): Startbild-Eintrag und Zeiger. */
    fun memorySnapshot(): Pair<FloatArray, FloatArray>? = condMem?.let { it.copyOf() to condPtr.copyOf() }

    fun reset() {
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

    /** Decoder; danach wie in Qualcomms App: Ist kein Objekt da, Maske leer und Zeiger neutral. */
    private fun decode(embeddings: ByteBuffer) {
        dec.run(
            inputs(
                dec,
                decEmb to embeddings,
                decHr1 to enc.output(outHr1),
                decHr2 to enc.output(outHr2),
                decSparse to enc.output(outSparse)
            ),
            mapOf(decMasks to maskBuf, decPtr to ptrBuf, decLogits to logitBuf)
        )
        decMs = dec.lastNativeMs
        objectScore = logitBuf.getFloat(0)
        present = objectScore > 0f
        if (present) {
            ptrBuf.asFloatBuffer().get(currentPtr)
        } else {
            System.arraycopy(noObjPtr, 0, currentPtr, 0, D)
            maskBuf.rewind()
            maskBuf.asFloatBuffer().put(hidden)
            maskBuf.rewind()
        }
    }

    private fun encodeMemory(embeddings: ByteBuffer, startFrame: Boolean): FloatArray {
        cond.rewind()
        cond.putFloat(if (startFrame) 1f else 0f)
        cond.rewind()
        memEnc.run(arrayOf(embeddings, maskBuf, cond))
        memMs = memEnc.lastNativeMs
        val out = FloatArray(TOK_MEM * MEM_DIM)
        memEnc.output(0).asFloatBuffer().get(out)
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

    private fun inputs(model: NpuModel, vararg pairs: Pair<Int, Any>): Array<Any> {
        val array = arrayOfNulls<Any>(model.inputCount)
        for ((i, v) in pairs) array[i] = v
        return Array(array.size) { requireNotNull(array[it]) { "${model.asset}: Eingang $it fehlt" } }
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
