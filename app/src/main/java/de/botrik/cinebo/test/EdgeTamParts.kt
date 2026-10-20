package de.botrik.cinebo.test

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Konstanten von EdgeTAM, die die App selbst braucht (edgetam_konstanten.json aus unserem
 * Export, gegen Metas Original geprüft).
 */
internal class EdgeTamConstants(json: JSONObject) {
    val version: Int = json.optInt("version", 1)
    val noObjPtr: FloatArray = floats(json.getJSONArray("noObjPtr"))

    /** Nur ab Version 2: alles, um die Eingabe-Einbettung in der App zu rechnen. */
    val prompt: PromptEncoder? = json.optJSONObject("prompt")?.let { PromptEncoder.from(it) }

    /** Proben aus dem Original: Eingaben und erwartete Einbettung, zum Prüfen von [prompt]. */
    val promptSamples: JSONArray? = json.optJSONObject("prompt")?.optJSONArray("proben")

    /** Prüfvideo: Datei, Startpunkt und Ergebnisse des Originals (nur ab Version 2). */
    val testVideo: JSONObject? = json.optJSONObject("pruefvideo")

    companion object {
        fun load(folder: File): EdgeTamConstants =
            EdgeTamConstants(JSONObject(File(folder, "edgetam_konstanten.json").readText()))

        fun floats(a: JSONArray): FloatArray = FloatArray(a.length()) { a.getDouble(it).toFloat() }
    }
}

/**
 * Rechnet die Eingabe-Einbettung (sparse_embedding) des Decoders so wie SAM2s Prompt-Encoder.
 * Qualcomms Encoder kann das nur für positive Punkte und, quantisiert, nur bis zur Bildmitte;
 * hier gehen alle Arten: positiv (1), negativ (0), Rahmenecken (2, 3) und „kein Punkt“ (-1).
 * Punkte sind (u, v) von 0 bis 1 im quadratischen Eingangsbild des Encoders.
 */
internal class PromptEncoder(
    private val gauss: Array<FloatArray>,
    private val labelEmbeddings: Array<FloatArray>,
    private val noPoint: FloatArray,
    private val imageSize: Int,
) {
    init {
        require(gauss.size == 2 && gauss.all { it.size == HALF }) { "Gauss-Matrix muss 2 × $HALF sein" }
        require(labelEmbeddings.size == 4 && labelEmbeddings.all { it.size == D }) { "4 Punkt-Einbettungen zu je $D" }
        require(noPoint.size == D)
    }

    /**
     * Schreibt die Einbettung von [labels].size Punkten und einem angehängten Leerpunkt nach [out]
     * ((n + 1) · 256 Werte), genau wie SAM2 mit pad=True.
     */
    fun embed(u: FloatArray, v: FloatArray, labels: IntArray, out: FloatArray) {
        val n = labels.size
        require(out.size == (n + 1) * D) { "Ausgabe muss ${(n + 1) * D} Werte haben" }
        for (i in 0..n) {
            val base = i * D
            val label = if (i < n) labels[i] else -1
            if (label == -1) {
                System.arraycopy(noPoint, 0, out, base, D)
                continue
            }
            require(label in 0..3) { "Unbekanntes Label $label" }
            // SAM2: (Pixel + 0,5) / Bildgröße, dann 2·x − 1.
            val x = 2f * u[i] - 1f + 1f / imageSize
            val y = 2f * v[i] - 1f + 1f / imageSize
            val add = labelEmbeddings[label]
            for (j in 0 until HALF) {
                val a = 2.0 * PI * (x * gauss[0][j] + y * gauss[1][j]).toDouble()
                out[base + j] = sin(a).toFloat() + add[j]
                out[base + HALF + j] = cos(a).toFloat() + add[HALF + j]
            }
        }
    }

    /** Wie [embed], schreibt aber gleich in den Eingang des Decoders. */
    fun write(u: FloatArray, v: FloatArray, labels: IntArray, scratch: FloatArray, target: ByteBuffer) {
        embed(u, v, labels, scratch)
        target.rewind()
        target.asFloatBuffer().put(scratch)
        target.rewind()
    }

    /** Vergleicht mit den Proben aus dem Original; größte Abweichung je Wert. */
    fun check(samples: JSONArray): JSONObject {
        var worst = 0.0
        var count = 0
        for (s in 0 until samples.length()) {
            val p = samples.getJSONObject(s)
            val u = EdgeTamConstants.floats(p.getJSONArray("u"))
            val v = EdgeTamConstants.floats(p.getJSONArray("v"))
            val labels = p.getJSONArray("labels").let { a -> IntArray(a.length()) { a.getInt(it) } }
            val expected = p.getJSONArray("einbettung")
            val out = FloatArray((labels.size + 1) * D)
            embed(u, v, labels, out)
            for (t in 0 until expected.length()) {
                val row = expected.getJSONArray(t)
                for (j in 0 until row.length()) {
                    worst = maxOf(worst, kotlin.math.abs(out[t * D + j] - row.getDouble(j)))
                }
            }
            count++
        }
        return JSONObject().put("proben", count).put("groessteAbweichung", worst)
    }

    companion object {
        const val D = 256
        const val HALF = 128

        fun from(json: JSONObject): PromptEncoder {
            val g = json.getJSONArray("gauss")
            val p = json.getJSONArray("punkte")
            return PromptEncoder(
                Array(2) { EdgeTamConstants.floats(g.getJSONArray(it)) },
                Array(4) { EdgeTamConstants.floats(p.getJSONArray(it)) },
                EdgeTamConstants.floats(json.getJSONArray("keinPunkt")),
                json.getInt("bildgroesse"),
            )
        }
    }
}

/** Bildmerkmale für den Decoder, gleich ob vom Encoder in fp16 oder quantisiert. */
internal interface ImageEncoder {
    val name: String

    /** Merkmale ohne no_mem_embed (für Memory-Attention und Memory-Encoder). */
    val embeddings: ByteBuffer

    /** Merkmale mit no_mem_embed (für den Decoder beim Startbild und in der Schnellverfolgung). */
    val pixFeat: ByteBuffer
    val hr1: ByteBuffer
    val hr2: ByteBuffer

    /** Die Eingabe-Einbettung des Encoders selbst, oder null, wenn er sie nicht verlässlich liefert. */
    val sparse: ByteBuffer?

    /** Im Modell gerechnet und, falls quantisiert, zusätzlich für das Umrechnen gebraucht. */
    val lastMs: Double
    val convertMs: Double

    /** [image]: 1024 × 1024 × 3, RGB 0 bis 1 (float). Punkte wie bei [EdgeTamTracker]. */
    fun run(image: ByteBuffer, coords: ByteBuffer, labels: ByteBuffer)
}

/** Qualcomms Encoder in fp16; die Ausgänge sind Sichten in die Ausgänge des Modells. */
internal class FloatEncoder(private val enc: NpuModel) : ImageEncoder {
    override val name = "fp16"
    private val inImage = enc.inputIndex("image")
    private val inCoords = enc.inputIndex("unnorm_coords")
    private val inLabels = enc.inputIndex("labels")
    private val outEmb = enc.outputIndex("image_embeddings")
    private val outPix = enc.outputIndex("pix_feat")
    private val outHr1 = enc.outputIndex("high_res_features1")
    private val outHr2 = enc.outputIndex("high_res_features2")
    private val outSparse = enc.outputIndex("sparse_embedding")
    override val embeddings: ByteBuffer get() = enc.output(outEmb)
    override val pixFeat: ByteBuffer get() = enc.output(outPix)
    override val hr1: ByteBuffer get() = enc.output(outHr1)
    override val hr2: ByteBuffer get() = enc.output(outHr2)
    override val sparse: ByteBuffer get() = enc.output(outSparse)
    override var lastMs = 0.0
        private set
    override val convertMs = 0.0

    override fun run(image: ByteBuffer, coords: ByteBuffer, labels: ByteBuffer) {
        val inputs = arrayOfNulls<Any>(3)
        inputs[inImage] = image
        inputs[inCoords] = coords
        inputs[inLabels] = labels
        @Suppress("UNCHECKED_CAST")
        enc.run(inputs as Array<Any>)
        lastMs = enc.lastNativeMs
    }
}

/**
 * Qualcomms quantisierter Encoder (w8a8): Bild nach UINT8, Ausgänge zurück nach float. Die
 * Punkte nimmt er nicht ernst (siehe [PromptEncoder]); die Einbettung kommt dann aus der App.
 */
internal class QuantEncoder(private val enc: NpuModel) : ImageEncoder {
    override val name = "w8a8"
    private val inImage = enc.inputIndex("image")
    private val inCoords = enc.inputIndex("unnorm_coords")
    private val inLabels = enc.inputIndex("labels")
    private val outs = listOf("image_embeddings", "pix_feat", "high_res_features1", "high_res_features2").map { enc.outputIndex(it) }
    private val image8 = enc.newInput(inImage)
    private val coords8 = enc.newInput(inCoords)
    private val labels8 = enc.newInput(inLabels)
    private val imageQuant = enc.inputQuant(inImage)
    private val floatsOut = outs.map { directBuffer(enc.outputFloats(it) * 4) }
    private val scratchIn = FloatArray(1024 * 3)
    private val bytesIn = ByteArray(1024 * 3)
    override val embeddings: ByteBuffer get() = floatsOut[0].duplicate().order(floatsOut[0].order())
    override val pixFeat: ByteBuffer get() = floatsOut[1].duplicate().order(floatsOut[1].order())
    override val hr1: ByteBuffer get() = floatsOut[2].duplicate().order(floatsOut[2].order())
    override val hr2: ByteBuffer get() = floatsOut[3].duplicate().order(floatsOut[3].order())
    override val sparse: ByteBuffer? = null
    override var lastMs = 0.0
        private set
    override var convertMs = 0.0
        private set

    init {
        require(enc.isQuantizedInput(inImage)) { "${enc.asset}: Bild ist nicht quantisiert" }
        // Punkte: Folgebilder ohne Punkt (Label -1). Mehr kann dieser Encoder nicht sicher.
        val (cs, cz) = enc.inputQuant(inCoords)
        val (ls, lz) = enc.inputQuant(inLabels)
        coords8.rewind()
        repeat(coords8.capacity()) { coords8.put(quantize(0f, cs, cz)) }
        labels8.rewind()
        repeat(labels8.capacity()) { labels8.put(quantize(-1f, ls, lz)) }
        coords8.rewind()
        labels8.rewind()
    }

    override fun run(image: ByteBuffer, coords: ByteBuffer, labels: ByteBuffer) {
        val t = System.nanoTime()
        val (scale, zero) = imageQuant
        val inv = 1f / scale
        val src = image.duplicate().order(image.order()).apply { rewind() }.asFloatBuffer()
        image8.rewind()
        while (src.hasRemaining()) {
            val n = minOf(scratchIn.size, src.remaining())
            src.get(scratchIn, 0, n)
            for (i in 0 until n) bytesIn[i] = (scratchIn[i] * inv + zero).roundToInt().coerceIn(0, 255).toByte()
            image8.put(bytesIn, 0, n)
        }
        image8.rewind()
        val prep = System.nanoTime()
        val inputs = arrayOfNulls<Any>(3)
        inputs[inImage] = image8
        inputs[inCoords] = coords8
        inputs[inLabels] = labels8
        @Suppress("UNCHECKED_CAST")
        enc.run(inputs as Array<Any>)
        lastMs = enc.lastNativeMs
        val after = System.nanoTime()
        for ((k, i) in outs.withIndex()) dequantize(enc.output(i), enc.outputQuant(i), floatsOut[k])
        convertMs = ((prep - t) + (System.nanoTime() - after)) / 1e6
    }

    private val chunk = ByteArray(4096)
    private val chunkF = FloatArray(4096)

    private fun dequantize(src: ByteBuffer, q: Pair<Float, Int>, dst: ByteBuffer) {
        val (scale, zero) = q
        val s = src.duplicate().apply { rewind() }
        val d = dst.duplicate().order(dst.order()).apply { rewind() }.asFloatBuffer()
        while (s.hasRemaining()) {
            val n = minOf(chunk.size, s.remaining())
            s.get(chunk, 0, n)
            for (i in 0 until n) chunkF[i] = ((chunk[i].toInt() and 0xFF) - zero) * scale
            d.put(chunkF, 0, n)
        }
    }

    private fun quantize(x: Float, scale: Float, zero: Int): Byte = (x / scale + zero).roundToInt().coerceIn(0, 255).toByte()
}

/** Decoder der Verfolgung: liefert die Maske (256 × 256 Logits), den Objekt-Zeiger und den Objekt-Wert. */
internal interface MaskDecoder {
    val name: String

    /** Punkte in der Eingabe-Einbettung, ohne den angehängten Leerpunkt. */
    val points: Int

    /** Im Modell gerechnet (ms), beim letzten Lauf. */
    val lastMs: Double

    /** Neuer Eingabepuffer für die Einbettung: (points + 1) · 256 float. */
    fun newSparse(): ByteBuffer

    /**
     * Rechnet und schreibt die Maske nach [mask] (direkter Puffer, 256 · 256 float, danach
     * zurückgespult) und den Objekt-Zeiger nach [ptr]. Gibt den Objekt-Wert zurück; größer 0
     * heißt, das Objekt ist im Bild.
     */
    fun run(embeddings: ByteBuffer, hr1: ByteBuffer, hr2: ByteBuffer, sparse: ByteBuffer, mask: ByteBuffer, ptr: FloatArray): Float
}

/**
 * Qualcomms Decoder: eine Maske (ist sie unsicher, die beste von dreien) mit dem Zeiger des
 * ersten Tokens; zwei Punkte plus Leerpunkt. Er weicht vom Original ab (in der Hochskalierung
 * fehlt eine LayerNorm), siehe [MetaDecoder].
 */
internal class QualcommDecoder(private val dec: NpuModel) : MaskDecoder {
    override val name = "qualcomm"
    override val points = 2
    private val inEmb = dec.inputIndex("image_embeddings")
    private val inHr1 = dec.inputIndex("high_res_features1")
    private val inHr2 = dec.inputIndex("high_res_features2")
    private val inSparse = dec.inputIndex("sparse_embedding")
    private val outMasks = dec.outputIndex("masks")
    private val outPtr = dec.outputIndex("obj_ptr")
    private val outLogits = dec.outputIndex("object_score_logits")
    private val ptrBuf = dec.newOutput(outPtr)
    private val logitBuf = dec.newOutput(outLogits)
    override var lastMs = 0.0
        private set

    init {
        require(dec.inputBytes(inSparse) == (points + 1) * PromptEncoder.D * 4) {
            "Qualcomms Decoder erwartet ${dec.inputBytes(inSparse) / 4 / PromptEncoder.D} Einbettungen statt ${points + 1}"
        }
    }

    override fun newSparse(): ByteBuffer = dec.newInput(inSparse)

    override fun run(embeddings: ByteBuffer, hr1: ByteBuffer, hr2: ByteBuffer, sparse: ByteBuffer, mask: ByteBuffer, ptr: FloatArray): Float {
        val inputs = arrayOfNulls<Any>(4)
        inputs[inEmb] = embeddings
        inputs[inHr1] = hr1
        inputs[inHr2] = hr2
        inputs[inSparse] = sparse
        @Suppress("UNCHECKED_CAST")
        dec.run(inputs as Array<Any>, mapOf(outMasks to mask, outPtr to ptrBuf, outLogits to logitBuf))
        lastMs = dec.lastNativeMs
        ptrBuf.asFloatBuffer().get(ptr)
        return logitBuf.getFloat(0)
    }
}

/**
 * Unser Decoder (tools/export/edgetam_export.py), der rechnet wie Metas Videoverfolgung: drei
 * Masken, es zählt die mit der besten IoU-Schätzung samt dem Zeiger ihres Tokens; ein Punkt
 * (oder Leerpunkt) plus Leerpunkt. Eingänge in der Reihenfolge des Exports: Merkmale,
 * high_res_features1, high_res_features2, Einbettung; die Ausgänge erkennt er an der Form.
 */
internal class MetaDecoder(private val dec: NpuModel) : MaskDecoder {
    override val name = "wie Meta"
    override val points = 1
    private val outMasks = dec.outputIndexByShape(1, 3, MASK, MASK)
    private val outIous = dec.outputIndexByShape(1, 3)
    private val outPtrs = dec.outputIndexByShape(1, 3, PromptEncoder.D)
    private val outScore = dec.outputIndexByShape(1, 1)
    private val plane = FloatArray(MASK * MASK)
    private val ious = FloatArray(3)
    override var lastMs = 0.0
        private set

    /** Welche der drei Masken zuletzt zählte (0 bis 2). */
    var lastChoice = 0
        private set

    init {
        require(dec.inputCount == 4 && dec.inputBytes(3) == (points + 1) * PromptEncoder.D * 4) {
            "Decoder wie Meta: unerwartete Eingänge (${dec.inputCount})"
        }
    }

    override fun newSparse(): ByteBuffer = dec.newInput(3)

    override fun run(embeddings: ByteBuffer, hr1: ByteBuffer, hr2: ByteBuffer, sparse: ByteBuffer, mask: ByteBuffer, ptr: FloatArray): Float {
        dec.run(arrayOf(embeddings, hr1, hr2, sparse))
        lastMs = dec.lastNativeMs
        dec.output(outIous).asFloatBuffer().get(ious)
        var best = 0
        for (i in 1 until 3) if (ious[i] > ious[best]) best = i // wie argmax: bei Gleichstand die erste
        lastChoice = best
        dec.output(outMasks).asFloatBuffer().apply { position(best * plane.size) }.get(plane)
        mask.rewind()
        mask.asFloatBuffer().put(plane)
        mask.rewind()
        dec.output(outPtrs).asFloatBuffer().apply { position(best * PromptEncoder.D) }.get(ptr)
        return dec.output(outScore).getFloat(0)
    }

    companion object {
        const val MASK = 256
    }
}

/** Schreibt einen neuen Speicher-Eintrag (512 × 64) aus Bildmerkmalen und Maske. */
internal interface MemoryWriter {
    val name: String
    val lastMs: Double

    /** [embeddings] ohne no_mem_embed, [mask] 256 × 256 Logits; [startFrame]: binär statt Sigmoid. */
    fun write(embeddings: ByteBuffer, mask: ByteBuffer, startFrame: Boolean, out: FloatArray)
}

/** Unser eigener Export (Stufe 2): rechnet Maske und Merkmale selbst hoch. */
internal class OwnMemoryWriter(private val memEnc: NpuModel) : MemoryWriter {
    override val name = "eigener"
    private val cond = memEnc.newInput(2)
    override var lastMs = 0.0
        private set

    init {
        require(memEnc.outputFloats(0) == EdgeTamTracker.TOK_MEM * EdgeTamTracker.MEM_DIM) { "Memory-Encoder liefert eine andere Größe" }
    }

    override fun write(embeddings: ByteBuffer, mask: ByteBuffer, startFrame: Boolean, out: FloatArray) {
        cond.rewind()
        cond.putFloat(if (startFrame) 1f else 0f)
        cond.rewind()
        mask.rewind()
        memEnc.run(arrayOf(embeddings, mask, cond))
        lastMs = memEnc.lastNativeMs
        memEnc.output(0).asFloatBuffer().get(out)
    }
}

/**
 * Qualcomms Memory-Encoder mit unserem kleinen Vorbereitungsmodell davor: das bringt die
 * Merkmale in Qualcomms Reihenfolge (NCHW) und die Maske auf 1024 × 1024, binär oder als Sigmoid.
 */
internal class QualcommMemoryWriter(private val prep: NpuModel, private val memEnc: NpuModel) : MemoryWriter {
    override val name = "qualcomm"
    private val cond = prep.newInput(2)
    private val outPix = prep.outputIndexByShape(1, EdgeTamTracker.D, 64, 64)
    private val outMask = prep.outputIndexByShape(1, 1, 1024, 1024)
    private val inPix = memEnc.inputIndex("pix_feat")
    private val inMask = memEnc.inputIndex("mask_for_mem")
    override var lastMs = 0.0
        private set

    init {
        require(memEnc.outputFloats(0) == EdgeTamTracker.TOK_MEM * EdgeTamTracker.MEM_DIM) { "Memory-Encoder liefert eine andere Größe" }
    }

    override fun write(embeddings: ByteBuffer, mask: ByteBuffer, startFrame: Boolean, out: FloatArray) {
        cond.rewind()
        cond.putFloat(if (startFrame) 1f else 0f)
        cond.rewind()
        mask.rewind()
        prep.run(arrayOf(embeddings, mask, cond))
        val prepMs = prep.lastNativeMs
        val inputs = arrayOfNulls<Any>(2)
        inputs[inPix] = prep.output(outPix)
        inputs[inMask] = prep.output(outMask)
        @Suppress("UNCHECKED_CAST")
        memEnc.run(inputs as Array<Any>)
        lastMs = prepMs + memEnc.lastNativeMs
        memEnc.output(0).asFloatBuffer().get(out)
    }
}
