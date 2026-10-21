package de.botrik.cinebo.test

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.os.SystemClock
import android.util.Base64
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Kantenlänge der Masken, die EdgeTAM ausgibt (Zeile 0 = oberste Bildzeile). */
internal const val MASK_SIZE = 256

/** Ein Kamerabild, fertig vorbereitet für beide Modelle (von der Grafik-Pipeline). */
internal class FrameSlot {
    /** EdgeTAM: ganzes Bild, aufrecht, auf 1024 × 1024 gestreckt, RGB 0 bis 1. */
    val enc: ByteBuffer = directBuffer(ENC * ENC * 3 * 4)

    /** Depth Anything: ganzes Bild, aufrecht, in 518 × 518 eingepasst, Rand schwarz. */
    val depth: ByteBuffer = directBuffer(DEPTH * DEPTH * 3 * 4)

    /** Wo das Bild im 518er-Quadrat liegt (0 bis 1). */
    val depthContent = RectF()
    var timestampNs = 0L
    var prepMs = 0.0
    var width = 0
    var height = 0
    /** Drehung vom st-Raum der Pipeline zum aufrechten Bild (beim Hochkant-Halten 0). */
    var rotation = 0

    fun copyFrom(other: FrameSlot) {
        other.enc.rewind()
        enc.rewind()
        enc.put(other.enc)
        enc.rewind()
        other.enc.rewind()
        other.depth.rewind()
        depth.rewind()
        depth.put(other.depth)
        depth.rewind()
        other.depth.rewind()
        depthContent.set(other.depthContent)
        timestampNs = other.timestampNs
        prepMs = other.prepMs
        width = other.width
        height = other.height
        rotation = other.rotation
    }

    /**
     * Kleines JPEG (Base64) dessen, was EdgeTAM bekommt, zurückgerechnet aus [enc]: zeigt im
     * Bericht, ob das KI-Bild aufrecht steht und die Farben stimmen. Zeile 0 ist oben.
     */
    fun thumbnailJpegBase64(size: Int = THUMBNAIL): String {
        val fb = enc.duplicate().order(ByteOrder.nativeOrder()).asFloatBuffer()
        val pixels = IntArray(size * size)
        val step = ENC.toFloat() / size
        for (y in 0 until size) {
            for (x in 0 until size) {
                // Vier Abgriffe je Bildpunkt, damit beim Verkleinern keine Muster entstehen.
                var r = 0f
                var g = 0f
                var b = 0f
                for (j in 0..1) {
                    val sy = ((y + 0.25f + 0.5f * j) * step).toInt().coerceIn(0, ENC - 1)
                    for (i in 0..1) {
                        val sx = ((x + 0.25f + 0.5f * i) * step).toInt().coerceIn(0, ENC - 1)
                        val k = (sy * ENC + sx) * 3
                        r += fb.get(k)
                        g += fb.get(k + 1)
                        b += fb.get(k + 2)
                    }
                }
                pixels[y * size + x] = Color.rgb(byteOf(r / 4f), byteOf(g / 4f), byteOf(b / 4f))
            }
        }
        val bitmap = Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
        val jpeg = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 75, jpeg)
        bitmap.recycle()
        return Base64.encodeToString(jpeg.toByteArray(), Base64.NO_WRAP)
    }

    companion object {
        const val ENC = 1024
        const val DEPTH = 518
        /** Kantenlänge des Vorschaubilds im Bericht: reicht, um Ausrichtung und Motiv zu erkennen. */
        const val THUMBNAIL = 128

        private fun byteOf(v: Float): Int = if (v.isNaN()) 0 else (v * 255f + 0.5f).toInt().coerceIn(0, 255)
    }
}

/**
 * Zwei Plätze für Kamerabilder zwischen Grafik-Pipeline und KI-Thread. Wartet schon ein
 * fertiges Bild, bereitet die Pipeline kein neues vor, statt es umsonst zu rechnen.
 */
internal class FramePool(size: Int = 2) {
    private val lock = Object()
    private val free = ArrayDeque<FrameSlot>().apply { repeat(size) { addLast(FrameSlot()) } }
    private var ready: FrameSlot? = null

    @Volatile var prepared = 0L
        private set
    @Volatile var skipped = 0L
        private set

    /** Für die Grafik-Pipeline: ein freier Platz, oder null, wenn schon ein Bild wartet. */
    fun acquire(): FrameSlot? = synchronized(lock) {
        if (ready != null) {
            skipped++
            null
        } else {
            free.removeFirstOrNull().also { if (it == null) skipped++ }
        }
    }

    fun publish(slot: FrameSlot) = synchronized(lock) {
        ready?.let { free.addLast(it) }
        ready = slot
        prepared++
        lock.notifyAll()
    }

    /** Platz zurückgeben, ohne ihn zu veröffentlichen (etwa nach einem Fehler). */
    fun giveBack(slot: FrameSlot) = synchronized(lock) { free.addLast(slot) }

    /** Für den KI-Thread: wartet höchstens [timeoutMs] auf ein fertiges Bild. */
    fun take(timeoutMs: Long): FrameSlot? = synchronized(lock) {
        val end = SystemClock.uptimeMillis() + timeoutMs
        while (ready == null) {
            val left = end - SystemClock.uptimeMillis()
            if (left <= 0) return null
            lock.wait(left)
        }
        val slot = ready
        ready = null
        slot
    }

    fun release(slot: FrameSlot) = synchronized(lock) { free.addLast(slot) }

    /** Ein wartendes, inzwischen altes Bild verwerfen. */
    fun drain() = synchronized(lock) {
        ready?.let { free.addLast(it) }
        ready = null
    }

    fun resetCounters() {
        prepared = 0
        skipped = 0
    }
}
