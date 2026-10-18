package de.botrik.cinebo.test

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min

/** Ein Kamerabild, fertig vorbereitet für beide Modelle. */
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

    companion object {
        const val ENC = 1024
        const val DEPTH = 518
    }
}

/**
 * Zwei Plätze für Kamerabilder zwischen Analyse-Thread und KI-Thread. Wartet schon ein
 * fertiges Bild, verwirft der Analyse-Thread neue Bilder, statt sie umsonst vorzubereiten.
 */
internal class FramePool(size: Int = 2) {
    private val lock = Object()
    private val free = ArrayDeque<FrameSlot>().apply { repeat(size) { addLast(FrameSlot()) } }
    private var ready: FrameSlot? = null

    @Volatile var prepared = 0L
        private set
    @Volatile var skipped = 0L
        private set

    /** Für den Analyse-Thread: ein freier Platz, oder null, wenn schon ein Bild wartet. */
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

/** Bereitet Kamerabilder (RGBA) für die Modelle vor. Nur vom Analyse-Thread benutzen. */
internal class FramePrep {
    private var source: Bitmap? = null
    private var rowCopy: ByteBuffer? = null
    private val encBitmap = Bitmap.createBitmap(FrameSlot.ENC, FrameSlot.ENC, Bitmap.Config.ARGB_8888)
    private val depthBitmap = Bitmap.createBitmap(FrameSlot.DEPTH, FrameSlot.DEPTH, Bitmap.Config.ARGB_8888)
    private val encCanvas = Canvas(encBitmap)
    private val depthCanvas = Canvas(depthBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val encPixels = IntArray(FrameSlot.ENC * FrameSlot.ENC)
    private val depthPixels = IntArray(FrameSlot.DEPTH * FrameSlot.DEPTH)
    private val encFloats = FloatArray(encPixels.size * 3)
    private val depthFloats = FloatArray(depthPixels.size * 3)
    private val matrix = Matrix()
    private val bounds = RectF()
    private val target = RectF()

    fun fill(image: ImageProxy, slot: FrameSlot) {
        val start = System.nanoTime()
        val w = image.width
        val h = image.height
        val src = source?.takeIf { it.width == w && it.height == h }
            ?: Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { source = it }
        val plane = image.planes[0]
        val buffer = plane.buffer
        buffer.rewind()
        if (plane.rowStride == w * 4 && plane.pixelStride == 4) {
            src.copyPixelsFromBuffer(buffer)
        } else {
            val rows = rowCopy?.takeIf { it.capacity() == w * h * 4 } ?: directBuffer(w * h * 4).also { rowCopy = it }
            rows.clear()
            val line = ByteArray(w * 4)
            for (y in 0 until h) {
                buffer.position(y * plane.rowStride)
                buffer.get(line, 0, min(line.size, buffer.remaining()))
                rows.put(line)
            }
            rows.rewind()
            src.copyPixelsFromBuffer(rows)
        }
        val rotation = image.imageInfo.rotationDegrees
        // EdgeTAM wird wie in SAM2 mit gestrecktem Quadrat gefüttert.
        target.set(0f, 0f, FrameSlot.ENC.toFloat(), FrameSlot.ENC.toFloat())
        draw(src, rotation, encCanvas, target)
        // Depth Anything wie in Qualcomms App: Seitenverhältnis bleibt, Rand schwarz.
        val uprightW = if (rotation % 180 == 0) w else h
        val uprightH = if (rotation % 180 == 0) h else w
        val size = FrameSlot.DEPTH.toFloat()
        val scale = min(size / uprightW, size / uprightH)
        val dw = uprightW * scale
        val dh = uprightH * scale
        target.set((size - dw) / 2f, (size - dh) / 2f, (size + dw) / 2f, (size + dh) / 2f)
        depthCanvas.drawColor(Color.BLACK)
        draw(src, rotation, depthCanvas, target)
        slot.depthContent.set(target.left / size, target.top / size, target.right / size, target.bottom / size)
        toFloats(encBitmap, encPixels, encFloats, slot.enc)
        toFloats(depthBitmap, depthPixels, depthFloats, slot.depth)
        slot.timestampNs = image.imageInfo.timestamp
        slot.width = uprightW
        slot.height = uprightH
        slot.rotation = rotation
        slot.prepMs = (System.nanoTime() - start) / 1e6
    }

    private fun draw(src: Bitmap, rotation: Int, canvas: Canvas, dst: RectF) {
        matrix.setRotate(rotation.toFloat())
        bounds.set(0f, 0f, src.width.toFloat(), src.height.toFloat())
        matrix.mapRect(bounds)
        val fit = Matrix()
        fit.setRectToRect(bounds, dst, Matrix.ScaleToFit.FILL)
        matrix.postConcat(fit)
        canvas.drawBitmap(src, matrix, paint)
    }

    private fun toFloats(bitmap: Bitmap, pixels: IntArray, floats: FloatArray, out: ByteBuffer) {
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var j = 0
        for (p in pixels) {
            floats[j] = ((p ushr 16) and 0xFF) * INV255
            floats[j + 1] = ((p ushr 8) and 0xFF) * INV255
            floats[j + 2] = (p and 0xFF) * INV255
            j += 3
        }
        out.rewind()
        out.asFloatBuffer().put(floats)
        out.rewind()
    }

    private companion object {
        const val INV255 = 1f / 255f
    }
}

/** Farbverlauf „Magma“: fern dunkel, nah hell. */
internal object DepthColors {
    private val stops = intArrayOf(
        0x000004, 0x020109, 0x030312, 0x06051A, 0x0A0822, 0x0E0B2B, 0x130D34, 0x180F3D,
        0x1D1147, 0x221150, 0x29115A, 0x2F1163, 0x36106B, 0x3D0F71, 0x440F76, 0x4A1079,
        0x52137C, 0x59157E, 0x5F187F, 0x651A80, 0x6B1D81, 0x721F81, 0x782281, 0x7E2482,
        0x842681, 0x8B2981, 0x912B81, 0x982D80, 0x9E2F7F, 0xA5317E, 0xAB337C, 0xB2357B,
        0xBA3878, 0xC03A76, 0xC73D73, 0xCD4071, 0xD3436E, 0xD9466B, 0xDF4A68, 0xE44F64,
        0xE95462, 0xED5A5F, 0xF1605D, 0xF4675C, 0xF66E5C, 0xF8765C, 0xFA7D5E, 0xFB8560,
        0xFC8E64, 0xFD9668, 0xFE9D6C, 0xFEA571, 0xFEAC76, 0xFEB47B, 0xFEBB81, 0xFEC287,
        0xFECA8D, 0xFED194, 0xFED89A, 0xFDE0A1, 0xFDE7A9, 0xFCEEB0, 0xFCF6B8, 0xFCFDBF
    )

    /** 256 Farben als ARGB. */
    val lut: IntArray = IntArray(256) { i ->
        val pos = i / 255f * (stops.size - 1)
        val a = pos.toInt().coerceAtMost(stops.size - 2)
        val t = pos - a
        val c0 = stops[a]
        val c1 = stops[a + 1]
        fun mix(shift: Int): Int {
            val v0 = (c0 shr shift) and 0xFF
            val v1 = (c1 shr shift) and 0xFF
            return (v0 + (v1 - v0) * t + 0.5f).toInt().coerceIn(0, 255)
        }
        Color.argb(255, mix(16), mix(8), mix(0))
    }
}

/**
 * Macht aus der Tiefenkarte ein kleines Farbbild für die Anzeige. Hell und dunkel
 * passen sich weich an die Szene an, damit das Bild beim Schwenken nicht flackert.
 */
internal class DepthPainter(val width: Int, val height: Int) {
    private val values = FloatArray(width * height)
    private var lo = Float.NaN
    private var hi = Float.NaN

    fun paint(depth: ByteBuffer, content: RectF, out: IntArray) {
        val fb = depth.asFloatBuffer()
        val n = FrameSlot.DEPTH
        var minV = Float.MAX_VALUE
        var maxV = -Float.MAX_VALUE
        var k = 0
        for (y in 0 until height) {
            val sy = ((content.top + (y + 0.5f) / height * content.height()) * n).toInt().coerceIn(0, n - 1)
            for (x in 0 until width) {
                val sx = ((content.left + (x + 0.5f) / width * content.width()) * n).toInt().coerceIn(0, n - 1)
                val v = fb.get(sy * n + sx)
                values[k++] = v
                if (v < minV) minV = v
                if (v > maxV) maxV = v
            }
        }
        if (lo.isNaN()) {
            lo = minV
            hi = maxV
        } else {
            lo += (minV - lo) * 0.25f
            hi += (maxV - hi) * 0.25f
        }
        val range = max(hi - lo, 1e-6f)
        val lut = DepthColors.lut
        for (i in values.indices) {
            val t = ((values[i] - lo) / range * 255f).toInt().coerceIn(0, 255)
            out[i] = lut[t]
        }
    }

    fun reset() {
        lo = Float.NaN
        hi = Float.NaN
    }
}

/** Zeichnet die Maske des verfolgten Objekts: Fläche zart, Rand kräftig. */
internal object MaskPainter {
    const val SIZE = 256

    /** Gibt den Anteil der Bildfläche zurück, den das Objekt bedeckt (0 bis 1). */
    fun paint(mask: FloatArray, out: IntArray, color: Int): Float {
        val fill = (color and 0x00FFFFFF) or (0x52 shl 24)
        val edge = (color and 0x00FFFFFF) or (0xF0 shl 24)
        var count = 0
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {
                val i = y * SIZE + x
                if (mask[i] <= 0f) {
                    out[i] = 0
                    continue
                }
                count++
                val border = x == 0 || y == 0 || x == SIZE - 1 || y == SIZE - 1 ||
                    mask[i - 1] <= 0f || mask[i + 1] <= 0f || mask[i - SIZE] <= 0f || mask[i + SIZE] <= 0f
                out[i] = if (border) edge else fill
            }
        }
        return count / (SIZE * SIZE).toFloat()
    }
}
