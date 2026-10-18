package de.botrik.cinebo.test

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.os.SystemClock
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Bühne der Vorschau: links das Kamerabild, rechts (nur während der KI läuft) die Tiefe.
 * Beide Felder haben das Seitenverhältnis des aufrechten Kamerabilds und stehen mittig.
 */
internal class StageLayout(context: Context, private val camera: View, private val depth: View) : ViewGroup(context) {

    /** Breite durch Höhe eines Feldes; aufrechtes 16:9-Bild. */
    private val aspect = 9f / 16f
    private val gap = context.dp(10f)

    var showDepth = false
        set(value) {
            if (field == value) return
            field = value
            depth.visibility = if (value) VISIBLE else GONE
            requestLayout()
        }

    init {
        addView(camera)
        addView(depth)
        depth.visibility = GONE
    }

    private fun panel(width: Int, height: Int): Pair<Int, Int> {
        val n = if (showDepth) 2 else 1
        var h = height.toFloat()
        var w = h * aspect
        if (n * w + (n - 1) * gap > width) {
            w = (width - (n - 1) * gap) / n.toFloat()
            h = w / aspect
        }
        return w.roundToInt() to h.roundToInt()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        val height = MeasureSpec.getSize(heightMeasureSpec)
        val (w, h) = panel(width, height)
        val ws = MeasureSpec.makeMeasureSpec(w, MeasureSpec.EXACTLY)
        val hs = MeasureSpec.makeMeasureSpec(h, MeasureSpec.EXACTLY)
        camera.measure(ws, hs)
        if (showDepth) depth.measure(ws, hs)
        setMeasuredDimension(width, height)
    }

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        val (w, h) = panel(width, height)
        val n = if (showDepth) 2 else 1
        var x = (width - (n * w + (n - 1) * gap)) / 2
        val y = (height - h) / 2
        camera.layout(x, y, x + w, y + h)
        x += w + gap
        if (showDepth) depth.layout(x, y, x + w, y + h)
    }
}

/**
 * Liegt über dem Kamerabild: zeigt die Maske des verfolgten Objekts und nimmt das
 * Antippen entgegen (Ring als Rückmeldung wie in einer Kamera-App).
 */
internal class MaskOverlay(context: Context) : View(context) {

    private val bitmap = Bitmap.createBitmap(MaskPainter.SIZE, MaskPainter.SIZE, Bitmap.Config.ARGB_8888)
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = context.dpf(1.5f)
        color = Color.WHITE
    }
    private val target = Rect()
    private var maskShown = false
    private var ringX = 0f
    private var ringY = 0f
    private var ringStart = 0L

    /** Wird mit (u, v) von 0 bis 1 aufgerufen; null schaltet das Antippen ab. */
    var onTap: ((Float, Float) -> Unit)? = null

    fun showMask(pixels: IntArray) {
        bitmap.setPixels(pixels, 0, MaskPainter.SIZE, 0, 0, MaskPainter.SIZE, MaskPainter.SIZE)
        maskShown = true
        invalidate()
    }

    fun clearMask() {
        maskShown = false
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        if (maskShown) {
            target.set(0, 0, width, height)
            canvas.drawBitmap(bitmap, null, target, bitmapPaint)
        }
        val age = SystemClock.uptimeMillis() - ringStart
        if (ringStart > 0 && age < RING_MS) {
            val f = age / RING_MS.toFloat()
            ringPaint.alpha = ((1f - f) * 255).roundToInt()
            val r = context.dpf(26f) - context.dpf(8f) * f
            canvas.drawCircle(ringX, ringY, r, ringPaint)
            postInvalidateOnAnimation()
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        val tap = onTap ?: return false
        if (event.actionMasked == MotionEvent.ACTION_UP && width > 0 && height > 0) {
            ringX = event.x
            ringY = event.y
            ringStart = SystemClock.uptimeMillis()
            invalidate()
            tap((event.x / width).coerceIn(0f, 1f), (event.y / height).coerceIn(0f, 1f))
            performClick()
        }
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private companion object {
        const val RING_MS = 650L
    }
}

/** Zeigt die Tiefenkarte (kleines Farbbild, gestreckt auf das Feld). */
internal class DepthView(context: Context) : View(context) {

    private val bitmap = Bitmap.createBitmap(AiEngine.DEPTH_W, AiEngine.DEPTH_H, Bitmap.Config.ARGB_8888)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val target = Rect()
    private var shown = false

    fun show(pixels: IntArray) {
        bitmap.setPixels(pixels, 0, AiEngine.DEPTH_W, 0, 0, AiEngine.DEPTH_W, AiEngine.DEPTH_H)
        shown = true
        invalidate()
    }

    fun clear() {
        shown = false
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        if (!shown) return
        val w = min(width, height * AiEngine.DEPTH_W / AiEngine.DEPTH_H)
        val h = w * AiEngine.DEPTH_H / AiEngine.DEPTH_W
        val x = (width - w) / 2
        val y = (height - h) / 2
        target.set(x, y, x + w, y + h)
        canvas.drawBitmap(bitmap, null, target, paint)
    }
}
