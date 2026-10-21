package de.botrik.cinebo.test

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import kotlin.math.roundToInt

/** Bühne der Vorschau: das Kamerabild im Seitenverhältnis des aufrechten 16:9-Bildes, mittig. */
internal class StageLayout(context: Context, private val camera: View) : ViewGroup(context) {

    /** Breite durch Höhe; aufrechtes 16:9-Bild. */
    private val aspect = 9f / 16f

    init {
        addView(camera)
    }

    private fun panel(width: Int, height: Int): Pair<Int, Int> {
        var h = height.toFloat()
        var w = h * aspect
        if (w > width) {
            w = width.toFloat()
            h = w / aspect
        }
        return w.roundToInt() to h.roundToInt()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        val height = MeasureSpec.getSize(heightMeasureSpec)
        val (w, h) = panel(width, height)
        camera.measure(MeasureSpec.makeMeasureSpec(w, MeasureSpec.EXACTLY), MeasureSpec.makeMeasureSpec(h, MeasureSpec.EXACTLY))
        setMeasuredDimension(width, height)
    }

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        val (w, h) = panel(width, height)
        val x = (width - w) / 2
        val y = (height - h) / 2
        camera.layout(x, y, x + w, y + h)
    }
}

/**
 * Liegt über dem Kamerabild und nimmt das Antippen entgegen: Ein Ring zeigt, wo getippt wurde,
 * wie in einer Kamera-App. Was die KI daraus macht, zeigt das Bild selbst (die Unschärfe).
 */
internal class TapOverlay(context: Context) : View(context) {

    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = context.dpf(1.5f)
        color = Color.WHITE
    }
    private var ringX = 0f
    private var ringY = 0f
    private var ringStart = 0L

    /** Wird mit (u, v) von 0 bis 1 im aufrechten Bild aufgerufen; null schaltet das Antippen ab. */
    var onTap: ((Float, Float) -> Unit)? = null
        set(value) {
            field = value
            if (value == null) {
                ringStart = 0L
                invalidate()
            }
        }

    override fun onDraw(canvas: Canvas) {
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
