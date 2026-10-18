package de.botrik.cinebo.test

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.text.TextUtils
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import kotlin.math.min
import kotlin.math.roundToInt

/** Farben der Test-App: dunkel wie ein Kinosaal, Rot nur für die Aufnahme. */
internal object Palette {
    val background = Color.rgb(0x11, 0x12, 0x14)
    val surface = Color.rgb(0x1A, 0x1C, 0x1F)
    val line = Color.rgb(0x2C, 0x30, 0x35)
    val text = Color.rgb(0xF2, 0xF4, 0xF5)
    val textDim = Color.rgb(0x9B, 0xA1, 0xA6)
    val textFaint = Color.rgb(0x6B, 0x71, 0x77)
    val red = Color.rgb(0xE5, 0x48, 0x4D)
    val green = Color.rgb(0x3D, 0xD6, 0x8C)
    val amber = Color.rgb(0xF5, 0xB8, 0x3D)
}

internal object Fonts {
    val regular: Typeface = Typeface.create(Typeface.DEFAULT, 400, false)
    val medium: Typeface = Typeface.create(Typeface.DEFAULT, 500, false)
}

internal fun Context.dpf(value: Float): Float =
    TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value, resources.displayMetrics)

internal fun Context.dp(value: Float): Int = dpf(value).roundToInt()

internal fun Context.label(sizeSp: Float, color: Int, face: Typeface = Fonts.regular): TextView =
    TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp)
        setTextColor(color)
        typeface = face
        setLineSpacing(0f, 1.25f)
    }

internal fun Context.dot(color: Int, sizeDp: Float): GradientDrawable = GradientDrawable().apply {
    shape = GradientDrawable.OVAL
    setColor(color)
    val px = dp(sizeDp)
    setSize(px, px)
}

/** Zustand einer Zeile in der Ergebnisliste. */
internal enum class Mark { OPEN, RUNNING, OK, WARN, FAIL }

/** Runder Knopf, entweder gefüllt (Hauptaktion) oder mit feinem Rand. */
internal class PillButton(context: Context) : TextView(context) {

    init {
        gravity = Gravity.CENTER
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
        typeface = Fonts.medium
        isAllCaps = false
        maxLines = 1
        ellipsize = TextUtils.TruncateAt.END
        setPadding(context.dp(16f), 0, context.dp(16f), 0)
        isClickable = true
        isFocusable = true
        outlined()
    }

    fun filled(fill: Int, textColor: Int) {
        setTextColor(textColor)
        background = shape(fill, null, if (fill == Palette.text) 0x22000000 else 0x33FFFFFF)
    }

    fun outlined() {
        setTextColor(Palette.text)
        background = shape(Color.TRANSPARENT, Palette.line, 0x33FFFFFF)
    }

    private fun shape(fill: Int, stroke: Int?, ripple: Int): RippleDrawable {
        val radius = context.dpf(26f)
        val content = GradientDrawable().apply {
            cornerRadius = radius
            setColor(fill)
            if (stroke != null) setStroke(context.dp(1f), stroke)
        }
        val mask = GradientDrawable().apply {
            cornerRadius = radius
            setColor(Color.WHITE)
        }
        return RippleDrawable(ColorStateList.valueOf(ripple), content, mask)
    }

    override fun setEnabled(enabled: Boolean) {
        super.setEnabled(enabled)
        alpha = if (enabled) 1f else 0.38f
    }
}

/** Kreis-Symbol: offen, läuft, gut, Hinweis oder Fehler. */
internal class StatusMark(context: Context) : View(context) {

    var mark: Mark = Mark.OPEN
        set(value) {
            field = value
            invalidate()
        }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val path = Path()

    override fun onDraw(canvas: Canvas) {
        val size = min(width, height).toFloat()
        if (size <= 0f) return
        val cx = width / 2f
        val cy = height / 2f
        val r = size / 2f - context.dpf(1f)
        val stroke = context.dpf(1.75f)
        paint.strokeCap = Paint.Cap.ROUND
        paint.strokeJoin = Paint.Join.ROUND
        when (mark) {
            Mark.OPEN -> {
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = context.dpf(1.5f)
                paint.color = Palette.textFaint
                canvas.drawCircle(cx, cy, r, paint)
            }
            Mark.RUNNING -> {
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = context.dpf(1.5f)
                paint.color = Palette.amber
                canvas.drawCircle(cx, cy, r, paint)
                paint.style = Paint.Style.FILL
                canvas.drawCircle(cx, cy, r * 0.42f, paint)
            }
            Mark.OK -> {
                filledCircle(canvas, cx, cy, r, Palette.green)
                path.reset()
                path.moveTo(cx - r * 0.42f, cy + r * 0.02f)
                path.lineTo(cx - r * 0.1f, cy + r * 0.34f)
                path.lineTo(cx + r * 0.45f, cy - r * 0.3f)
                symbol(canvas, stroke)
            }
            Mark.WARN -> {
                filledCircle(canvas, cx, cy, r, Palette.amber)
                path.reset()
                path.moveTo(cx, cy - r * 0.45f)
                path.lineTo(cx, cy + r * 0.08f)
                symbol(canvas, stroke)
                paint.style = Paint.Style.FILL
                canvas.drawCircle(cx, cy + r * 0.42f, stroke * 0.7f, paint)
            }
            Mark.FAIL -> {
                filledCircle(canvas, cx, cy, r, Palette.red)
                val a = r * 0.34f
                path.reset()
                path.moveTo(cx - a, cy - a)
                path.lineTo(cx + a, cy + a)
                path.moveTo(cx + a, cy - a)
                path.lineTo(cx - a, cy + a)
                symbol(canvas, stroke)
            }
        }
    }

    private fun filledCircle(canvas: Canvas, cx: Float, cy: Float, r: Float, color: Int) {
        paint.style = Paint.Style.FILL
        paint.color = color
        canvas.drawCircle(cx, cy, r, paint)
    }

    private fun symbol(canvas: Canvas, stroke: Float) {
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = stroke
        paint.color = Palette.background
        canvas.drawPath(path, paint)
        paint.style = Paint.Style.FILL
    }
}

/** Dünner Fortschrittsbalken mit runden Enden. */
internal class ProgressLine(context: Context) : View(context) {

    var fraction = 0f
        set(value) {
            field = value.coerceIn(0f, 1f)
            invalidate()
        }

    var color = Palette.red
        set(value) {
            field = value
            invalidate()
        }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val rect = RectF()

    override fun onDraw(canvas: Canvas) {
        val h = height.toFloat()
        val r = h / 2f
        paint.color = Palette.line
        rect.set(0f, 0f, width.toFloat(), h)
        canvas.drawRoundRect(rect, r, r, paint)
        if (fraction > 0f) {
            paint.color = color
            rect.set(0f, 0f, maxOf(h, width * fraction), h)
            canvas.drawRoundRect(rect, r, r, paint)
        }
    }
}

/** Eine Zeile der Ergebnisliste: Symbol, Titel und Ergebnis. */
internal class ResultRow(context: Context, title: String) : LinearLayout(context) {

    private val markView = StatusMark(context)
    private val detail = context.label(13f, Palette.textDim)

    init {
        orientation = HORIZONTAL
        gravity = Gravity.TOP
        setPadding(context.dp(16f), context.dp(14f), context.dp(16f), context.dp(14f))
        addView(markView, LinearLayout.LayoutParams(context.dp(20f), context.dp(20f)).apply {
            marginEnd = context.dp(14f)
        })
        val column = LinearLayout(context).apply { orientation = VERTICAL }
        val titleView = context.label(15f, Palette.text, Fonts.medium).apply { text = title }
        column.addView(titleView)
        column.addView(
            detail,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                topMargin = context.dp(3f)
            }
        )
        addView(column, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
    }

    fun show(mark: Mark, text: String) {
        markView.mark = mark
        detail.text = text
        detail.setTextColor(if (mark == Mark.OPEN) Palette.textFaint else Palette.textDim)
    }
}
