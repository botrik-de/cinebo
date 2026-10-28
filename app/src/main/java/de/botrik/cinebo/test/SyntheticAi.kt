package de.botrik.cinebo.test

import android.os.SystemClock
import kotlin.math.max

/**
 * Ersatz-KI für den Strombedarf: nimmt KI-Bilder [hz]-mal pro Sekunde wie die echte KI und gibt
 * sofort eine feste Ellipse als Maske zurück. So läuft die ganze Arbeit der Grafik-Pipeline
 * samt Unschärfe, Lesen der KI-Bilder (Tiefenbild jedes [depthEvery]. Mal, 0 = nie) und Masken,
 * aber nichts auf dem KI-Chip. Der Unterschied zur echten KI im selben Takt ist dann genau, was
 * KI-Chip und KI-Thread verbrauchen.
 */
internal class SyntheticAi(
    private val pool: FramePool,
    private val hz: Double = 5.0,
    private val depthEvery: Int = 0,
) : AiDemand {
    @Volatile private var waitingFlag = false
    @Volatile private var nextTake = 0L
    @Volatile private var taken = 0
    @Volatile private var running = false
    private var thread: Thread? = null

    /** Bekommt die Ellipse zu jedem Bild (die Pipeline des Abschnitts). */
    @Volatile var sink: MaskSink? = null

    override val nextTakeNs: Long get() = nextTake
    override val waiting: Boolean get() = waitingFlag
    override val depthNeeded: Boolean get() = depthEvery > 0 && (taken + 1) % depthEvery == 0

    /** Durchgänge seit dem Start. */
    val count: Int get() = taken

    fun start() {
        if (running) return
        running = true
        taken = 0
        thread = Thread({ loop() }, "ersatz-ki").apply { isDaemon = true; start() }
    }

    fun stop() {
        running = false
        thread?.join(1000)
        thread = null
        waitingFlag = false
        nextTake = 0L
    }

    private fun loop() {
        val periodMs = 1000.0 / hz
        while (running) {
            val begin = SystemClock.elapsedRealtime()
            waitingFlag = true
            val slot = try {
                pool.take(100)
            } finally {
                waitingFlag = false
            } ?: continue
            try {
                // Eigene Generation: Die Pipeline mischt die Ellipse nie mit einer Maske der echten KI.
                sink?.onMask(slot.timestampNs, ELLIPSE, true, GENERATION)
                taken++
            } finally {
                pool.release(slot)
            }
            val rest = periodMs - (SystemClock.elapsedRealtime() - begin)
            nextTake = System.nanoTime() + (max(rest, 0.0) * 1e6).toLong()
            if (rest > 0) {
                try {
                    Thread.sleep(rest.toLong())
                } catch (_: InterruptedException) {
                    break
                }
            }
        }
    }

    companion object {
        /** Generation der Ersatz-Masken; die echte KI zählt ab 1 aufwärts. */
        const val GENERATION = -1

        /** Ellipse in der Bildmitte (Breite 50 %, Höhe 70 %), weicher Rand über 2 Zellen; Wahrscheinlichkeit 0 bis 255. */
        val ELLIPSE: ByteArray = ByteArray(MASK_SIZE * MASK_SIZE).also { m ->
            val n = MASK_SIZE
            for (y in 0 until n) {
                for (x in 0 until n) {
                    val dx = ((x + 0.5) / n - 0.5) / 0.25
                    val dy = ((y + 0.5) / n - 0.5) / 0.35
                    val r = kotlin.math.sqrt(dx * dx + dy * dy)
                    val edge = (1.0 - r) * n * 0.25 / 2.0 // Abstand zum Rand in etwa Zellen / 2
                    val p = (0.5 + edge).coerceIn(0.0, 1.0)
                    m[y * n + x] = (p * 255 + 0.5).toInt().toByte()
                }
            }
        }
    }
}
