package de.botrik.cinebo.test

import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Rechnungen der Grafik-Pipeline ohne Android, damit sie sich ohne Handy prüfen lassen.
 * Matrizen sind 4 × 4, spaltenweise wie bei OpenGL (m[spalte * 4 + zeile]).
 *
 * Koordinaten:
 * - Textur (s, t): der Kamerapuffer, wie er im Speicher liegt (quer, wie der Sensor ihn liefert),
 *   t = 0 ist die erste Speicherzeile, also oben.
 * - st-Raum (x, y): wohin die Matrix der SurfaceTexture („st“) abbildet, 0 bis 1, Ursprung unten
 *   links wie bei OpenGL. Bei Kamerabildern enthält st schon die Drehung des Sensors: Der st-Raum
 *   steht aufrecht in der natürlichen Lage des Geräts (beim Find N6 hochkant), siehe [stRotation].
 *   Die Pipeline hält ihre Bilder in diesem Raum. Funktionen hier nennen ihn „Puffer“.
 * - Aufrecht (u, v): das Bild so, wie man es sieht, 0 bis 1, u nach rechts, v nach unten.
 *   Masken und KI-Bilder liegen so (Zeile 0 = oberste Bildzeile).
 */
internal object PipelineMath {

    fun identity(): FloatArray = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)

    /** a · b (erst b, dann a anwenden). */
    fun multiply(a: FloatArray, b: FloatArray): FloatArray {
        val out = FloatArray(16)
        for (col in 0 until 4) {
            for (row in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) sum += a[k * 4 + row] * b[col * 4 + k]
                out[col * 4 + row] = sum
            }
        }
        return out
    }

    /** Wendet m auf den Punkt (x, y, 0, 1) an. */
    fun apply(m: FloatArray, x: Float, y: Float): Pair<Float, Float> =
        (m[0] * x + m[4] * y + m[12]) to (m[1] * x + m[5] * y + m[13])

    /** Umkehrung einer 4 × 4-Matrix; null, wenn sie sich nicht umkehren lässt. */
    fun invert(m: FloatArray): FloatArray? {
        val inv = DoubleArray(16)
        val a = DoubleArray(16) { m[it].toDouble() }
        inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10]
        inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10]
        inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9]
        inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9]
        inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10]
        inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10]
        inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9]
        inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9]
        inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6]
        inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6]
        inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5]
        inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5]
        inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6]
        inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6]
        inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5]
        inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5]
        val det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12]
        if (kotlin.math.abs(det) < 1e-12) return null
        return FloatArray(16) { (inv[it] / det).toFloat() }
    }

    /** 2D-Abbildung s = a·u + b·v + tx, t = c·u + d·v + ty als 4 × 4-Matrix. */
    private fun affine(a: Float, b: Float, c: Float, d: Float, tx: Float, ty: Float): FloatArray =
        floatArrayOf(a, c, 0f, 0f, b, d, 0f, 0f, 0f, 0f, 1f, 0f, tx, ty, 0f, 1f)

    /**
     * Aufrecht → Puffer, für die Drehung im Uhrzeigersinn, die den Puffer aufrecht stellt
     * (wie rotationDegrees bei CameraX). Bei 90° liegt oben links im Bild unten links im Puffer.
     */
    fun uprightToBuffer(rotationDegrees: Int): FloatArray = when (((rotationDegrees % 360) + 360) % 360) {
        0 -> affine(1f, 0f, 0f, -1f, 0f, 1f)        // s = u, t = 1 − v
        90 -> affine(0f, 1f, 1f, 0f, 0f, 0f)        // s = v, t = u
        180 -> affine(-1f, 0f, 0f, 1f, 1f, 0f)      // s = 1 − u, t = v
        270 -> affine(0f, -1f, -1f, 0f, 1f, 1f)     // s = 1 − v, t = 1 − u
        else -> throw IllegalArgumentException("Drehung $rotationDegrees° wird nicht unterstützt")
    }

    /** Puffer → aufrecht (Umkehrung von [uprightToBuffer]); für Maskenkoordinaten. */
    fun bufferToUpright(rotationDegrees: Int): FloatArray = requireNotNull(invert(uprightToBuffer(rotationDegrees)))

    /** Aufrechte Größe aus der Puffergröße. */
    fun uprightSize(bufferW: Int, bufferH: Int, rotationDegrees: Int): Pair<Int, Int> =
        if ((((rotationDegrees % 360) + 360) % 360) % 180 == 0) bufferW to bufferH else bufferH to bufferW

    /**
     * Um wie viel Grad im Uhrzeigersinn die Matrix [st] der SurfaceTexture das gespeicherte Bild
     * schon dreht (0, 90, 180 oder 270). Kamerapuffer tragen die Drehung des Sensors mit; st bildet
     * dann den aufrechten st-Raum auf den quer gespeicherten Puffer ab. Gemessen daran, aus welcher
     * Ecke des Speichers die obere linke Ecke des st-Raums kommt (der Zuschnitt rückt sie nur ein
     * wenig nach innen).
     */
    fun stRotation(st: FloatArray): Int {
        val (s, t) = apply(st, 0f, 1f)
        val left = s < 0.5f
        val top = t < 0.5f
        return when {
            left && top -> 0
            left -> 90
            !top -> 180
            else -> 270
        }
    }

    /** Ob [st] das Bild zusätzlich spiegelt (etwa bei der Frontkamera); dann stimmt [stRotation] allein nicht. */
    fun stMirrored(st: FloatArray): Boolean {
        val rotation = stRotation(st)
        val (s, t) = apply(st, 1f, 1f) // obere rechte Ecke des st-Raums
        val right = s >= 0.5f
        val top = t < 0.5f
        // Ohne Spiegelung kommt sie aus der Ecke, die im Uhrzeigersinn auf die von oben links folgt.
        val expected = when (rotation) {
            0 -> right && top
            90 -> !right && top
            180 -> !right && !top
            else -> right && !top
        }
        return !expected
    }

    /** Größe des st-Raums zur Puffergröße: Dreht st um 90° oder 270°, tauschen Breite und Höhe. */
    fun stSize(bufferW: Int, bufferH: Int, st: FloatArray): Pair<Int, Int> =
        if (stRotation(st) % 180 == 90) bufferH to bufferW else bufferW to bufferH

    /**
     * Drehung vom st-Raum zum aufrechten Bild: CameraX nennt die Drehung vom gespeicherten Puffer
     * zum aufrechten Bild ([rotationDegrees], beim Hochkant-Halten 90); st enthält davon schon
     * [stRotation]. Ohne Abzug drehte die Pipeline doppelt.
     */
    fun uprightFromSt(rotationDegrees: Int, st: FloatArray): Int =
        (((rotationDegrees - stRotation(st)) % 360) + 360) % 360

    /**
     * Wo ein aufrechtes Bild W × H in einem Quadrat der Kantenlänge [size] liegt, wenn es
     * eingepasst wird (Rand schwarz): links, oben, rechts, unten von 0 bis 1. Wie die
     * Vorbereitung auf der CPU bis Stufe 2b und wie in Qualcomms Beispiel-App.
     */
    fun letterbox(uprightW: Int, uprightH: Int, size: Int): FloatArray {
        val s = size.toFloat()
        val scale = min(s / uprightW, s / uprightH)
        val w = uprightW * scale
        val h = uprightH * scale
        return floatArrayOf((s - w) / 2f / s, (s - h) / 2f / s, (s + w) / 2f / s, (s + h) / 2f / s)
    }

    /** Wie die Zahlen eines Quadrats (RGB als float) in eine RGBA32F-Textur gepackt werden. */
    class PackLayout(val size: Int, val width: Int) {
        val total = size * size * 3
        val texels = (total + 3) / 4
        val height = (texels + width - 1) / width
        /** Volle Zeilen, die sich am Stück lesen lassen. */
        val fullRows = texels / width
        /** Texel in der angefangenen letzten Zeile (0, wenn es keine gibt). */
        val rest = texels - fullRows * width
        /** Bytes, die das Lesen schreibt: genau die Größe des Modelleingangs, wenn total durch 4 teilbar. */
        val bytes = texels * 16

        init {
            require(size > 0 && width > 0)
        }
    }

    /**
     * Gewichte und Abstände für die 13 Pixel breite Gaußsche Unschärfe über 7 bilineare
     * Abgriffe (Mitte und drei Paare). [0] ist die Mitte.
     */
    fun gaussTaps(sigma: Double): Pair<FloatArray, FloatArray> {
        val k = DoubleArray(7) { exp(-(it * it) / (2 * sigma * sigma)) }
        val norm = k[0] + 2 * (1 until 7).sumOf { k[it] }
        for (i in k.indices) k[i] /= norm
        val w = FloatArray(4)
        val o = FloatArray(4)
        w[0] = k[0].toFloat()
        for ((j, i) in listOf(1, 3, 5).withIndex()) {
            val sum = k[i] + k[i + 1]
            w[j + 1] = sum.toFloat()
            o[j + 1] = ((i * k[i] + (i + 1) * k[i + 1]) / sum).toFloat()
        }
        return w to o
    }

    /** Welche Masken zu einem Bild gehören: davor (a), danach (b), Mischung, Abstand zur nächsten. */
    data class MaskPick(val a: Int, val b: Int, val mix: Float, val distanceNs: Long)

    /**
     * Sucht zu [frameNs] die KI-Masken direkt davor und danach (Zeitstempel der Kamerabilder,
     * auf denen die KI gerechnet hat). Liegen beide vor, mischt das Bild sie nach der Zeit.
     * Masken weiter als [maxNs] weg zählen nicht. Null: keine passende Maske.
     */
    fun pickMasks(maskNs: LongArray, count: Int, frameNs: Long, maxNs: Long): MaskPick? {
        var before = -1
        var after = -1
        for (i in 0 until count) {
            val t = maskNs[i]
            if (t <= frameNs) {
                if (frameNs - t <= maxNs && (before < 0 || t > maskNs[before])) before = i
            } else {
                if (t - frameNs <= maxNs && (after < 0 || t < maskNs[after])) after = i
            }
        }
        return when {
            before >= 0 && after >= 0 -> {
                val span = (maskNs[after] - maskNs[before]).toDouble()
                val mix = if (span > 0) ((frameNs - maskNs[before]) / span).toFloat() else 0f
                MaskPick(before, after, mix.coerceIn(0f, 1f), min(frameNs - maskNs[before], maskNs[after] - frameNs))
            }
            before >= 0 -> MaskPick(before, before, 0f, frameNs - maskNs[before])
            after >= 0 -> MaskPick(after, after, 0f, maskNs[after] - frameNs)
            else -> null
        }
    }

    /**
     * Soll die Pipeline jetzt ein Bild für die KI vorbereiten? Ja, wenn die KI schon wartet oder
     * voraussichtlich in weniger als [leadMs] das nächste Bild nimmt ([nextTakeNs], System.nanoTime;
     * 0 = unbekannt). So rechnet die KI ohne Pause, und das Bild ist beim Start höchstens etwa
     * [leadMs] alt.
     */
    fun shouldPrepare(nowNs: Long, nextTakeNs: Long, waiting: Boolean, leadMs: Double): Boolean {
        if (waiting) return true
        if (nextTakeNs <= 0L) return false
        return (nextTakeNs - nowNs) / 1e6 <= leadMs
    }

    /**
     * Wärmeregelung: kürzester Abstand zwischen zwei KI-Durchgängen je Stufe der Regelung
     * (0 keine, 1 leicht, 2 mittel, 3 stark, 4 kritisch, 5 Notfall, 6 Abschaltung; siehe
     * [nextLevel]). In 0.3.15 rechnete die KI ungebremst 7,5-mal pro Sekunde auf Spitzenleistung,
     * und OPPO beendete die App nach 3–4 Minuten Dauertest wegen Hitze. Jetzt von Anfang an
     * höchstens 5-mal pro Sekunde (keine, leicht), dann 4 (mittel), 3 (stark) und 2 (kritisch).
     * Ab „Notfall“ beendet die App den Test selbst ([EMERGENCY_STATUS]). Mit 12 Bildern
     * Verzögerung hat jedes Bild bis 5 Durchgänge pro Sekunde Masken davor und danach.
     */
    fun minIntervalMs(level: Int): Long = when {
        level >= 4 -> 500L
        level >= 3 -> 333L
        level >= 2 -> 250L
        else -> 200L
    }

    /** So lange (ms) muss die Wärmestufe unter der Regelstufe liegen, bevor diese eine Stufe zurückgeht. */
    const val LEVEL_HOLD_MS = 90_000L

    /**
     * Nächste Stufe der Regelung: Steigt die Wärmestufe, folgt die Regelung sofort; sinkt sie, geht
     * die Regelung erst nach [LEVEL_HOLD_MS] darunter eine Stufe zurück (dann wieder 90 s für die
     * nächste). Android meldet die Stufen mit Verzögerung, und das Handy kühlt langsamer, als es
     * sich erwärmt. Gibt (Stufe, seit wann die Wärmestufe darunter liegt; 0 = nicht) zurück.
     */
    fun nextLevel(level: Int, status: Int, belowSinceMs: Long, nowMs: Long): Pair<Int, Long> = when {
        status >= level -> status to 0L
        belowSinceMs == 0L -> level to nowMs
        nowMs - belowSinceMs >= LEVEL_HOLD_MS -> (level - 1) to (if (status < level - 1) nowMs else 0L)
        else -> level to belowSinceMs
    }

    /**
     * Wie oft die KI live die Tiefe rechnet: 3 = jeden dritten Durchgang, 0 = gar nicht (nur beim
     * Antippen). In Stufe 3 braucht die Unschärfe die Tiefe noch nicht (nur die Wahl des ganzen
     * Objekts, siehe [SubjectPick]); sie läuft für die Messung mit, seltener als die Verfolgung,
     * und fällt bei Hitze zuerst weg.
     */
    fun depthEvery(level: Int): Int = when {
        level >= 3 -> 0
        level >= 2 -> 6
        level >= 1 -> 4
        else -> 3
    }

    /** Ab dieser Wärmestufe (Notfall) beendet die App einen laufenden Test selbst, bevor das System eingreift. */
    const val EMERGENCY_STATUS = 5

    /** Ab dieser Akkutemperatur (°C) beendet die App einen laufenden Test selbst; in Stufe 2b lief es bei 42 °C noch. */
    const val MAX_BATTERY_C = 44.0

    /**
     * Zu heiß, um weiterzumachen: Wärmestufe ab „Notfall“ ([EMERGENCY_STATUS]) oder der Akku ab
     * [MAX_BATTERY_C]. Ohne Akkutemperatur (NaN) zählt nur die Wärmestufe.
     */
    fun tooHot(status: Int, batteryC: Double): Boolean =
        status >= EMERGENCY_STATUS || (!batteryC.isNaN() && batteryC >= MAX_BATTERY_C)

    /**
     * Zieltemperatur des Akkus vor dem Dauertest: 3 °C über dem Start des Schnelltests, aber
     * zwischen 31 und 34 °C (ohne Wert 33 °C). So beginnt jeder Dauertest ähnlich kühl.
     */
    fun coolTarget(quickStartC: Double): Double = if (quickStartC.isNaN()) 33.0 else (quickStartC + 3.0).coerceIn(31.0, 34.0)

    /** Abgekühlt: Wärmestufe „keine“ und Akku höchstens [targetC] (ohne Temperatur zählt nur die Stufe). */
    fun isCool(stage: String, tempC: Double, targetC: Double): Boolean = stage == "keine" && (tempC.isNaN() || tempC <= targetC)

    /**
     * Stärke der Unschärfe nach dem Alter der nächsten KI-Maske: voll bis 0,4 s, dann gleichmäßig
     * weniger bis 0 bei 1,5 s. So schaltet die Unschärfe nie hart ab, wenn die KI stockt.
     */
    fun maskFade(ageNs: Long): Float = when {
        ageNs <= 400_000_000L -> 1f
        ageNs >= 1_500_000_000L -> 0f
        else -> 1f - (ageNs - 400_000_000L) / 1_100_000_000f
    }

    /** Wahrscheinlichkeit 0 bis 255 zu Logits von −16 bis 16 in Achtelschritten (Sigmoid). */
    private val sigmoid = ByteArray(256) { i ->
        val x = i / 8.0 - 16.0 + 1.0 / 16.0
        (255.0 / (1.0 + exp(-x)) + 0.5).toInt().coerceIn(0, 255).toByte()
    }

    /** Maske der KI (Logits) als Wahrscheinlichkeiten 0 bis 255 für die Grafik-Pipeline. */
    fun maskProbabilities(logits: FloatArray, out: ByteArray) {
        require(out.size >= logits.size)
        for (i in logits.indices) {
            val k = ((logits[i] + 16f) * 8f).toInt()
            out[i] = sigmoid[if (k < 0) 0 else if (k > 255) 255 else k]
        }
    }

    /** Wie viele bilineare Abgriffe je Achse ein Verkleinern um [factor] braucht (höchstens 8). */
    fun taps(factor: Double): Int = max(1, min(8, ceil(factor - 0.001).toInt()))
}
