package de.botrik.cinebo.test

import kotlin.math.abs

/*
 * Rechnungen für Strommessung, Verlauf und Beendigungsgrund, ganz ohne Android: So lassen sie
 * sich auf dem Rechner prüfen (siehe PowerMathTest).
 */

/** Rechnungen der Strommessung. */
internal object PowerMath {
    /**
     * Strom in Ampere aus dem Rohwert von CURRENT_NOW: Android sagt µA; manche Hersteller melden
     * mA. Beträge unter 20 000 gelten als mA (20 A in mA wären unmöglich, 20 mA in µA bei laufender
     * Kamera auch). Vorzeichen egal; 0 und der Fehlwert (Integer.MIN_VALUE) sind ungültig.
     */
    fun amps(raw: Int): Double {
        if (raw == Int.MIN_VALUE || raw == 0) return Double.NaN
        val v = abs(raw.toDouble())
        return if (v < 20_000) v / 1_000.0 else v / 1_000_000.0
    }

    /** Spannung in Volt aus EXTRA_VOLTAGE (mV; manche Geräte melden V). */
    fun volts(raw: Int): Double = when {
        raw <= 0 -> Double.NaN
        raw < 100 -> raw.toDouble()
        else -> raw / 1000.0
    }

    /** Mittlerer Strom aus der Abnahme des Ladungszählers (µAh) über [seconds]; NaN, wenn er nicht sinkt. */
    fun ampsFromCharge(start: Long, end: Long, seconds: Double): Double {
        val drop = start - end
        if (drop <= 0 || seconds <= 0) return Double.NaN
        return drop / 1e6 * 3600.0 / seconds
    }

    /**
     * Mittlere Leistung in Watt aus der Abnahme des Energiezählers (ENERGY_COUNTER, nWh) über
     * [seconds]; NaN, wenn das Handy ihn nicht führt (Fehlwert, 0) oder er nicht sinkt.
     */
    fun wattsFromEnergy(startNwh: Long, endNwh: Long, seconds: Double): Double {
        if (startNwh == Long.MIN_VALUE || endNwh == Long.MIN_VALUE || startNwh <= 0 || endNwh <= 0) return Double.NaN
        val drop = startNwh - endNwh
        if (drop <= 0 || seconds <= 0) return Double.NaN
        return drop / 1e9 * 3600.0 / seconds
    }
}

/** Trend einer Zeitreihe, etwa der Akkutemperatur im Dauertest. */
internal object Trend {
    /**
     * Steigung der Ausgleichsgeraden je Minute aus Punkten ([seconds], [values]); Punkte mit NaN
     * fallen weg. NaN bei weniger als zwei Punkten oder ohne Zeitspanne.
     */
    fun slopePerMinute(seconds: DoubleArray, values: DoubleArray): Double {
        require(seconds.size == values.size)
        var n = 0
        var sx = 0.0
        var sy = 0.0
        for (i in seconds.indices) {
            if (seconds[i].isNaN() || values[i].isNaN()) continue
            n++
            sx += seconds[i]
            sy += values[i]
        }
        if (n < 2) return Double.NaN
        val mx = sx / n
        val my = sy / n
        var sxx = 0.0
        var sxy = 0.0
        for (i in seconds.indices) {
            if (seconds[i].isNaN() || values[i].isNaN()) continue
            val dx = seconds[i] - mx
            sxx += dx * dx
            sxy += dx * (values[i] - my)
        }
        if (sxx <= 0.0) return Double.NaN
        return sxy / sxx * 60.0
    }

    /** Mittelwert ohne NaN; NaN, wenn keiner bleibt. */
    fun mean(values: DoubleArray): Double {
        var n = 0
        var sum = 0.0
        for (v in values) {
            if (v.isNaN()) continue
            n++
            sum += v
        }
        return if (n == 0) Double.NaN else sum / n
    }
}

/**
 * Kurzer Zustand für den Beendigungsgrund (setProcessStateSummary): Android hebt ihn beim
 * nächsten Rauswurf mit auf. Nur ASCII und höchstens [MAX_BYTES] Bytes, als „schlüssel=wert“,
 * getrennt durch Leerzeichen; fehlende Werte (null, NaN, leer) fallen weg.
 */
internal object StateText {
    const val MAX_BYTES = 128

    fun line(vararg parts: Pair<String, Any?>): String {
        val out = StringBuilder()
        for ((key, value) in parts) {
            val text = when (value) {
                null -> continue
                is Double -> if (value.isFinite()) one(value) else continue
                is Float -> if (value.isFinite()) one(value.toDouble()) else continue
                else -> value.toString()
            }
            if (text.isEmpty()) continue
            if (out.isNotEmpty()) out.append(' ')
            out.append(key).append('=').append(text)
        }
        return ascii(out.toString())
    }

    /** Nur ASCII: Umlaute umschrieben, das Gradzeichen fällt weg, alles andere wird „?“; höchstens [max] Zeichen. */
    fun ascii(text: String, max: Int = MAX_BYTES): String {
        val out = StringBuilder(text.length)
        for (c in text) {
            when (c) {
                'ä' -> out.append("ae")
                'ö' -> out.append("oe")
                'ü' -> out.append("ue")
                'Ä' -> out.append("Ae")
                'Ö' -> out.append("Oe")
                'Ü' -> out.append("Ue")
                'ß' -> out.append("ss")
                '°' -> Unit
                else -> out.append(if (c.code in 32..126) c else '?')
            }
        }
        return if (out.length > max) out.substring(0, max) else out.toString()
    }

    /** Eine Nachkommastelle mit Punkt, unabhängig von der Spracheinstellung. */
    private fun one(v: Double): String = (Math.round(v * 10.0) / 10.0).toString()
}
