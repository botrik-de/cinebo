package de.botrik.cinebo.test

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

internal fun r1(v: Double): Double = Math.round(v * 10.0) / 10.0

internal fun r2(v: Double): Double = Math.round(v * 100.0) / 100.0

/** Zahl mit deutschem Komma, z. B. 29,97. */
internal fun de(v: Double, digits: Int = 1): String = String.format(Locale.GERMANY, "%.${digits}f", v)

/** Minuten und Sekunden, z. B. 9:05. */
internal fun clock(seconds: Long): String = String.format(Locale.ROOT, "%d:%02d", seconds / 60, seconds % 60)

/** Zeitpunkt mit Zeitzone, z. B. 2026-10-18T15:40:00+02:00. */
internal fun nowIso(): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.ROOT).format(Date())

internal fun fileStamp(): String = SimpleDateFormat("yyyy-MM-dd-HHmm", Locale.ROOT).format(Date())
