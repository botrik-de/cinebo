package de.botrik.cinebo.test

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Warum Android die App zuletzt beendet hat (ApplicationExitInfo). Android hebt die letzten
 * Einträge je App auf, auch über ein Update hinweg; so steht nach einem Rauswurf beim nächsten
 * Start im Bericht, ob es ein Absturz, Speichermangel oder ein Eingriff des Systems war.
 */
internal object ExitHistory {

    /** Die letzten [max] Beendigungen, die neueste zuerst. */
    fun read(context: Context, max: Int = 10): JSONArray {
        val out = JSONArray()
        val am = context.getSystemService(ActivityManager::class.java) ?: return out
        val list = try {
            am.getHistoricalProcessExitReasons(context.packageName, 0, max)
        } catch (e: Exception) {
            return out.put(JSONObject().put("fehler", "${e.javaClass.simpleName}: ${e.message}".take(200)))
        }
        for (info in list) out.put(describe(info))
        return out
    }

    private fun describe(info: ApplicationExitInfo): JSONObject {
        val o = JSONObject()
        o.put("zeit", iso(info.timestamp))
        o.put("zeitMs", info.timestamp)
        o.put("pid", info.pid)
        o.put("grund", reasonName(info.reason))
        o.put("grundNr", info.reason)
        info.description?.takeIf { it.isNotBlank() }?.let { o.put("beschreibung", it.take(300)) }
        // Bei „Signal“ die Signalnummer (9 = beendet), bei „Absturz“ der Exit-Code.
        o.put("status", info.status)
        o.put("wichtigkeit", importanceName(info.importance))
        if (info.pss > 0) o.put("pssMB", info.pss / 1024)
        if (info.rss > 0) o.put("rssMB", info.rss / 1024)
        info.processStateSummary?.let { o.put("zustand", String(it, Charsets.UTF_8)) }
        o.put("prozess", info.processName)
        return o
    }

    fun reasonName(reason: Int): String = when (reason) {
        ApplicationExitInfo.REASON_EXIT_SELF -> "selbst beendet"
        ApplicationExitInfo.REASON_SIGNALED -> "vom System per Signal beendet"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "Speichermangel"
        ApplicationExitInfo.REASON_CRASH -> "Absturz (Java)"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "Absturz (nativ)"
        ApplicationExitInfo.REASON_ANR -> "reagierte nicht (ANR)"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "Start gescheitert"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "Berechtigung geändert"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "zu viel Verbrauch"
        // Auch Systemdienste erzwingen so einen Stopp (etwa Wärmeschutz oder Akku-Optimierung).
        ApplicationExitInfo.REASON_USER_REQUESTED -> "Stopp erzwungen (vom Nutzer oder vom System)"
        ApplicationExitInfo.REASON_USER_STOPPED -> "vom Nutzer gestoppt"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "abhängiger Prozess beendet"
        ApplicationExitInfo.REASON_OTHER -> "anderer Grund des Systems"
        ApplicationExitInfo.REASON_FREEZER -> "eingefroren"
        ApplicationExitInfo.REASON_PACKAGE_STATE_CHANGE -> "App-Zustand geändert"
        ApplicationExitInfo.REASON_PACKAGE_UPDATED -> "App aktualisiert"
        ApplicationExitInfo.REASON_UNKNOWN -> "unbekannt"
        else -> "Grund $reason"
    }

    private fun importanceName(importance: Int): String = when {
        importance <= ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND -> "im Vordergrund"
        importance <= ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND_SERVICE -> "Vordergrunddienst"
        importance <= ActivityManager.RunningAppProcessInfo.IMPORTANCE_VISIBLE -> "sichtbar"
        importance <= ActivityManager.RunningAppProcessInfo.IMPORTANCE_PERCEPTIBLE -> "wahrnehmbar"
        importance < ActivityManager.RunningAppProcessInfo.IMPORTANCE_GONE -> "im Hintergrund"
        else -> "beendet"
    }

    private fun iso(ms: Long): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.ROOT).format(Date(ms))

    /**
     * Kurzer Zustand für den nächsten Beendigungsgrund (nur ASCII, höchstens 128 Bytes, siehe
     * [StateText]): Wird die App beendet, steht dieser Text bei ihrem Eintrag („zustand“).
     * Fehler hier dürfen nie einen Test stören.
     */
    fun setState(context: Context, text: String) {
        try {
            val bytes = StateText.ascii(text).toByteArray(Charsets.US_ASCII)
            context.getSystemService(ActivityManager::class.java)?.setProcessStateSummary(bytes)
        } catch (_: Exception) {
        }
    }
}

/**
 * Leistungsaufnahme des ganzen Handys über den Akku. Den Strom (BATTERY_PROPERTY_CURRENT_NOW)
 * liest ein eigener Thread etwa 4-mal pro Sekunde in zufälligem Abstand (150 bis 350 ms), damit
 * die Messung nicht im Gleichtakt mit Kamera (33 ms) und KI (200 ms) abtastet; Spannung,
 * Temperatur und Akkustand aus der letzten Akku-Meldung einmal pro Sekunde. Gegenproben: der
 * Ladungszähler, der Durchschnittsstrom des Akkus und, falls das Handy ihn führt, der
 * Energiezähler. Ergebnis in Watt und Ampere je Abschnitt ([closeWindow]) und insgesamt
 * ([total]); beim Laden ungültig. Hersteller melden den Strom mal in µA, mal in mA und mit
 * beliebigem Vorzeichen: Kleine Beträge gelten als mA, gezählt wird der Betrag. Wie oft sich der
 * Rohwert ändert, steht dabei; [probe] liest ihn kurz alle 50 ms und zeigt so, wie oft der
 * Akkuwächter ihn erneuert.
 */
internal class PowerMeter(context: Context) {
    private val app = context.applicationContext
    private val battery = app.getSystemService(BatteryManager::class.java)

    private class Sums {
        var n = 0
        var watts = 0.0
        var watts2 = 0.0
        var amps = 0.0
        var volts = 0.0
        var charging = 0
        var invalid = 0
        /** Messungen, deren Rohwert sich gegenüber der vorigen geändert hat. */
        var changed = 0
        var avgN = 0
        var avgAmps = 0.0
        var startMs = SystemClock.elapsedRealtime()
        var chargeStart = Long.MIN_VALUE
        var chargeStartMs = 0L
        var chargeEnd = Long.MIN_VALUE
        var chargeEndMs = 0L
        /** Kleinste Änderung des Ladungszählers (µAh) zwischen zwei Ablesungen. */
        var chargeStep = Long.MAX_VALUE
        var energyStart = Long.MIN_VALUE
        var energyStartMs = 0L
        var energyEnd = Long.MIN_VALUE
        var energyEndMs = 0L
        var levelStart = -1
        var levelEnd = -1
        var tempStart = Double.NaN
        var tempEnd = Double.NaN
    }

    private val lock = Any()
    private var window = Sums()
    private var all = Sums()
    /** (Zeit, Watt) der letzten Sekunden für [recentWatts]. */
    private val recent = ArrayDeque<Pair<Long, Double>>()
    @Volatile private var running = false
    private var thread: Thread? = null
    /** Rohwerte der ersten Messung, damit sich die Einheiten im Bericht nachprüfen lassen. */
    private var firstRaw: JSONObject? = null
    private var previousRaw = Int.MIN_VALUE
    private var previousCharge = Long.MIN_VALUE
    // Aus der letzten Akku-Meldung (einmal pro Sekunde gelesen; nur der Mess-Thread).
    private var volts = Double.NaN
    private var voltsRaw = -1
    private var plugged = false
    private var errors = 0

    /** Akkutemperatur der letzten Akku-Meldung in °C (NaN: noch keine); für den Hitzestopp. */
    @Volatile var lastTempC = Double.NaN
        private set

    fun start() {
        if (running) return
        running = true
        synchronized(lock) {
            window = Sums()
            all = Sums()
            recent.clear()
            previousRaw = Int.MIN_VALUE
            previousCharge = Long.MIN_VALUE
            errors = 0
        }
        thread = Thread({
            val random = java.util.Random()
            var nextSlow = 0L
            while (running) {
                val now = SystemClock.elapsedRealtime()
                val slow = now >= nextSlow
                if (slow) nextSlow = now + SLOW_MS
                // Eine Messung darf nie die App beenden.
                try {
                    sample(slow)
                } catch (_: Throwable) {
                    synchronized(lock) { errors++ }
                }
                try {
                    Thread.sleep(MIN_GAP_MS + random.nextInt(GAP_SPREAD_MS + 1))
                } catch (_: InterruptedException) {
                    break
                }
            }
        }, "strom").apply { isDaemon = true; start() }
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
    }

    private fun sample(slow: Boolean) {
        val bm = battery ?: return
        val now = SystemClock.elapsedRealtime()
        val raw = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val averageRaw = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE)
        var level = -1
        var temp = Double.NaN
        var charge = Long.MIN_VALUE
        var energy = Long.MIN_VALUE
        if (slow) {
            val sticky = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            if (sticky != null) {
                voltsRaw = sticky.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1)
                volts = PowerMath.volts(voltsRaw)
                plugged = sticky.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0
                val tenths = sticky.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
                if (tenths != Int.MIN_VALUE) {
                    temp = tenths / 10.0
                    lastTempC = temp
                }
                val l = sticky.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = sticky.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                if (l >= 0 && scale > 0) level = l * 100 / scale
            }
            charge = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
            energy = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER)
        }
        val amps = PowerMath.amps(raw)
        val average = PowerMath.amps(averageRaw)
        val v = volts
        synchronized(lock) {
            if (firstRaw == null) {
                firstRaw = JSONObject().put("stromRoh", raw).put("durchschnittRoh", averageRaw)
                    .put("spannungRoh", voltsRaw).put("ladungRoh", charge).put("energieRoh", energy)
            }
            val changed = previousRaw != Int.MIN_VALUE && raw != previousRaw
            previousRaw = raw
            val chargeValid = charge != Long.MIN_VALUE && charge > 0
            var step = Long.MAX_VALUE
            if (chargeValid) {
                if (previousCharge != Long.MIN_VALUE && charge != previousCharge) step = kotlin.math.abs(charge - previousCharge)
                previousCharge = charge
            }
            val energyValid = energy != Long.MIN_VALUE && energy > 0
            for (s in listOf(window, all)) {
                if (level >= 0) {
                    if (s.levelStart < 0) s.levelStart = level
                    s.levelEnd = level
                }
                if (!temp.isNaN()) {
                    if (s.tempStart.isNaN()) s.tempStart = temp
                    s.tempEnd = temp
                }
                if (chargeValid) {
                    if (s.chargeStart == Long.MIN_VALUE) {
                        s.chargeStart = charge
                        s.chargeStartMs = now
                    }
                    s.chargeEnd = charge
                    s.chargeEndMs = now
                    if (step < s.chargeStep) s.chargeStep = step
                }
                if (energyValid) {
                    if (s.energyStart == Long.MIN_VALUE) {
                        s.energyStart = energy
                        s.energyStartMs = now
                    }
                    s.energyEnd = energy
                    s.energyEndMs = now
                }
                when {
                    plugged -> s.charging++
                    amps.isNaN() || v.isNaN() -> s.invalid++
                    else -> {
                        val w = amps * v
                        s.n++
                        s.amps += amps
                        s.volts += v
                        s.watts += w
                        s.watts2 += w * w
                        if (changed) s.changed++
                        if (!average.isNaN()) {
                            s.avgN++
                            s.avgAmps += average
                        }
                    }
                }
            }
            if (!plugged && !amps.isNaN() && !v.isNaN()) {
                recent.addLast(now to amps * v)
                while (recent.isNotEmpty() && now - recent.first().first > RECENT_MS) recent.removeFirst()
            }
        }
    }

    /** Mittlere Leistung der letzten Sekunden (für die Anzeige), NaN ohne Messwerte. */
    fun recentWatts(): Double = synchronized(lock) { if (recent.isEmpty()) Double.NaN else recent.sumOf { it.second } / recent.size }

    /** Abschnitt seit dem letzten Aufruf abschließen. */
    fun closeWindow(): JSONObject = synchronized(lock) {
        val o = json(window)
        window = Sums()
        o
    }

    fun total(): JSONObject = synchronized(lock) {
        json(all).also { o ->
            firstRaw?.let { o.put("ersteMessung", it) }
            if (errors > 0) o.put("messfehler", errors)
        }
    }

    /**
     * Liest den Strom [durationMs] lang alle [stepMs] als Rohwert mit Zeit. Daran sieht man, wie oft
     * der Akkuwächter ihn erneuert und ob er ihn glättet. Blockiert: nicht auf dem Hauptthread.
     */
    fun probe(durationMs: Long = 3000L, stepMs: Long = 50L): JSONObject {
        val bm = battery ?: return JSONObject().put("fehler", "kein Zugriff auf den Akku")
        val ms = JSONArray()
        val raw = JSONArray()
        val distinct = HashSet<Int>()
        var changes = 0
        var previous = Int.MIN_VALUE
        val t0 = SystemClock.elapsedRealtime()
        try {
            while (SystemClock.elapsedRealtime() - t0 < durationMs) {
                val value = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                ms.put(SystemClock.elapsedRealtime() - t0)
                raw.put(value)
                if (previous != Int.MIN_VALUE && value != previous) changes++
                previous = value
                distinct.add(value)
                Thread.sleep(stepMs)
            }
        } catch (_: InterruptedException) {
        } catch (e: Exception) {
            return JSONObject().put("fehler", e.toString().take(200))
        }
        return JSONObject().put("schrittMs", stepMs).put("wechsel", changes).put("verschiedeneWerte", distinct.size)
            .put("ms", ms).put("roh", raw)
    }

    private fun json(s: Sums): JSONObject {
        val o = JSONObject()
        val seconds = (SystemClock.elapsedRealtime() - s.startMs) / 1000.0
        o.put("sekunden", r1(seconds))
        o.put("messwerte", s.n)
        if (s.n > 0) {
            val mean = s.watts / s.n
            o.put("watt", r2(mean))
            o.put("ampere", r4(s.amps / s.n))
            o.put("volt", r2(s.volts / s.n))
            if (s.n > 1) o.put("wattStreuung", r2(kotlin.math.sqrt(maxOf(0.0, (s.watts2 - s.n * mean * mean) / (s.n - 1)))))
            o.put("stromWechselProzent", Math.round(s.changed * 100.0 / s.n))
        }
        if (s.avgN > 0) o.put("ampereDurchschnittAkku", r4(s.avgAmps / s.avgN))
        if (s.charging > 0) o.put("beimLaden", s.charging)
        if (s.invalid > 0) o.put("ungueltig", s.invalid)
        // Gegenproben über die Zähler: über die Zeit zwischen erster und letzter Ablesung, und erst
        // ab einer Minute (sonst ist eine Zählerstufe ein großer Teil davon).
        if (s.chargeStart != Long.MIN_VALUE && s.chargeEnd != Long.MIN_VALUE) {
            o.put("ladungAbnahmeMah", r1((s.chargeStart - s.chargeEnd) / 1000.0))
            if (s.chargeStep != Long.MAX_VALUE) o.put("ladungStufeUah", s.chargeStep)
            val span = (s.chargeEndMs - s.chargeStartMs) / 1000.0
            if (span >= CROSS_CHECK_S) {
                val amps = PowerMath.ampsFromCharge(s.chargeStart, s.chargeEnd, span)
                if (!amps.isNaN() && amps <= MAX_AMPS) o.put("ampereAusLadung", r2(amps))
            }
        }
        if (s.energyStart != Long.MIN_VALUE && s.energyEnd != Long.MIN_VALUE) {
            val span = (s.energyEndMs - s.energyStartMs) / 1000.0
            if (span >= CROSS_CHECK_S) {
                val watts = PowerMath.wattsFromEnergy(s.energyStart, s.energyEnd, span)
                if (!watts.isNaN() && watts <= MAX_WATTS) o.put("wattAusEnergie", r2(watts))
            }
        }
        if (s.levelStart >= 0) o.put("akkuProzent", JSONArray().put(s.levelStart).put(s.levelEnd))
        if (!s.tempStart.isNaN()) o.put("akkuGrad", JSONArray().put(s.tempStart).put(s.tempEnd))
        return o
    }

    private companion object {
        /** Abstand der Strommessungen: 150 bis 350 ms, zufällig. */
        const val MIN_GAP_MS = 150L
        const val GAP_SPREAD_MS = 200
        /** So oft (ms) liest der Thread Spannung, Temperatur und Zähler. */
        const val SLOW_MS = 1000L
        /** Über so viele Millisekunden mittelt [recentWatts]. */
        const val RECENT_MS = 5000L
        /** Gegenproben über die Zähler erst ab dieser Zeitspanne (s). */
        const val CROSS_CHECK_S = 60.0
        /** Mehr ist nicht plausibel (ein Zählersprung), solche Gegenproben fallen weg. */
        const val MAX_AMPS = 10.0
        const val MAX_WATTS = 40.0
    }
}

/**
 * Wechsel der Wärmestufe mit Zeitpunkt (Sekunden seit [start]) und Akkutemperatur; Android meldet
 * sie selbst (Listener), so fehlt kein kurzer Ausflug in eine höhere Stufe.
 */
internal class ThermalLog(context: Context) {
    private val app = context.applicationContext
    private val pm = app.getSystemService(PowerManager::class.java)
    private val entries = JSONArray()
    private var startMs = 0L
    @Volatile var status = 0
        private set
    private var listener: PowerManager.OnThermalStatusChangedListener? = null

    fun start() {
        startMs = SystemClock.elapsedRealtime()
        status = pm?.currentThermalStatus ?: 0
        note(status)
        val l = PowerManager.OnThermalStatusChangedListener { s ->
            status = s
            note(s)
        }
        listener = l
        try {
            pm?.addThermalStatusListener(app.mainExecutor, l)
        } catch (_: Exception) {
        }
    }

    fun stop() {
        listener?.let { l -> try { pm?.removeThermalStatusListener(l) } catch (_: Exception) {} }
        listener = null
    }

    @Synchronized private fun note(s: Int) {
        val o = JSONObject().put("s", r1((SystemClock.elapsedRealtime() - startMs) / 1000.0)).put("stufe", DeviceHeat.stageNameAt(s.coerceIn(0, 6)))
        val sticky = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tenths = sticky?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        if (tenths != Int.MIN_VALUE) o.put("akkuGrad", tenths / 10.0)
        // Höchstens die ersten 60 Wechsel; Android meldet Stufen mit eigener Trägheit.
        if (entries.length() < 60) entries.put(o)
    }

    @Synchronized fun toJson(): JSONArray = JSONArray(entries.toString())
}
