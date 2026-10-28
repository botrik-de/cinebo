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
 * Leistungsaufnahme des ganzen Handys über den Akku: Strom jede Sekunde
 * (BATTERY_PROPERTY_CURRENT_NOW), Spannung aus der letzten Akku-Meldung, dazu als Gegenproben
 * der Ladungszähler, der Durchschnittsstrom des Akkus und (falls das Handy ihn führt) der
 * Energiezähler. Ergebnis in Watt je Abschnitt ([closeWindow]) und insgesamt ([total]); beim
 * Laden ungültig. Hersteller melden den Strom mal in µA, mal in mA und mit beliebigem
 * Vorzeichen: Kleine Beträge gelten als mA, gezählt wird der Betrag. Wie oft sich der Rohwert
 * von Messung zu Messung ändert, steht dabei: Erneuert das Handy ihn seltener als jede Sekunde,
 * stecken in einem Abschnitt weniger echte Messungen. Eigener Thread.
 */
internal class PowerMeter(context: Context, private val periodMs: Long = 1000L) {
    private val app = context.applicationContext
    private val battery = app.getSystemService(BatteryManager::class.java)

    private class Sums {
        var n = 0
        var watts = 0.0
        var amps = 0.0
        var volts = 0.0
        var charging = 0
        var invalid = 0
        var startMs = SystemClock.elapsedRealtime()
        var chargeStart = Long.MIN_VALUE
        var chargeEnd = Long.MIN_VALUE
        var levelStart = -1
        var levelEnd = -1
        var tempStart = Double.NaN
        var tempEnd = Double.NaN
        /** Messungen, deren Rohwert sich gegenüber der vorigen geändert hat. */
        var changed = 0
        var avgN = 0
        var avgAmps = 0.0
        var energyStart = Long.MIN_VALUE
        var energyEnd = Long.MIN_VALUE
    }

    private val lock = Any()
    private var window = Sums()
    private var all = Sums()
    private var last = ArrayDeque<Double>()
    @Volatile private var running = false
    private var thread: Thread? = null
    /** Rohwerte der ersten Messung, damit sich die Einheit im Bericht nachprüfen lässt. */
    private var firstRaw: JSONObject? = null
    private var previousRaw = Int.MIN_VALUE

    /** Akkutemperatur der letzten Messung in °C (NaN: noch keine); für den Hitzestopp. */
    @Volatile var lastTempC = Double.NaN
        private set

    fun start() {
        if (running) return
        running = true
        synchronized(lock) {
            window = Sums()
            all = Sums()
            last.clear()
            previousRaw = Int.MIN_VALUE
        }
        thread = Thread({
            while (running) {
                sample()
                try {
                    Thread.sleep(periodMs)
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

    private fun sample() {
        val bm = battery ?: return
        val raw = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val sticky = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val mv = sticky?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1
        val plugged = (sticky?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val tenths = sticky?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        val level = sticky?.let {
            val l = it.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val s = it.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            if (l >= 0 && s > 0) l * 100 / s else -1
        } ?: -1
        val charge = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
        val average = PowerMath.amps(bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE))
        val energy = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER)
        val amps = PowerMath.amps(raw)
        val volts = PowerMath.volts(mv)
        if (tenths != Int.MIN_VALUE) lastTempC = tenths / 10.0
        synchronized(lock) {
            if (firstRaw == null) {
                firstRaw = JSONObject().put("stromRoh", raw).put("spannungRohMv", mv).put("ladungRoh", charge)
                    .put("durchschnittRoh", bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE))
                    .put("energieRoh", energy)
            }
            val changed = previousRaw != Int.MIN_VALUE && raw != previousRaw
            previousRaw = raw
            for (s in listOf(window, all)) {
                if (s.levelStart < 0) s.levelStart = level
                s.levelEnd = level
                if (tenths != Int.MIN_VALUE) {
                    if (s.tempStart.isNaN()) s.tempStart = tenths / 10.0
                    s.tempEnd = tenths / 10.0
                }
                if (charge != Long.MIN_VALUE && charge > 0) {
                    if (s.chargeStart == Long.MIN_VALUE) s.chargeStart = charge
                    s.chargeEnd = charge
                }
                if (energy != Long.MIN_VALUE && energy > 0) {
                    if (s.energyStart == Long.MIN_VALUE) s.energyStart = energy
                    s.energyEnd = energy
                }
                when {
                    plugged -> s.charging++
                    amps.isNaN() || volts.isNaN() -> s.invalid++
                    else -> {
                        s.n++
                        s.amps += amps
                        s.volts += volts
                        s.watts += amps * volts
                        if (changed) s.changed++
                        if (!average.isNaN()) {
                            s.avgN++
                            s.avgAmps += average
                        }
                    }
                }
            }
            if (!plugged && !amps.isNaN() && !volts.isNaN()) {
                last.addLast(amps * volts)
                while (last.size > RECENT) last.removeFirst()
            }
        }
    }

    /** Mittlere Leistung der letzten Sekunden (für die Anzeige), NaN ohne Messwerte. */
    fun recentWatts(): Double = synchronized(lock) { if (last.isEmpty()) Double.NaN else last.sum() / last.size }

    /** Abschnitt seit dem letzten Aufruf abschließen. */
    fun closeWindow(): JSONObject = synchronized(lock) {
        val o = json(window)
        window = Sums()
        o
    }

    fun total(): JSONObject = synchronized(lock) { json(all).also { o -> firstRaw?.let { o.put("ersteMessung", it) } } }

    private fun json(s: Sums): JSONObject {
        val o = JSONObject()
        val seconds = (SystemClock.elapsedRealtime() - s.startMs) / 1000.0
        o.put("sekunden", r1(seconds))
        o.put("messwerte", s.n)
        if (s.n > 0) {
            o.put("watt", r2(s.watts / s.n))
            o.put("ampere", r2(s.amps / s.n))
            o.put("volt", r2(s.volts / s.n))
            o.put("stromWechselProzent", Math.round(s.changed * 100.0 / s.n))
        }
        if (s.avgN > 0) o.put("ampereDurchschnittAkku", r2(s.avgAmps / s.avgN))
        if (s.charging > 0) o.put("beimLaden", s.charging)
        if (s.invalid > 0) o.put("ungueltig", s.invalid)
        // Gegenprobe über den Ladungszähler (µAh): mittlerer Strom aus der Abnahme.
        if (s.chargeStart != Long.MIN_VALUE && s.chargeEnd != Long.MIN_VALUE && seconds >= 10) {
            o.put("ladungAbnahmeMah", r1((s.chargeStart - s.chargeEnd) / 1000.0))
            val amps = PowerMath.ampsFromCharge(s.chargeStart, s.chargeEnd, seconds)
            if (!amps.isNaN()) o.put("ampereAusLadung", r2(amps))
        }
        if (seconds >= 10) {
            val watts = PowerMath.wattsFromEnergy(s.energyStart, s.energyEnd, seconds)
            if (!watts.isNaN()) o.put("wattAusEnergie", r2(watts))
        }
        if (s.levelStart >= 0) o.put("akkuProzent", JSONArray().put(s.levelStart).put(s.levelEnd))
        if (!s.tempStart.isNaN()) o.put("akkuGrad", JSONArray().put(s.tempStart).put(s.tempEnd))
        return o
    }

    private companion object {
        const val RECENT = 5
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
