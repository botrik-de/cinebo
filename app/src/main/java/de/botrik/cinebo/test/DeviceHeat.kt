package de.botrik.cinebo.test

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import org.json.JSONObject

/** Wärme und Akku: was Android fremden Apps darüber verrät. */
object DeviceHeat {

    private val stageNames = mapOf(
        PowerManager.THERMAL_STATUS_NONE to "keine",
        PowerManager.THERMAL_STATUS_LIGHT to "leicht",
        PowerManager.THERMAL_STATUS_MODERATE to "mittel",
        PowerManager.THERMAL_STATUS_SEVERE to "stark",
        PowerManager.THERMAL_STATUS_CRITICAL to "kritisch",
        PowerManager.THERMAL_STATUS_EMERGENCY to "Notfall",
        PowerManager.THERMAL_STATUS_SHUTDOWN to "Abschaltung"
    )

    /** Reihenfolge der Stufen, damit sich die heißeste finden lässt. */
    val stageOrder = listOf("keine", "leicht", "mittel", "stark", "kritisch", "Notfall", "Abschaltung")

    fun stageName(status: Int): String = stageNames[status] ?: status.toString()

    fun read(context: Context): JSONObject {
        val o = JSONObject()
        val pm = context.getSystemService(PowerManager::class.java)
        o.put("stufe", stageName(pm.currentThermalStatus))
        // 1,0 heißt: Das System drosselt gleich stark. Kleiner ist besser.
        val index = pm.getThermalHeadroom(10)
        if (!index.isNaN()) o.put("drosselIndex", r2(index.toDouble()))
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        if (battery != null) {
            val tenths = battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            if (tenths != Int.MIN_VALUE) o.put("akkuGrad", tenths / 10.0)
            val level = battery.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = battery.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            if (level >= 0 && scale > 0) o.put("akkuProzent", level * 100 / scale)
            o.put("laedt", battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0)
        }
        return o
    }

    /** Name der Wärmestufe zu ihrer Nummer (0 keine bis 6 Abschaltung). */
    fun stageNameAt(status: Int): String = stageOrder.getOrElse(status) { status.toString() }

    /** Ab welchem Drosselindex das System welche Stufe meldet (ab Android 15). */
    fun thresholds(context: Context): JSONObject? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.VANILLA_ICE_CREAM) return null
        val pm = context.getSystemService(PowerManager::class.java)
        val o = JSONObject()
        for ((status, value) in pm.thermalHeadroomThresholds) o.put(stageName(status), r2(value.toDouble()))
        return o
    }
}

/**
 * Wärmeregelung der KI: liest die Wärmestufe höchstens alle 2 s und sagt, wie viel Abstand
 * zwei KI-Durchgänge mindestens haben sollen (siehe [PipelineMath.minIntervalMs]). Zählt mit,
 * wie lange jede Stufe galt und wie oft die KI deshalb wartete. Nur vom KI-Thread benutzen.
 */
internal class HeatGovernor(context: Context) {
    private val pm = context.getSystemService(PowerManager::class.java)
    private val seconds = DoubleArray(7)
    private var lastRead = 0L
    private var lastAccount = 0L
    private var waits = 0
    private var waitedMs = 0L
    private var running = false

    /** Aktuelle Wärmestufe (0 keine bis 6 Abschaltung). */
    var status = 0
        private set

    /** Beim Start der Live-Schleife: Zählung von vorn. */
    fun reset() {
        seconds.fill(0.0)
        lastRead = 0L
        lastAccount = SystemClock.elapsedRealtime()
        waits = 0
        waitedMs = 0L
        status = 0
        running = true
    }

    /** Beim Ende der Live-Schleife: Die Zeit danach zählt nicht mehr mit. */
    fun finish() {
        if (!running) return
        account(SystemClock.elapsedRealtime())
        running = false
    }

    /** Mindestabstand zwischen zwei Durchgängen in ms (0 = keiner). */
    fun minIntervalMs(): Long {
        val now = SystemClock.elapsedRealtime()
        if (now - lastRead >= 2000L) {
            account(now)
            status = pm.currentThermalStatus.coerceIn(0, 6)
            lastRead = now
        }
        return PipelineMath.minIntervalMs(status)
    }

    /** Die KI hat [ms] gewartet, weil die Wärmeregelung bremste. */
    fun noteWait(ms: Long) {
        waits++
        waitedMs += ms
    }

    private fun account(now: Long) {
        if (lastAccount > 0L) seconds[status] += (now - lastAccount) / 1000.0
        lastAccount = now
    }

    fun toJson(): JSONObject {
        if (running) account(SystemClock.elapsedRealtime())
        val stages = JSONObject()
        for (i in seconds.indices) if (seconds[i] > 0.0) stages.put(DeviceHeat.stageNameAt(i), r1(seconds[i]))
        return JSONObject()
            .put("sekundenJeStufe", stages)
            .put("gebremsteDurchgaenge", waits)
            .put("gewartetS", r1(waitedMs / 1000.0))
            .put("regel", "ab stark höchstens 5/s, ab kritisch 3/s, ab Notfall 1/s")
    }
}
