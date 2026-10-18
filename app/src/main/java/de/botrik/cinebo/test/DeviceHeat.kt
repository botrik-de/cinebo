package de.botrik.cinebo.test

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
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

    /** Ab welchem Drosselindex das System welche Stufe meldet (ab Android 15). */
    fun thresholds(context: Context): JSONObject? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.VANILLA_ICE_CREAM) return null
        val pm = context.getSystemService(PowerManager::class.java)
        val o = JSONObject()
        for ((status, value) in pm.thermalHeadroomThresholds) o.put(stageName(status), r2(value.toDouble()))
        return o
    }
}
