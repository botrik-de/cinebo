package de.botrik.cinebo.test

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

/** Hebt den Bericht auf, damit nach einem Absturz oder Neustart nichts verloren geht. */
internal class ReportStore(context: Context) {

    private val file = File(context.filesDir, "bericht.json")

    fun load(): JSONObject = try {
        if (file.exists()) JSONObject(file.readText()) else JSONObject()
    } catch (_: Exception) {
        JSONObject()
    }

    fun save(report: JSONObject) {
        try {
            val tmp = File(file.path + ".neu")
            tmp.writeText(report.toString())
            if (!tmp.renameTo(file)) {
                file.writeText(report.toString())
                tmp.delete()
            }
        } catch (_: Exception) {
        }
    }

    fun clear() {
        file.delete()
    }
}

/** Schreibt einen Absturz in eine Datei; beim nächsten Start landet er im Bericht. */
internal object CrashLog {

    private const val NAME = "absturz.txt"
    @Volatile private var installed = false

    fun install(context: Context) {
        if (installed) return
        installed = true
        val file = File(context.applicationContext.filesDir, NAME)
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            try {
                file.writeText("${nowIso()}\nThread: ${thread.name}\n${Log.getStackTraceString(error)}")
            } catch (_: Throwable) {
            }
            previous?.uncaughtException(thread, error)
        }
    }

    /** Liefert den letzten Absturz einmalig und löscht die Datei. */
    fun take(context: Context): String? {
        val file = File(context.filesDir, NAME)
        if (!file.exists()) return null
        val text = try {
            file.readText()
        } catch (_: Exception) {
            null
        }
        file.delete()
        return text?.take(12000)
    }
}
