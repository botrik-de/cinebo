package de.botrik.cinebo.test

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicLong

/** Hebt den Bericht auf, damit nach einem Absturz oder Neustart nichts verloren geht. */
internal class ReportStore(context: Context, name: String = "bericht.json") {

    private val file = File(context.filesDir, name)

    fun load(): JSONObject = try {
        if (file.exists()) JSONObject(file.readText()) else JSONObject()
    } catch (_: Exception) {
        JSONObject()
    }

    private val lock = Any()
    private val counter = AtomicLong()
    private var written = 0L

    /** Nummer für den nächsten Stand; ein Stand mit kleinerer Nummer überschreibt nie einen neueren. */
    fun nextSeq(): Long = counter.incrementAndGet()

    fun save(report: JSONObject) = saveText(report.toString(), nextSeq())

    /** Schreibt [text] als Stand [seq] (von jedem Thread; nacheinander, nie rückwärts). */
    fun saveText(text: String, seq: Long) {
        synchronized(lock) {
            if (seq < written) return
            try {
                val tmp = File(file.path + ".neu")
                tmp.writeText(text)
                if (!tmp.renameTo(file)) {
                    file.writeText(text)
                    tmp.delete()
                }
                written = seq
            } catch (_: Exception) {
            }
        }
    }

    /**
     * Legt [report] als eigene Datei neben dem Bericht ab, etwa den Bericht einer früheren Version:
     * [base].json, und gibt es die schon, [base]-2.json usw. Überschreibt nichts. Gibt den
     * Dateinamen zurück, oder null, wenn das Schreiben scheiterte.
     */
    fun archive(base: String, report: JSONObject): String? = try {
        val folder = file.parentFile ?: throw IllegalStateException("kein Ordner")
        var n = 1
        var target = File(folder, "$base.json")
        while (target.exists()) {
            n++
            target = File(folder, "$base-$n.json")
        }
        target.writeText(report.toString())
        target.name
    } catch (_: Exception) {
        null
    }

    /** Löscht den gespeicherten Bericht; ein noch ausstehender älterer Stand schreibt ihn nicht wieder hin. */
    fun clear() {
        synchronized(lock) {
            written = counter.incrementAndGet()
            file.delete()
        }
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
