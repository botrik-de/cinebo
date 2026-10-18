package de.botrik.cinebo.test

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.ConnectException
import java.net.HttpURLConnection
import java.net.ProtocolException
import java.net.SocketException
import java.net.SocketTimeoutException
import java.net.URL
import java.net.UnknownHostException
import java.security.MessageDigest
import java.util.zip.GZIPInputStream
import java.util.zip.ZipFile
import javax.net.ssl.SSLException
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Die KI-Modelle und die große QNN-Laufzeit stecken nicht in der App: Sie werden einmal geladen
 * und bleiben dann auf dem Handy. Welche Dateien dazugehören, steht in assets/modelle.json (beim
 * Bauen erzeugt). Jede fertige Datei kommt aus einer Quelle: einzeln aus unserem Release, meist
 * gepackt (gzip), oder aus einem Archiv, etwa Qualcomms QNN-Paket direkt von Maven Central (die
 * Lizenz erlaubt nicht, die Bibliotheken einzeln weiterzugeben). Bricht die Verbindung ab, geht
 * es an derselben Stelle weiter. Jede Quelle und jede fertige Datei wird Byte für Byte geprüft;
 * erst dann gilt eine Datei als fertig.
 */
internal class ModelStore(private val context: Context) {

    /** Was über das Netz kommt: eine einzelne, vielleicht gepackte Datei oder ein Archiv mit mehreren. */
    class Source(
        val name: String,
        val url: String,
        val bytes: Long,
        val sha256: String,
        val gzip: Boolean,
        val archive: Boolean,
    )

    /** Eine fertige Datei auf dem Handy; aus einem Archiv kommt sie von [pathInArchive]. */
    class Entry(
        val name: String,
        val bytes: Long,
        val sha256: String,
        val source: Source,
        val pathInArchive: String?,
    )

    /** Ein Versuch ist gescheitert; der nächste kommt in [inS] Sekunden. */
    class Retry(val inS: Int, val reason: String)

    /** Stand des Ladens in Bytes der Übertragung; [retry] ist gesetzt, solange auf einen neuen Versuch gewartet wird. */
    fun interface Progress {
        fun update(done: Long, total: Long, retry: Retry?)
    }

    val dir: File = File(context.filesDir, FOLDER).apply { mkdirs() }

    private val list: JSONObject? by lazy {
        try {
            JSONObject(context.assets.open(LIST).bufferedReader().use { it.readText() })
        } catch (_: Exception) {
            null
        }
    }

    /** Kennung des Satzes, z. B. für den Bericht. */
    val version: String get() = list?.optString("version").orEmpty()

    /** Quellen und fertige Dateien; leer, wenn die Liste fehlt oder nicht stimmt. */
    private val parsed: Pair<List<Source>, List<Entry>> by lazy {
        try {
            val o = list ?: return@lazy emptyList<Source>() to emptyList()
            val sourceList = o.getJSONArray("quellen")
            val sources = List(sourceList.length()) { i ->
                val s = sourceList.getJSONObject(i)
                Source(
                    name = plainName(s.getString("name")),
                    url = s.getString("url"),
                    bytes = s.getLong("bytes"),
                    sha256 = s.getString("sha256"),
                    gzip = s.optBoolean("gzip", false),
                    archive = s.optBoolean("archiv", false),
                )
            }
            val byName = sources.associateBy { it.name }
            val fileList = o.getJSONArray("dateien")
            val files = List(fileList.length()) { i ->
                val f = fileList.getJSONObject(i)
                val source = byName[f.getString("quelle")] ?: throw IllegalStateException("Quelle fehlt")
                Entry(
                    name = plainName(f.getString("name")),
                    bytes = f.getLong("bytes"),
                    sha256 = f.getString("sha256"),
                    source = source,
                    pathInArchive = if (source.archive) f.getString("pfad") else null,
                )
            }
            require(sources.map { it.name }.distinct().size == sources.size) { "Quellen doppelt" }
            require(files.map { it.name }.distinct().size == files.size) { "Dateien doppelt" }
            // Eine einzelne Quelle gehört zu genau einer Datei; ungenutzte Quellen gibt es nicht.
            for (s in sources) {
                val n = files.count { it.source === s }
                require(if (s.archive) n >= 1 else n == 1) { "Quelle ${s.name} passt nicht" }
            }
            sources to files
        } catch (_: Exception) {
            emptyList<Source>() to emptyList()
        }
    }

    val sources: List<Source> get() = parsed.first

    val entries: List<Entry> get() = parsed.second

    /** Nur schlichte Dateinamen: Die Liste darf nichts außerhalb des Ordners anfassen. */
    private fun plainName(name: String): String {
        require(name.isNotEmpty() && !name.contains('/') && !name.startsWith(".")) { "Ungültiger Name: $name" }
        return name
    }

    /** Ohne Liste lässt sich nichts laden; das wäre ein Fehler beim Bauen. */
    val hasList: Boolean get() = entries.isNotEmpty()

    fun file(name: String): File = File(dir, name)

    /** Übertragene Daten einer Quelle, auch halbe; die Prüfsumme im Namen hält Reste eines anderen Satzes fern. */
    private fun part(s: Source) = File(dir, "${s.name}.${s.sha256.take(12)}.part")

    /** Entpackt, aber noch nicht geprüft. */
    private fun unpacked(e: Entry) = File(dir, "${e.name}.neu")

    /** Enthält die Prüfsumme der fertigen Datei, sobald sie bestätigt ist. */
    private fun marker(e: Entry) = File(dir, "${e.name}.ok")

    private fun entriesOf(s: Source): List<Entry> = entries.filter { it.source === s }

    /** Fertig heißt: Datei da, Größe stimmt und die Prüfsumme wurde nach dem Laden bestätigt. */
    fun ready(e: Entry): Boolean {
        if (file(e.name).length() != e.bytes) return false
        return try {
            marker(e).readText().trim() == e.sha256
        } catch (_: IOException) {
            false
        }
    }

    private fun sourceDone(s: Source): Boolean = entriesOf(s).all { ready(it) }

    fun allReady(): Boolean = hasList && entries.all { ready(it) }

    /** Was über das Netz kommt, zusammen. */
    fun totalBytes(): Long = sources.sumOf { it.bytes }

    /** Was davon schon auf dem Handy ist, halb geladene Quellen mitgezählt. */
    fun doneBytes(): Long = sources.sumOf { if (sourceDone(it)) it.bytes else min(part(it).length(), it.bytes) }

    /** Platz, den die fertigen Dateien auf dem Handy brauchen. */
    fun storageBytes(): Long = entries.sumOf { it.bytes }

    /**
     * Räumt auf: Dateien, die nicht zur Liste gehören (etwa aus einem früheren Satz), halb
     * entpackte Reste und die Übertragungsdaten fertiger Quellen. Läuft gerade ein Download,
     * bleibt alles, wie es ist.
     */
    fun cleanup() {
        if (!hasList || !LOCK.tryLock()) return
        try {
            val keep = HashSet<String>()
            for (e in entries) {
                keep += e.name
                keep += marker(e).name
            }
            for (s in sources) if (!sourceDone(s)) keep += part(s).name
            dir.listFiles()?.forEach { if (it.name !in keep) it.deleteRecursively() }
        } finally {
            LOCK.unlock()
        }
    }

    fun describe(): JSONObject = JSONObject()
        .put("version", version)
        .put("dateien", entries.size)
        .put("fertig", entries.count { ready(it) })
        .put("mbUebertragung", r1(totalBytes() / 1e6))
        .put("mbAufDemHandy", r1(storageBytes() / 1e6))

    /**
     * Lädt alle fehlenden Dateien und meldet den Stand an [progress]. Läuft schon ein Download
     * (etwa von einer früheren Ansicht), wartet dieser hier, bis jener fertig ist. Wirft erst,
     * wenn es nach vielen Versuchen nicht weitergeht oder etwas grundsätzlich nicht passt.
     */
    suspend fun download(progress: Progress) = withContext(Dispatchers.IO) {
        LOCK.withLock {
            check(hasList) { "die Liste der KI-Modelle fehlt in der App" }
            val total = totalBytes()
            checkSpace()
            progress.update(doneBytes(), total, null)
            // Archive zuerst: Sie kommen von fremden Servern; klemmt dort etwas, zeigt es sich früh.
            for (s in sources.sortedBy { !it.archive }) {
                // Liegt eine fertige Datei schon richtig da (etwa nach einem Absturz kurz vor dem
                // Bestätigen), gilt sie ohne neues Laden.
                val missing = entriesOf(s).filter { !ready(it) && !adopt(it) }
                if (missing.isEmpty()) {
                    part(s).delete()
                    continue
                }
                var damaged = 0
                while (true) {
                    fetch(s) { retry -> progress.update(doneBytes(), total, retry) }
                    val p = part(s)
                    if (sha256(p) == s.sha256) {
                        install(s, p, missing)
                        break
                    }
                    // Beschädigt angekommen: einmal ganz neu laden, dann aufgeben.
                    p.delete()
                    if (++damaged >= 2) throw LoadException("${s.name} kam zweimal beschädigt an")
                }
                progress.update(doneBytes(), total, null)
            }
        }
    }

    private fun adopt(e: Entry): Boolean {
        val f = file(e.name)
        if (f.length() != e.bytes || sha256(f) != e.sha256) return false
        marker(e).writeText(e.sha256)
        return true
    }

    /**
     * Macht aus der geprüften Übertragung die fertigen Dateien: aus einem Archiv entpackt, aus
     * gzip entpackt oder einfach umbenannt. Entpacktes wird noch einmal geprüft.
     */
    private fun install(s: Source, p: File, missing: List<Entry>) {
        when {
            s.archive -> ZipFile(p).use { zip ->
                for (e in missing) {
                    val item = zip.getEntry(e.pathInArchive) ?: throw IllegalStateException("${e.name} fehlt in ${s.name}")
                    unpack(e, zip.getInputStream(item))
                }
            }
            s.gzip -> unpack(missing.single(), GZIPInputStream(FileInputStream(p), BUFFER))
            else -> {
                val e = missing.single()
                FileOutputStream(p, true).use { it.fd.sync() }
                replace(p, file(e.name))
                marker(e).writeText(e.sha256)
            }
        }
        p.delete()
    }

    /** Schreibt [input] in die fertige Datei, prüft dabei Größe und Prüfsumme und bestätigt sie. */
    private fun unpack(e: Entry, input: InputStream) {
        val tmp = unpacked(e)
        val md = MessageDigest.getInstance("SHA-256")
        var n = 0L
        input.use { source ->
            FileOutputStream(tmp).use { out ->
                val buffer = ByteArray(BUFFER)
                while (true) {
                    val r = source.read(buffer)
                    if (r < 0) break
                    md.update(buffer, 0, r)
                    out.write(buffer, 0, r)
                    n += r
                }
                out.fd.sync()
            }
        }
        if (n != e.bytes || hex(md.digest()) != e.sha256) {
            tmp.delete()
            throw IllegalStateException("${e.name} passt entpackt nicht zur Liste")
        }
        replace(tmp, file(e.name))
        marker(e).writeText(e.sha256)
    }

    private fun replace(from: File, to: File) {
        to.delete()
        if (!from.renameTo(to)) throw LoadException("${to.name} ließ sich nicht ablegen")
    }

    /**
     * Bricht früh ab, wenn der Speicher nicht reicht, statt mittendrin. Gebraucht werden die
     * fertigen Dateien und kurz die größte gepackte Übertragung, solange sie entpackt wird.
     */
    private fun checkSpace() {
        var need = SPACE_RESERVE
        var largestPacked = 0L
        for (s in sources) {
            val missing = entriesOf(s).filter { !ready(it) }
            if (missing.isEmpty()) continue
            val have = part(s).length()
            need += missing.sumOf { it.bytes }
            if (s.gzip || s.archive) largestPacked = max(largestPacked, s.bytes) else need -= min(have, s.bytes)
        }
        need += largestPacked
        val free = dir.usableSpace
        if (free in 1 until need) {
            throw IllegalStateException("zu wenig Speicher frei: nötig ${mb(need)} MB, frei ${mb(free)} MB")
        }
    }

    /** Lädt eine Quelle vollständig in ihre .part-Datei; setzt nach Abbrüchen fort. */
    private suspend fun fetch(s: Source, report: (Retry?) -> Unit) {
        val p = part(s)
        if (p.length() > s.bytes) p.delete()
        if (s.bytes == 0L) {
            p.createNewFile() // leere Datei: nichts zu übertragen
            return
        }
        var failures = 0
        var restarts = 0
        var lastReport = 0L
        while (p.length() < s.bytes) {
            currentCoroutineContext().ensureActive()
            val have = p.length()
            var received = 0L
            try {
                val conn = open(s.url, have)
                try {
                    val code = conn.responseCode
                    // Fehlt die Datei auf dem Server, hilft Warten nicht.
                    if (code == HttpURLConnection.HTTP_NOT_FOUND) throw IllegalStateException("${s.name} fehlt auf dem Server")
                    if (code == HTTP_RANGE_NOT_SATISFIABLE) {
                        // Den angefragten Bereich gibt es nicht: von vorn, aber nicht endlos.
                        if (++restarts > MAX_RESTARTS) throw IllegalStateException("${s.name} passt auf dem Server nicht zur Liste")
                        p.delete()
                        continue
                    }
                    if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL) {
                        throw LoadException("Server meldet $code")
                    }
                    // Ohne Teilbereich schickt der Server die ganze Datei; dann von vorn schreiben.
                    val append = code == HttpURLConnection.HTTP_PARTIAL
                    checkResponse(s, conn, append, have)
                    conn.inputStream.use { input ->
                        FileOutputStream(p, append).use { out ->
                            val buffer = ByteArray(BUFFER)
                            while (true) {
                                val n = input.read(buffer)
                                if (n < 0) break
                                out.write(buffer, 0, n)
                                received += n
                                val now = System.nanoTime() / 1_000_000
                                if (now - lastReport >= REPORT_MS) {
                                    lastReport = now
                                    report(null)
                                }
                                currentCoroutineContext().ensureActive()
                            }
                        }
                    }
                } finally {
                    conn.disconnect()
                }
            } catch (io: IOException) {
                // Wer Daten bekommen hat, kam voran: Dann zählen nur die Fehler seitdem.
                if (received > 0) failures = 0
                val why = reason(io)
                if (++failures > MAX_FAILURES) throw LoadException("$why, $MAX_FAILURES Versuche ohne Fortschritt", io)
                pause(failures, why, report)
                continue
            }
            when {
                p.length() > s.bytes -> {
                    // Mehr bekommen als erwartet: verwerfen und neu anfangen, aber nicht endlos.
                    p.delete()
                    if (++restarts > MAX_RESTARTS) throw IllegalStateException("${s.name} ist auf dem Server größer als laut Liste")
                }
                received == 0L -> {
                    // Antwort ohne Daten: wie ein Fehler behandeln, damit die Schleife nicht leer dreht.
                    if (++failures > MAX_FAILURES) throw LoadException("Server liefert keine Daten, $MAX_FAILURES Versuche ohne Fortschritt")
                    pause(failures, "Server liefert keine Daten", report)
                }
                else -> failures = 0
            }
        }
        report(null)
    }

    /** Prüft, ob der Server dieselbe Datei meint wie die Liste und an der richtigen Stelle weitermacht. */
    private fun checkResponse(s: Source, conn: HttpURLConnection, partial: Boolean, have: Long) {
        val size = if (partial) {
            val m = CONTENT_RANGE.find(conn.getHeaderField("Content-Range").orEmpty())
                ?: throw LoadException("Teilantwort ohne Bereichsangabe")
            val start = m.groupValues[1].toLong()
            if (start != have) throw LoadException("Teilantwort ab Byte $start statt $have")
            m.groupValues[3].toLongOrNull() ?: -1L
        } else {
            conn.contentLengthLong
        }
        if (size >= 0 && size != s.bytes) {
            throw IllegalStateException("${s.name} ist auf dem Server $size Bytes groß, laut Liste ${s.bytes}")
        }
    }

    /** Wartet vor dem nächsten Versuch, immer länger bis höchstens 30 s, und zählt dabei sichtbar herunter. */
    private suspend fun pause(failures: Int, why: String, report: (Retry?) -> Unit) {
        var left = min(MAX_PAUSE_MS, 1000L shl min(failures, 5))
        while (left > 0) {
            report(Retry(((left + 999) / 1000).toInt(), why))
            val step = min(1000L, left)
            delay(step)
            left -= step
        }
    }

    /** Folgt Weiterleitungen selbst, damit der Teilbereich auch beim eigentlichen Speicher ankommt. */
    private fun open(url: String, from: Long): HttpURLConnection {
        var target = URL(url)
        repeat(MAX_REDIRECTS) {
            val c = target.openConnection() as HttpURLConnection
            try {
                c.instanceFollowRedirects = false
                c.useCaches = false
                c.connectTimeout = CONNECT_TIMEOUT_MS
                c.readTimeout = READ_TIMEOUT_MS
                c.setRequestProperty("User-Agent", "CineBo-Test")
                c.setRequestProperty("Accept-Encoding", "identity")
                if (from > 0) c.setRequestProperty("Range", "bytes=$from-")
                val code = c.responseCode
                if (code !in 300..399) return c
                val location = c.getHeaderField("Location") ?: throw LoadException("Weiterleitung ohne Ziel")
                target = URL(target, location)
            } catch (io: IOException) {
                c.disconnect()
                throw io
            }
            c.disconnect()
        }
        throw LoadException("zu viele Weiterleitungen")
    }

    /** Fehler mit einem Grund, der schon in Worten dasteht. */
    private class LoadException(message: String, cause: Throwable? = null) : IOException(message, cause)

    /** Der Grund in Worten, die auch ohne Technik verständlich sind. */
    private fun reason(io: IOException): String = when (io) {
        is LoadException -> io.message.orEmpty()
        is UnknownHostException -> "keine Internetverbindung"
        is SocketTimeoutException -> "Zeitüberschreitung"
        is ConnectException -> "Server nicht erreichbar"
        is SSLException -> "sichere Verbindung gescheitert"
        is SocketException, is ProtocolException, is EOFException -> "Verbindung abgebrochen"
        else -> "Verbindung gestört"
    }

    private fun sha256(file: File): String {
        if (!file.exists()) return ""
        val md = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(BUFFER)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                md.update(buffer, 0, n)
            }
        }
        return hex(md.digest())
    }

    private fun hex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }

    private fun mb(bytes: Long): String = de(bytes / 1e6, 0)

    companion object {
        const val FOLDER = "modelle"
        const val LIST = "modelle.json"
        private const val BUFFER = 256 * 1024
        private const val MAX_FAILURES = 25
        private const val MAX_RESTARTS = 2
        private const val MAX_REDIRECTS = 6
        private const val MAX_PAUSE_MS = 30_000L
        private const val REPORT_MS = 250L
        private const val CONNECT_TIMEOUT_MS = 20_000
        private const val READ_TIMEOUT_MS = 30_000
        private const val HTTP_RANGE_NOT_SATISFIABLE = 416
        private const val SPACE_RESERVE = 64L * 1024 * 1024
        private val CONTENT_RANGE = Regex("""bytes\s+(\d+)-(\d+)/(\d+|\*)""", RegexOption.IGNORE_CASE)

        /** Nur ein Download zur selben Zeit, auch wenn die Ansicht neu entsteht. */
        private val LOCK = Mutex()
    }
}

/**
 * Schätzt aus dem Fortschritt, wie lange das Laden noch dauert. Mittelt gleitend über einige
 * Sekunden, damit kurze Aussetzer die Anzeige nicht springen lassen.
 */
internal class RemainingTime {
    private var lastMs = 0L
    private var lastBytes = 0L
    private var rate = 0.0 // Bytes je ms
    private var measuredMs = 0L

    fun reset() {
        lastMs = 0L
        rate = 0.0
        measuredMs = 0L
    }

    /** Nach einer Pause neu ansetzen, ohne die Pause als langsames Laden zu zählen. */
    fun pause() {
        lastMs = 0L
    }

    /** Neuer Stand; gibt die geschätzten Sekunden bis zum Ende zurück oder null, solange es dafür zu früh ist. */
    fun update(nowMs: Long, done: Long, total: Long): Long? {
        if (lastMs == 0L || done < lastBytes) {
            lastMs = nowMs
            lastBytes = done
            return estimate(done, total)
        }
        val dt = nowMs - lastMs
        if (dt >= MIN_STEP_MS) {
            val now = (done - lastBytes).toDouble() / dt
            rate = if (measuredMs == 0L) now else rate + (1 - exp(-dt / MEMORY_MS)) * (now - rate)
            measuredMs += dt
            lastMs = nowMs
            lastBytes = done
        }
        return estimate(done, total)
    }

    private fun estimate(done: Long, total: Long): Long? {
        if (measuredMs < SETTLE_MS || rate <= 0.0) return null
        return ((total - done) / rate / 1000.0).toLong()
    }

    private companion object {
        const val MIN_STEP_MS = 200L
        const val MEMORY_MS = 5000.0
        const val SETTLE_MS = 3000L
    }
}
