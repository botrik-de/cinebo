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
 * Die KI-Modelle und die große QNN-Laufzeit stecken nicht in der App: Sie werden einmal
 * heruntergeladen und bleiben dann auf dem Handy. Welche Dateien dazugehören, steht in
 * assets/modelle.json (beim Bauen erzeugt). Jede fertige Datei kommt aus einer Quelle: einzeln
 * aus unserem Release, meist gepackt (gzip), oder aus einem Archiv, etwa Qualcomms QNN-Paket
 * direkt von Maven Central (die Lizenz erlaubt nicht, die Bibliotheken einzeln weiterzugeben).
 * Bricht die Verbindung ab, geht es an derselben Stelle weiter. Jede Quelle und jede fertige
 * Datei wird Byte für Byte geprüft; erst dann gilt eine Datei als fertig.
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

    /**
     * Pause im Laden: [inS] Sekunden bis zum nächsten Versuch nach einem Fehler, oder 0, solange
     * auf etwas gewartet wird (WLAN, Internet) oder neu verbunden wird. [reason] steht in Worten da.
     */
    class Retry(val inS: Int, val reason: String)

    /** Stand des Ladens in Bytes der Übertragung; [retry] ist gesetzt, solange es gerade nicht vorangeht. */
    fun interface Progress {
        fun update(done: Long, total: Long, retry: Retry?)
    }

    /** Was nach einem endgültigen Fehler hilft. */
    enum class Kind {
        /** Verbindung: später noch einmal versuchen, es geht an derselben Stelle weiter. */
        NETWORK,

        /** Speicher auf dem Handy: Platz schaffen, dann weiter. */
        STORAGE,

        /** Liste und Server passen nicht zusammen: ein Fehler beim Bauen. */
        BUILD,
    }

    /** Das Laden hat endgültig aufgehört; [message] nennt den Grund in Worten. */
    class Failure(val kind: Kind, message: String, cause: Throwable? = null) : Exception(message, cause)

    val dir: File = File(context.filesDir, FOLDER).apply { mkdirs() }

    /** Das letzte technische Problem beim Laden, für den Bericht. */
    @Volatile var lastProblem: String? = null
        private set

    private val list: JSONObject? by lazy {
        try {
            JSONObject(context.assets.open(LIST).bufferedReader().use { it.readText() })
        } catch (_: Exception) {
            null
        }
    }

    /** Kennung des Satzes, z. B. für den Bericht. */
    val version: String get() = list?.optString("version").orEmpty()

    /** Quellen und fertige Dateien; leer, wenn die Liste fehlt, ein anderes Format hat oder nicht stimmt. */
    private val parsed: Pair<List<Source>, List<Entry>> by lazy {
        try {
            val o = list ?: return@lazy emptyList<Source>() to emptyList()
            require(o.optInt("format", 0) == LIST_FORMAT) { "Listenformat passt nicht" }
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

    /** Ohne gültige Liste lässt sich nichts laden; das wäre ein Fehler beim Bauen. */
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
        .apply { lastProblem?.let { put("letztesProblem", it) } }

    /**
     * Lädt alle fehlenden Dateien und meldet den Stand an [progress]. Vor jeder Verbindung fragt es
     * [gate]: Kommt ein Grund zurück (etwa „wartet auf WLAN“), wartet es, ohne das als Fehler zu
     * zählen. Läuft schon ein Download (etwa von einer früheren Ansicht), wartet dieser hier, bis
     * jener fertig ist. Wirft [Failure], wenn es endgültig nicht weitergeht.
     */
    suspend fun download(gate: () -> String? = { null }, progress: Progress) = withContext(Dispatchers.IO) {
        LOCK.withLock {
            if (!hasList) throw Failure(Kind.BUILD, "die Liste der KI-Modelle fehlt oder ist ungültig")
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
                    fetch(s, gate) { retry -> progress.update(doneBytes(), total, retry) }
                    val p = part(s)
                    if (sha256(p) == s.sha256) {
                        install(s, p, missing)
                        break
                    }
                    // Anders angekommen als erwartet: einmal ganz neu laden, dann aufgeben.
                    p.delete()
                    lastProblem = "${s.name}: Prüfsumme der Übertragung falsch"
                    if (++damaged >= 2) throw Failure(Kind.BUILD, "${s.name} kam zweimal anders an als erwartet")
                }
                progress.update(doneBytes(), total, null)
            }
        }
    }

    private fun adopt(e: Entry): Boolean {
        val f = file(e.name)
        if (f.length() != e.bytes || sha256(f) != e.sha256) return false
        writeMarker(e)
        return true
    }

    private fun writeMarker(e: Entry) = storage { marker(e).writeText(e.sha256) }

    /**
     * Macht aus der geprüften Übertragung die fertigen Dateien: aus einem Archiv entpackt, aus
     * gzip entpackt oder einfach umbenannt. Entpacktes wird noch einmal geprüft.
     */
    private fun install(s: Source, p: File, missing: List<Entry>) {
        when {
            s.archive -> {
                val zip = try {
                    ZipFile(p)
                } catch (io: IOException) {
                    lastProblem = technical(io)
                    throw Failure(Kind.BUILD, "${s.name} lässt sich nicht öffnen", io)
                }
                zip.use {
                    for (e in missing) {
                        val item = zip.getEntry(e.pathInArchive) ?: throw Failure(Kind.BUILD, "${e.name} fehlt in ${s.name}")
                        unpack(e) { zip.getInputStream(item) }
                    }
                }
            }
            s.gzip -> storage("Lesen auf dem Handy gescheitert") { FileInputStream(p) }.use { raw ->
                unpack(missing.single()) { GZIPInputStream(raw, BUFFER) }
            }
            else -> {
                val e = missing.single()
                storage { FileOutputStream(p, true).use { it.fd.sync() } }
                replace(p, file(e.name))
                writeMarker(e)
            }
        }
        p.delete()
    }

    /** Schreibt, was [open] liefert, in die fertige Datei, prüft Größe und Prüfsumme und bestätigt sie. */
    private fun unpack(e: Entry, open: () -> InputStream) {
        val tmp = unpacked(e)
        val md = MessageDigest.getInstance("SHA-256")
        var n = 0L
        try {
            open().use { input ->
                storage { FileOutputStream(tmp) }.use { out ->
                    val buffer = ByteArray(BUFFER)
                    while (true) {
                        val r = input.read(buffer)
                        if (r < 0) break
                        md.update(buffer, 0, r)
                        storage { out.write(buffer, 0, r) }
                        n += r
                    }
                    storage { out.fd.sync() }
                }
            }
        } catch (io: IOException) {
            tmp.delete()
            lastProblem = technical(io)
            throw Failure(Kind.STORAGE, "${e.name} ließ sich nicht entpacken", io)
        } catch (f: Failure) {
            tmp.delete()
            throw f
        }
        if (n != e.bytes || hex(md.digest()) != e.sha256) {
            tmp.delete()
            throw Failure(Kind.BUILD, "${e.name} passt entpackt nicht zur Liste")
        }
        replace(tmp, file(e.name))
        writeMarker(e)
    }

    private fun replace(from: File, to: File) {
        to.delete()
        if (!from.renameTo(to)) throw Failure(Kind.STORAGE, "${to.name} ließ sich nicht ablegen")
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
            throw Failure(Kind.STORAGE, "zu wenig Speicher frei, nötig ${mb(need)} MB, frei ${mb(free)} MB")
        }
    }

    /**
     * Lädt eine Quelle vollständig in ihre .part-Datei. Bricht ein Versuch ab, endet er vorzeitig
     * oder kommt nichts, folgt eine Pause und der nächste Versuch an derselben Stelle. Als Fehler
     * zählt nur, was nicht vorankommt; nach [MAX_FAILURES] Fehlern in Folge ist Schluss.
     */
    private suspend fun fetch(s: Source, gate: () -> String?, report: (Retry?) -> Unit) {
        val p = part(s)
        if (p.length() > s.bytes) p.delete()
        if (s.bytes == 0L) {
            storage { p.createNewFile() } // leere Datei: nichts zu übertragen
            return
        }
        var failures = 0
        var restarts = 0
        var notFound = 0
        var lastReport = 0L
        var paused = false
        while (p.length() < s.bytes) {
            currentCoroutineContext().ensureActive()
            paused = waitForGate(gate, report) || paused
            if (paused) report(Retry(0, "verbindet …"))
            val have = p.length()
            var problem: String? = null
            try {
                val conn = open(s.url, have)
                try {
                    val code = conn.responseCode
                    if (code == HttpURLConnection.HTTP_NOT_FOUND) {
                        // Ein kurzer Aussetzer des Servers ist möglich; fehlt die Datei wirklich, hilft Warten nicht.
                        if (++notFound >= MAX_NOT_FOUND) throw Failure(Kind.BUILD, "${s.name} fehlt auf dem Server")
                        throw LoadException("Datei auf dem Server nicht gefunden", "HTTP 404")
                    }
                    if (code == HTTP_RANGE_NOT_SATISFIABLE) {
                        // Den angefragten Bereich gibt es nicht: von vorn, aber nicht endlos.
                        if (++restarts > MAX_RESTARTS) throw Failure(Kind.BUILD, "${s.name} passt auf dem Server nicht zur Liste")
                        p.delete()
                        continue
                    }
                    if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL) {
                        throw LoadException("Server meldet $code", "HTTP $code")
                    }
                    // Ohne Teilbereich schickt der Server die ganze Datei; dann von vorn, aber nicht endlos.
                    val append = code == HttpURLConnection.HTTP_PARTIAL
                    if (!append && have > 0 && ++restarts > MAX_RESTARTS) {
                        throw Failure(Kind.NETWORK, "Server kann nicht an derselben Stelle weitermachen")
                    }
                    checkResponse(s, conn, append, have)
                    conn.inputStream.use { input ->
                        storage { FileOutputStream(p, append) }.use { out ->
                            val buffer = ByteArray(BUFFER)
                            while (true) {
                                val n = input.read(buffer)
                                if (n < 0) break
                                storage { out.write(buffer, 0, n) }
                                val now = System.nanoTime() / 1_000_000
                                if (now - lastReport >= REPORT_MS) {
                                    lastReport = now
                                    paused = false
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
                problem = reason(io)
                lastProblem = technical(io)
            }
            val now = p.length()
            if (now >= s.bytes) {
                if (now == s.bytes) break
                // Mehr bekommen als erwartet: verwerfen und neu anfangen, aber nicht endlos.
                p.delete()
                if (++restarts > MAX_RESTARTS) throw Failure(Kind.BUILD, "${s.name} ist auf dem Server größer als laut Liste")
                continue
            }
            // Nicht bis zum Ende gekommen: Abbruch, vorzeitiges Ende oder gar keine Daten.
            if (now > have) failures = 0
            val why = problem ?: if (now > have) "Verbindung abgebrochen" else "Server liefert keine Daten"
            if (++failures > MAX_FAILURES) throw Failure(Kind.NETWORK, "$why, $MAX_FAILURES Versuche ohne Fortschritt")
            pause(failures, why, report)
            paused = true
        }
        report(null)
    }

    /** Wartet, solange [gate] einen Grund nennt; gibt zurück, ob gewartet wurde. */
    private suspend fun waitForGate(gate: () -> String?, report: (Retry?) -> Unit): Boolean {
        var waited = false
        while (true) {
            val why = gate() ?: return waited
            waited = true
            report(Retry(0, why))
            delay(GATE_POLL_MS)
        }
    }

    /** Prüft, ob der Server dieselbe Datei meint wie die Liste und an der richtigen Stelle weitermacht. */
    private fun checkResponse(s: Source, conn: HttpURLConnection, partial: Boolean, have: Long) {
        val size = if (partial) {
            val m = CONTENT_RANGE.find(conn.getHeaderField("Content-Range").orEmpty())
                ?: throw LoadException("Server antwortet unerwartet", "Teilantwort ohne Content-Range")
            val start = m.groupValues[1].toLong()
            if (start != have) throw LoadException("Server antwortet unerwartet", "Teilantwort ab Byte $start statt $have")
            m.groupValues[3].toLongOrNull() ?: -1L
        } else {
            conn.contentLengthLong
        }
        if (size >= 0 && size != s.bytes) {
            throw Failure(Kind.BUILD, "${s.name} ist auf dem Server $size Bytes groß, laut Liste ${s.bytes}")
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
                val location = c.getHeaderField("Location")
                    ?: throw LoadException("Server antwortet unerwartet", "Weiterleitung ohne Location")
                target = URL(target, location)
            } catch (io: IOException) {
                c.disconnect()
                throw io
            }
            c.disconnect()
        }
        throw LoadException("Server antwortet unerwartet", "mehr als $MAX_REDIRECTS Weiterleitungen")
    }

    /** Netzfehler mit einem Grund in Worten ([friendly]) und den Einzelheiten für den Bericht. */
    private class LoadException(val friendly: String, technical: String) : IOException(technical)

    /** Der Grund in Worten, die auch ohne Technik verständlich sind. */
    private fun reason(io: IOException): String = when (io) {
        is LoadException -> io.friendly
        is UnknownHostException -> "keine Internetverbindung"
        is SocketTimeoutException -> "Zeitüberschreitung"
        is ConnectException -> "Server nicht erreichbar"
        is SSLException -> "sichere Verbindung gescheitert"
        is SocketException, is ProtocolException, is EOFException -> "Verbindung abgebrochen"
        else -> "Verbindung gestört"
    }

    private fun technical(t: Throwable): String = "${t.javaClass.simpleName}: ${t.message}".take(300)

    /**
     * Führt Dateiarbeit auf dem Handy aus. Scheitert sie, ist das kein Netzfehler: Es gibt keine
     * Wiederholung, sondern gleich einen [Failure] mit Grund, bei vollem Speicher ausdrücklich.
     */
    private inline fun <T> storage(what: String = "Speichern auf dem Handy gescheitert", block: () -> T): T = try {
        block()
    } catch (io: IOException) {
        lastProblem = technical(io)
        throw Failure(Kind.STORAGE, if (isFull(io)) "zu wenig Speicher frei" else what, io)
    }

    private fun isFull(e: Throwable): Boolean = generateSequence(e) { it.cause }.any { t ->
        val m = t.message.orEmpty()
        "ENOSPC" in m || m.contains("No space left", ignoreCase = true)
    }

    private fun sha256(file: File): String {
        if (!file.exists()) return ""
        val md = MessageDigest.getInstance("SHA-256")
        storage("Lesen auf dem Handy gescheitert") {
            FileInputStream(file).use { input ->
                val buffer = ByteArray(BUFFER)
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    md.update(buffer, 0, n)
                }
            }
        }
        return hex(md.digest())
    }

    private fun hex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }

    private fun mb(bytes: Long): String = de(bytes / 1e6, 0)

    companion object {
        const val FOLDER = "modelle"
        const val LIST = "modelle.json"

        /** Muss zu LISTEN_FORMAT in .github/ki_dateien.py passen. */
        const val LIST_FORMAT = 1
        private const val BUFFER = 256 * 1024
        private const val MAX_FAILURES = 25
        private const val MAX_RESTARTS = 3
        private const val MAX_NOT_FOUND = 3
        private const val MAX_REDIRECTS = 6
        private const val MAX_PAUSE_MS = 30_000L
        private const val GATE_POLL_MS = 2_000L
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
 * Sekunden, damit kurze Aussetzer die Anzeige nicht springen lassen. Lücken ohne Meldung (Prüfen,
 * Entpacken, neu Verbinden) zählen nicht als langsames Laden.
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
        val dt = nowMs - lastMs
        if (lastMs == 0L || done < lastBytes || dt > MAX_GAP_MS) {
            lastMs = nowMs
            lastBytes = done
            return estimate(done, total)
        }
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
        const val MAX_GAP_MS = 2_000L
        const val MEMORY_MS = 5000.0
        const val SETTLE_MS = 3000L
    }
}
