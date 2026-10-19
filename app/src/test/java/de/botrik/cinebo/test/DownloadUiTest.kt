package de.botrik.cinebo.test

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Was die Oberfläche zeigt, solange die KI-Modelle fehlen; läuft bei jedem Bau ohne Handy. */
class DownloadUiTest {

    private val total = 254_000_000L
    private val net = ModelStore.Failure(ModelStore.Kind.NETWORK, "Zeitüberschreitung, 25 Versuche ohne Fortschritt")
    private val disk = ModelStore.Failure(ModelStore.Kind.STORAGE, "zu wenig Speicher frei")
    private val build = ModelStore.Failure(ModelStore.Kind.BUILD, "x.gz fehlt auf dem Server")

    private fun state(
        downloading: Boolean,
        waiting: String? = null,
        failure: ModelStore.Failure? = null,
        done: Long = 0,
        hasList: Boolean = true,
    ) = DownloadUi.State(hasList, downloading, waiting, failure, done, total)

    @Test
    fun beimHerunterladenIstDerKnopfGesperrt() {
        val s = state(downloading = true)
        assertEquals("KI-Modelle werden heruntergeladen …", DownloadUi.buttonText(s))
        assertFalse(DownloadUi.buttonEnabled(s))
        assertTrue(DownloadUi.hint(s).contains("zusammen 254 MB"))
    }

    @Test
    fun beimWartenAufsWlanLaesstSichDasDatenvolumenFreigeben() {
        val s = state(downloading = true, waiting = DownloadUi.WAIT_WIFI, done = 54_000_000)
        assertEquals("Trotzdem jetzt herunterladen", DownloadUi.buttonText(s))
        assertTrue(DownloadUi.buttonEnabled(s))
        assertTrue(DownloadUi.hint(s).contains("noch 200 MB"))
    }

    @Test
    fun beimWartenAufsInternetPasstDerKnopfZurStatuszeile() {
        val s = state(downloading = true, waiting = DownloadUi.WAIT_INTERNET)
        assertEquals("Wartet auf Internet …", DownloadUi.buttonText(s))
        assertFalse(DownloadUi.buttonEnabled(s))
    }

    @Test
    fun nachEinemNetzfehlerIstDerKnopfFrei() {
        // Der Fehler aus der ersten Prüfung: Danach blieb der Knopf gesperrt.
        val s = state(downloading = false, failure = net, done = 1)
        assertTrue(DownloadUi.buttonEnabled(s))
        assertEquals("Weiter herunterladen", DownloadUi.buttonText(s))
        assertTrue(DownloadUi.hint(s).contains("noch einmal zu versuchen"))
        assertFalse(DownloadUi.hint(s).contains("an derselben Stelle"))
        assertEquals("Herunterladen unterbrochen", DownloadUi.idleLine(s, false, false, true)?.text)
    }

    @Test
    fun jederFehlerBekommtDenPassendenRat() {
        assertTrue(DownloadUi.hint(state(downloading = false, failure = disk)).contains("Gib Speicher frei"))
        assertTrue(DownloadUi.hint(state(downloading = false, failure = build)).contains("Fehler beim Bauen"))
    }

    @Test
    fun ohneListeIstAllesGesperrt() {
        val s = state(downloading = false, failure = build, hasList = false)
        assertFalse(DownloadUi.buttonEnabled(s))
        assertEquals("KI-Modelle: Liste fehlt oder ist ungültig", DownloadUi.idleLine(s, false, false, true)?.text)
    }

    @Test
    fun statuszeileOhneTest() {
        assertNull(DownloadUi.idleLine(state(downloading = true), false, false, true))
        assertEquals("Bereit", DownloadUi.idleLine(state(downloading = false), true, false, true)?.text)
        assertEquals("Kamera nicht bereit", DownloadUi.idleLine(state(downloading = false), true, true, true)?.text)
        assertEquals("Kamera startet …", DownloadUi.idleLine(state(downloading = false), true, false, false)?.text)
        assertEquals("KI-Modelle fehlen", DownloadUi.idleLine(state(downloading = false), false, false, true)?.text)
    }

    @Test
    fun verspaetetesWartenNachDerFreigabeZaehltNicht() {
        assertNull(DownloadUi.waitingFrom(ModelStore.Retry(0, DownloadUi.WAIT_WIFI), allowMetered = true))
        assertEquals(DownloadUi.WAIT_WIFI, DownloadUi.waitingFrom(ModelStore.Retry(0, DownloadUi.WAIT_WIFI), allowMetered = false))
        assertEquals(DownloadUi.WAIT_INTERNET, DownloadUi.waitingFrom(ModelStore.Retry(0, DownloadUi.WAIT_INTERNET), allowMetered = true))
        assertNull(DownloadUi.waitingFrom(ModelStore.Retry(4, "Server meldet 503"), allowMetered = false))
        assertNull(DownloadUi.waitingFrom(ModelStore.Retry(0, DownloadUi.CONNECTING), allowMetered = false))
        assertEquals("Verbindet …", DownloadUi.progressLine(1, total, ModelStore.Retry(0, DownloadUi.WAIT_WIFI), null, null).text)
    }

    @Test
    fun statuszeileBeimHerunterladen() {
        val loading = DownloadUi.progressLine(100_000_000, total, null, null, 95)
        assertEquals("Lädt die KI-Modelle herunter · 100 von 254 MB", loading.text)
        assertEquals("noch 2 min", loading.time)
        val retry = DownloadUi.progressLine(100_000_000, total, ModelStore.Retry(4, "Server meldet 503"), null, null)
        assertEquals("Server meldet 503 · neuer Versuch", retry.text)
        assertEquals("in 4 s", retry.time)
        val wifi = DownloadUi.progressLine(100_000_000, total, ModelStore.Retry(0, DownloadUi.WAIT_WIFI), DownloadUi.WAIT_WIFI, null)
        assertEquals("Wartet auf WLAN", wifi.text)
        assertEquals("noch 154 MB", wifi.time)
        assertEquals(DownloadUi.Tone.IDLE, wifi.tone)
    }

    @Test
    fun restzeitWirdRuhigGerundet() {
        assertEquals("noch 2 min", DownloadUi.remainingText(95))
        assertEquals("noch 45 s", DownloadUi.remainingText(42))
        assertEquals("gleich fertig", DownloadUi.remainingText(5))
    }
}
