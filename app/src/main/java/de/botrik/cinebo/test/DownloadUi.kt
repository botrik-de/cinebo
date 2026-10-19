package de.botrik.cinebo.test

/**
 * Was die Oberfläche zeigt, solange die KI-Modelle fehlen: Knopf, Hinweis und Statuszeile.
 * Ohne Android-Klassen, damit sich jeder Zustand auch ohne Handy prüfen lässt.
 */
internal object DownloadUi {

    /** Worauf das Herunterladen warten kann; so meldet es auch das Tor in [ModelStore.download]. */
    const val WAIT_WIFI = "wartet auf WLAN"
    const val WAIT_INTERNET = "wartet auf Internet"
    const val CONNECTING = "verbindet …"

    /** Ton der Statuszeile; die Activity macht daraus Farbe und Punkt. */
    enum class Tone { IDLE, BUSY, ERROR, DONE }

    /** Eine Statuszeile: Text links, Zeit rechts (oder nichts), Anteil für den Balken (oder keiner). */
    class Line(val tone: Tone, val text: String, val time: String? = null, val fraction: Float? = null)

    /** Alles, wovon Knopf und Hinweis abhängen. */
    class State(
        val hasList: Boolean,
        val downloading: Boolean,
        val waiting: String?,
        val failure: ModelStore.Failure?,
        val doneBytes: Long,
        val totalBytes: Long,
    )

    /**
     * Worauf eine Meldung des Downloads wartet, oder null. Wurde das Datenvolumen schon freigegeben,
     * ist eine verspätete Meldung „wartet auf WLAN“ überholt.
     */
    fun waitingFrom(retry: ModelStore.Retry?, allowMetered: Boolean): String? {
        val reason = retry?.takeIf { it.inS == 0 }?.reason ?: return null
        return when {
            reason == WAIT_WIFI && allowMetered -> null
            reason == WAIT_WIFI || reason == WAIT_INTERNET -> reason
            else -> null
        }
    }

    fun buttonText(s: State): String = when {
        s.downloading && s.waiting == WAIT_WIFI -> "Trotzdem jetzt herunterladen"
        s.downloading && s.waiting == WAIT_INTERNET -> "Wartet auf Internet …"
        s.downloading -> "KI-Modelle werden heruntergeladen …"
        s.doneBytes > 0 -> "Weiter herunterladen"
        else -> "KI-Modelle herunterladen"
    }

    /** Antippen geht, wenn ein Download starten oder das Warten aufs WLAN enden kann. */
    fun buttonEnabled(s: State): Boolean = s.hasList && (!s.downloading || s.waiting == WAIT_WIFI)

    fun hint(s: State): String {
        val left = mb(s.totalBytes - s.doneBytes)
        val failure = s.failure
        return when {
            !s.hasList -> "In dieser Version fehlt die Liste der KI-Modelle oder ist ungültig. Das ist ein Fehler beim " +
                "Bauen; bitte sag mir Bescheid."
            s.downloading && s.waiting == WAIT_WIFI -> "Die App lädt die KI-Modelle herunter, sobald du im WLAN bist; es fehlen " +
                "noch $left MB. Tippe auf „Trotzdem jetzt herunterladen“, um sie gleich über dein Datenvolumen zu laden."
            s.downloading && s.waiting == WAIT_INTERNET -> "Gerade ist keine Internetverbindung da. Die App lädt von selbst " +
                "weiter, sobald das Handy wieder online ist; es fehlen noch $left MB."
            s.downloading -> "Die App lädt die KI-Modelle herunter, zusammen ${mb(s.totalBytes)} MB. Lass sie dabei offen; " +
                "bricht die Verbindung ab, macht sie von selbst an derselben Stelle weiter. Danach bleiben die Modelle auf dem Handy."
            failure != null && failure.kind == ModelStore.Kind.BUILD -> "Die KI-Modelle passen nicht zu dieser Version " +
                "(${failure.message}). Das ist ein Fehler beim Bauen; bitte schick mir den Bericht."
            failure != null && failure.kind == ModelStore.Kind.STORAGE -> "Das Herunterladen ist unterbrochen: " +
                "${failure.message}. Gib Speicher frei und tippe dann auf „${buttonText(s)}“."
            failure != null -> "Das Herunterladen ist unterbrochen (${failure.message}). Tippe auf „${buttonText(s)}“, um es " +
                "noch einmal zu versuchen; was schon da ist, bleibt erhalten."
            else -> "Die App muss die KI-Modelle herunterladen, noch $left MB. Tippe auf „${buttonText(s)}“."
        }
    }

    /** Statuszeile zu einer Meldung des Downloads; [remainingS] ist die geschätzte Restzeit oder null. */
    fun progressLine(done: Long, total: Long, retry: ModelStore.Retry?, waiting: String?, remainingS: Long?): Line {
        val fraction = if (total > 0) (done.toDouble() / total).toFloat() else 0f
        return when {
            retry == null -> Line(
                Tone.BUSY, "Lädt die KI-Modelle herunter · ${mb(done)} von ${mb(total)} MB",
                remainingS?.let { remainingText(it) }, fraction
            )
            retry.inS > 0 -> Line(Tone.BUSY, "${capitalized(retry.reason)} · neuer Versuch", "in ${retry.inS} s", fraction)
            waiting != null -> Line(Tone.IDLE, capitalized(waiting), "noch ${mb(total - done)} MB", fraction)
            // Verspätetes „wartet auf WLAN“ nach der Freigabe oder ein Neuverbinden.
            retry.reason == WAIT_WIFI -> Line(Tone.BUSY, capitalized(CONNECTING), null, fraction)
            else -> Line(Tone.BUSY, capitalized(retry.reason), null, fraction)
        }
    }

    /** Statuszeile, wenn kein Test läuft; null, solange der Download seinen Fortschritt selbst zeigt. */
    fun idleLine(s: State, modelsOk: Boolean, cameraProblem: Boolean, cameraReady: Boolean): Line? = when {
        s.downloading -> null
        !modelsOk && !s.hasList -> Line(Tone.ERROR, "KI-Modelle: Liste fehlt oder ist ungültig")
        !modelsOk && s.failure != null -> Line(Tone.ERROR, "Herunterladen unterbrochen")
        !modelsOk -> Line(Tone.IDLE, "KI-Modelle fehlen")
        cameraProblem -> Line(Tone.ERROR, "Kamera nicht bereit")
        !cameraReady -> Line(Tone.IDLE, "Kamera startet …")
        else -> Line(Tone.IDLE, "Bereit")
    }

    /** Restzeit grob gerundet, damit die Anzeige ruhig bleibt. */
    fun remainingText(seconds: Long): String = when {
        seconds >= 90 -> "noch ${(seconds + 30) / 60} min"
        seconds >= 10 -> "noch ${(seconds + 4) / 5 * 5} s"
        else -> "gleich fertig"
    }

    fun mb(bytes: Long): String = de(bytes / 1e6, 0)

    private fun capitalized(text: String): String = text.replaceFirstChar { it.uppercase() }
}
