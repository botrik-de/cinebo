package de.botrik.cinebo.test

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.floor

/** Rechnungen der Grafik-Pipeline: Drehungen, Einpassen, Packen, Maskenwahl, Takt und Wärme. */
class PipelineMathTest {

    /** Puffer W × H mit eindeutigen Werten; Zeile 0 = oberste Zeile. */
    private fun buffer(w: Int, h: Int) = Array(h) { y -> IntArray(w) { x -> y * 1000 + x } }

    /** Dreht ein Bild um 90° im Uhrzeigersinn. */
    private fun rotateCw(img: Array<IntArray>): Array<IntArray> {
        val h = img.size
        val w = img[0].size
        return Array(w) { y -> IntArray(h) { x -> img[h - 1 - x][y] } }
    }

    @Test
    fun aufrechtNachPufferStimmtFuerAlleDrehungen() {
        val w = 8
        val h = 6
        val buf = buffer(w, h)
        for (rotation in listOf(0, 90, 180, 270)) {
            var upright = buf
            repeat(rotation / 90) { upright = rotateCw(upright) }
            val uh = upright.size
            val uw = upright[0].size
            assertEquals(PipelineMath.uprightSize(w, h, rotation), uw to uh)
            val m = PipelineMath.uprightToBuffer(rotation)
            for (j in 0 until uh) {
                for (i in 0 until uw) {
                    val (s, t) = PipelineMath.apply(m, (i + 0.5f) / uw, (j + 0.5f) / uh)
                    val x = floor(s * w).toInt()
                    val yTop = floor((1f - t) * h).toInt()
                    assertEquals("Drehung $rotation, Pixel $i/$j", upright[j][i], buf[yTop][x])
                }
            }
        }
    }

    // ------------------------------------------------------------------ Matrix der SurfaceTexture

    /** Spaltenweise 4 × 4 aus den Spalten (wie mat4 in AOSP). */
    private fun cols(vararg v: Float) = v

    private val flipH = cols(-1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 1f, 0f, 0f, 1f)
    private val flipV = cols(1f, 0f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 1f, 0f, 1f)
    private val rot90 = cols(0f, 1f, 0f, 0f, -1f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 1f, 0f, 0f, 1f)

    /**
     * Nachbau von GLConsumer::computeTransformMatrix (AOSP): Drehung und Spiegelung des Puffers
     * ([flags] aus H, V, R wie NATIVE_WINDOW_TRANSFORM_*), Zuschnitt mit Schrumpfen um einen
     * Texel (YUV) und zuletzt die senkrechte Spiegelung für OpenGL.
     */
    private fun stFor(flags: String, bufW: Int = 8, bufH: Int = 6, cropH: Int = bufH, shrink: Float = 1f): FloatArray {
        var x = PipelineMath.identity()
        if ('H' in flags) x = PipelineMath.multiply(x, flipH)
        if ('V' in flags) x = PipelineMath.multiply(x, flipV)
        if ('R' in flags) x = PipelineMath.multiply(x, rot90)
        if (cropH < bufH) {
            val ty = (bufH - cropH + shrink) / bufH
            val sy = (cropH - 2 * shrink) / bufH
            val crop = cols(1f, 0f, 0f, 0f, 0f, sy, 0f, 0f, 0f, 0f, 1f, 0f, 0f, ty, 0f, 1f)
            x = PipelineMath.multiply(crop, x)
        }
        return PipelineMath.multiply(flipV, x)
    }

    @Test
    fun drehungInDerMatrixDerSurfaceTexture() {
        assertEquals(0, PipelineMath.stRotation(stFor("")))
        assertEquals(90, PipelineMath.stRotation(stFor("R")))
        assertEquals(180, PipelineMath.stRotation(stFor("HV")))
        assertEquals(270, PipelineMath.stRotation(stFor("HVR")))
        for (flags in listOf("", "R", "HV", "HVR")) {
            assertFalse(flags, PipelineMath.stMirrored(stFor(flags)))
            // Ein Puffer mit Füllzeilen (3840 × 2176, Bild 2160 hoch) ändert nichts daran.
            val padded = stFor(flags, 3840, 2176, cropH = 2160)
            assertEquals(flags, PipelineMath.stRotation(stFor(flags)), PipelineMath.stRotation(padded))
        }
        assertTrue(PipelineMath.stMirrored(stFor("H")))
        assertEquals(6 to 8, PipelineMath.stSize(8, 6, stFor("R")))
        assertEquals(8 to 6, PipelineMath.stSize(8, 6, stFor("")))
    }

    /**
     * Der ganze Weg vom gespeicherten Puffer zum aufrechten KI-Bild, wie auf dem Gerät: Die
     * Pipeline tastet die Textur mit st · aufrecht→st-Raum ab. Erwartet wird der Puffer, um
     * rotationDegrees (CameraX) im Uhrzeigersinn gedreht, ohne doppelte Drehung.
     */
    @Test
    fun kiBildStehtAufrechtWieCameraXEsVerlangt() {
        val w = 8
        val h = 6
        val stored = buffer(w, h) // Zeile 0 = erste Speicherzeile = oben
        // (Sensor-Drehung in st, rotationDegrees von CameraX): hochkant gehalten und quer gehalten.
        val cases = listOf(
            "R" to 90, "R" to 0, "R" to 180, "R" to 270,
            "HVR" to 270, "HVR" to 0, "" to 90, "" to 0, "HV" to 180
        )
        for ((flags, rotationDegrees) in cases) {
            val st = stFor(flags, w, h)
            var expected = stored
            repeat(rotationDegrees / 90) { expected = rotateCw(expected) }
            val r = PipelineMath.uprightFromSt(rotationDegrees, st)
            val (sw, sh) = PipelineMath.stSize(w, h, st)
            val (uw, uh) = PipelineMath.uprightSize(sw, sh, r)
            assertEquals("$flags/$rotationDegrees: Breite", expected[0].size, uw)
            assertEquals("$flags/$rotationDegrees: Höhe", expected.size, uh)
            val m = PipelineMath.multiply(st, PipelineMath.uprightToBuffer(r))
            for (j in 0 until uh) {
                for (i in 0 until uw) {
                    val (s, t) = PipelineMath.apply(m, (i + 0.5f) / uw, (j + 0.5f) / uh)
                    val x = floor(s * w).toInt()
                    val y = floor(t * h).toInt()
                    assertEquals("$flags/$rotationDegrees, Pixel $i/$j", expected[j][i], stored[y][x])
                }
            }
        }
    }

    /**
     * Ausgabe und Maske: Für jeden Punkt der Ausgabe muss die Maske an der Stelle des aufrechten
     * Bildes nachsehen, die denselben Pufferpunkt zeigt. Die Ausgabe von StreamSharing liegt
     * quer wie der Puffer (updated = senkrechte Spiegelung), die der reinen Vorschau aufrecht.
     */
    @Test
    fun maskeSitztAufDemAusgegebenenBild() {
        val st = stFor("R", 8, 6)
        val r = PipelineMath.uprightFromSt(90, st)
        val toTex = PipelineMath.multiply(st, PipelineMath.uprightToBuffer(r))
        val bufToMask = PipelineMath.bufferToUpright(r)
        val stInverse = PipelineMath.invert(st)!!
        // StreamSharing: Ausgabe quer wie der Puffer; reine Vorschau: Ausgabe aufrecht wie der st-Raum.
        val outputs = listOf(flipV, st)
        for (updated in outputs) {
            val outToBuf = PipelineMath.multiply(stInverse, updated)
            for ((u, v) in listOf(0.1f to 0.2f, 0.5f to 0.5f, 0.9f to 0.3f, 0.25f to 0.75f)) {
                val (bx, by) = PipelineMath.apply(outToBuf, u, v)
                val (mu, mv) = PipelineMath.apply(bufToMask, bx, by)
                // Derselbe Texturpunkt, einmal über die Ausgabe, einmal über die Maske.
                val viaOutput = PipelineMath.apply(updated, u, v)
                val viaMask = PipelineMath.apply(toTex, mu, mv)
                assertEquals(viaOutput.first, viaMask.first, 1e-5f)
                assertEquals(viaOutput.second, viaMask.second, 1e-5f)
                assertTrue(mu in 0f..1f && mv in 0f..1f)
            }
        }
    }

    @Test
    fun pufferNachAufrechtIstDieUmkehrung() {
        for (rotation in listOf(0, 90, 180, 270)) {
            val back = PipelineMath.multiply(PipelineMath.bufferToUpright(rotation), PipelineMath.uprightToBuffer(rotation))
            assertArrayEquals(PipelineMath.identity(), back, 1e-6f)
        }
    }

    @Test
    fun umkehrungUndMultiplikation() {
        // Typische Matrix einer Kamera-SurfaceTexture: Spiegelung in t, dazu ein Ausschnitt.
        val st = floatArrayOf(0.98f, 0f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 0f, 1f, 0f, 0.01f, 1f, 0f, 1f)
        val inv = assertNotNullAndGet(PipelineMath.invert(st))
        assertArrayEquals(PipelineMath.identity(), PipelineMath.multiply(st, inv), 1e-6f)
        assertArrayEquals(PipelineMath.identity(), PipelineMath.multiply(inv, st), 1e-6f)
        val (x, y) = PipelineMath.apply(st, 0.5f, 0.25f)
        assertEquals(0.5f, x, 1e-6f)
        assertEquals(0.75f, y, 1e-6f)
        assertNull(PipelineMath.invert(FloatArray(16)))
    }

    private fun assertNotNullAndGet(m: FloatArray?): FloatArray {
        assertNotNull(m)
        return m!!
    }

    @Test
    fun einpassenWieInStufe2() {
        // Hochkant 2160 × 3840 in 518 × 518: Ränder links und rechts.
        val r = PipelineMath.letterbox(2160, 3840, 518)
        assertArrayEquals(floatArrayOf(0.21875f, 0f, 0.78125f, 1f), r, 1e-6f)
        val q = PipelineMath.letterbox(3840, 2160, 518)
        assertArrayEquals(floatArrayOf(0f, 0.21875f, 1f, 0.78125f), q, 1e-6f)
    }

    @Test
    fun packenPasstGenauInDenModelleingang() {
        val enc = PipelineMath.PackLayout(1024, 768)
        assertEquals(1024, enc.height)
        assertEquals(1024, enc.fullRows)
        assertEquals(0, enc.rest)
        assertEquals(1024 * 1024 * 3 * 4, enc.bytes)
        val depth = PipelineMath.PackLayout(518, 1024)
        assertEquals(197, depth.height)
        assertEquals(196, depth.fullRows)
        assertEquals(539, depth.rest)
        assertEquals(518 * 518 * 3 * 4, depth.bytes)
    }

    @Test
    fun gaussGewichteErgebenEins() {
        val (w, o) = PipelineMath.gaussTaps(3.5)
        assertEquals(1.0, (w[0] + 2 * (w[1] + w[2] + w[3])).toDouble(), 1e-6)
        assertEquals(0f, o[0], 0f)
        assertTrue(o[1] in 1f..2f && o[2] in 3f..4f && o[3] in 5f..6f)
    }

    @Test
    fun maskenWahl() {
        val ts = longArrayOf(100, 233, 366, 0)
        // Zwischen zwei Masken: Mischung nach der Zeit.
        val p = PipelineMath.pickMasks(ts, 3, 300, 400)!!
        assertEquals(1, p.a)
        assertEquals(2, p.b)
        assertEquals((300 - 233) / 133f, p.mix, 1e-6f)
        assertEquals(66L, p.distanceNs)
        // Genau auf einer Maske.
        val q = PipelineMath.pickMasks(ts, 3, 233, 400)!!
        assertEquals(1, q.a)
        assertEquals(0f, q.mix, 0f)
        assertEquals(0L, q.distanceNs)
        // Nur ältere Masken.
        val r = PipelineMath.pickMasks(ts, 3, 500, 400)!!
        assertEquals(2, r.a)
        assertEquals(2, r.b)
        assertEquals(134L, r.distanceNs)
        // Zu weit weg: keine Maske.
        assertNull(PipelineMath.pickMasks(ts, 3, 900, 400))
        // Nur neuere Masken.
        val s = PipelineMath.pickMasks(ts, 3, 50, 400)!!
        assertEquals(0, s.a)
        assertEquals(50L, s.distanceNs)
        assertNull(PipelineMath.pickMasks(ts, 0, 300, 400))
    }

    @Test
    fun taktDerKiBilder() {
        val ms = 1_000_000L
        // Die KI wartet: sofort vorbereiten.
        assertTrue(PipelineMath.shouldPrepare(0, 0, waiting = true, leadMs = 45.0))
        // Takt unbekannt und keine wartende KI: nicht vorbereiten.
        assertFalse(PipelineMath.shouldPrepare(10 * ms, 0, waiting = false, leadMs = 45.0))
        // Die KI nimmt in 50 ms das nächste Bild: noch nicht; in 44 ms: jetzt.
        assertFalse(PipelineMath.shouldPrepare(1000 * ms, 1050 * ms, waiting = false, leadMs = 45.0))
        assertTrue(PipelineMath.shouldPrepare(1006 * ms, 1050 * ms, waiting = false, leadMs = 45.0))
        // Schon überfällig: vorbereiten.
        assertTrue(PipelineMath.shouldPrepare(1100 * ms, 1050 * ms, waiting = false, leadMs = 45.0))
    }

    @Test
    fun waermeregelung() {
        assertEquals(0L, PipelineMath.minIntervalMs(0))
        assertEquals(0L, PipelineMath.minIntervalMs(2))
        assertEquals(167L, PipelineMath.minIntervalMs(3))
        assertEquals(200L, PipelineMath.minIntervalMs(4))
        assertEquals(333L, PipelineMath.minIntervalMs(5))
        assertEquals(333L, PipelineMath.minIntervalMs(6))
        // Zuerst fällt die Tiefe weg, die Verfolgung bleibt bis „stark“ über 5,7 pro Sekunde.
        assertEquals(1, PipelineMath.depthEvery(2))
        assertEquals(4, PipelineMath.depthEvery(3))
        assertEquals(0, PipelineMath.depthEvery(4))
        assertTrue(1000.0 / PipelineMath.minIntervalMs(3) >= 5.7)
    }

    @Test
    fun unschaerfeKlingtMitAlterDerMaskeAus() {
        assertEquals(1f, PipelineMath.maskFade(0L), 0f)
        assertEquals(1f, PipelineMath.maskFade(400_000_000L), 0f)
        assertEquals(0.5f, PipelineMath.maskFade(950_000_000L), 1e-6f)
        assertEquals(0f, PipelineMath.maskFade(1_500_000_000L), 0f)
        assertEquals(0f, PipelineMath.maskFade(Long.MAX_VALUE), 0f)
    }

    @Test
    fun maskeAlsWahrscheinlichkeit() {
        val logits = floatArrayOf(-1024f, -16f, -2f, 0f, 2f, 16f, 1024f, Float.NaN)
        val out = ByteArray(logits.size)
        PipelineMath.maskProbabilities(logits, out)
        val p = out.map { it.toInt() and 0xFF }
        assertEquals(0, p[0])
        assertEquals(0, p[1])
        assertTrue(p[2] in 25..34)          // Sigmoid(−2) ≈ 0,12
        assertTrue(p[3] in 127..135)        // Sigmoid(0) = 0,5 (Achtelschritt nach oben)
        assertTrue(p[4] in 220..230)        // Sigmoid(2) ≈ 0,88
        assertEquals(255, p[5])
        assertEquals(255, p[6])
        assertEquals(0, p[7])               // NaN zählt als kein Objekt
    }

    @Test
    fun abgriffeBeimVerkleinern() {
        assertEquals(1, PipelineMath.taps(1.0))
        assertEquals(3, PipelineMath.taps(2.11))
        assertEquals(4, PipelineMath.taps(3.75))
        assertEquals(8, PipelineMath.taps(7.4))
        assertEquals(8, PipelineMath.taps(20.0))
    }
}
