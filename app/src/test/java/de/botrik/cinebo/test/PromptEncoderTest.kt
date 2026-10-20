package de.botrik.cinebo.test

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/**
 * Die Eingabe-Einbettung der App gegen Werte aus Metas eigenem Code (SAM2, _embed_points).
 * Die Gewichte sind künstlich, damit der Test klein bleibt; die erwarteten Werte hat SAM2s
 * Prompt-Encoder mit genau diesen Gewichten gerechnet. Die echten Gewichte prüft die App
 * zusätzlich auf dem Handy gegen Proben aus dem Export (Bericht: „einbettung“).
 */
class PromptEncoderTest {

    private val d = PromptEncoder.D
    private val half = PromptEncoder.HALF

    private val encoder = PromptEncoder(
        gauss = Array(2) { r -> FloatArray(half) { j -> (1.7 * sin(0.37 * r + 0.113 * j + 0.2)).toFloat() } },
        labelEmbeddings = Array(4) { l -> FloatArray(d) { j -> (0.01 * (l + 1) * cos(0.05 * j * (l + 1))).toFloat() } },
        noPoint = FloatArray(d) { j -> (-0.5 + j / 256.0).toFloat() },
        imageSize = 1024,
    )

    /** Geprüfte Stellen je Einbettung (Sinus-Hälfte, Grenze, Kosinus-Hälfte). */
    private val idx = intArrayOf(0, 1, 63, 127, 128, 129, 200, 255)
    private val leer = floatArrayOf(-0.5f, -0.4960938f, -0.2539062f, -0.0039062f, 0.0f, 0.0039062f, 0.28125f, 0.4960938f)

    private fun embed(u: FloatArray, v: FloatArray, labels: IntArray): FloatArray =
        FloatArray((labels.size + 1) * d).also { encoder.embed(u, v, labels, it) }

    private fun assertToken(out: FloatArray, token: Int, expected: FloatArray) {
        val got = FloatArray(idx.size) { out[token * d + idx[it]] }
        assertArrayEquals("Einbettung $token", expected, got, 5e-5f)
    }

    @Test
    fun startklickWieImOriginal() {
        // So ruft die App den Decoder beim Startbild auf: Punkt, leerer Punkt, angehängter Leerpunkt.
        val out = embed(floatArrayOf(0.3f, 0f), floatArrayOf(0.6f, 0f), intArrayOf(1, -1))
        assertToken(out, 0, floatArrayOf(0.3266587f, 0.0623561f, -0.9797665f, -0.6687889f, 0.9712762f, 1.0179956f, -0.703723f, -0.706465f))
        assertToken(out, 1, leer)
        assertToken(out, 2, leer)
    }

    @Test
    fun rahmenWieImOriginal() {
        // Ein Rahmen als zwei Ecken (Labels 2 und 3), so wie SAM2 Rahmen eingibt.
        val out = embed(floatArrayOf(0.1f, 0.8f), floatArrayOf(0.2f, 0.95f), intArrayOf(2, 3))
        assertToken(out, 0, floatArrayOf(0.9364541f, -0.3440046f, -0.9154034f, 0.2482878f, 0.450481f, 0.9538838f, 0.7077858f, 1.0013117f))
        assertToken(out, 1, floatArrayOf(0.2243286f, 1.0199883f, 0.6784938f, 0.1649051f, 1.018577f, -0.1636675f, 0.8301972f, 1.0216751f))
        assertToken(out, 2, leer)
    }

    @Test
    fun negativerPunktUndBildrandWieImOriginal() {
        val out = embed(floatArrayOf(0.999f, 0f), floatArrayOf(0f, 1f), intArrayOf(0, 1))
        assertToken(out, 0, floatArrayOf(0.4867157f, 0.3126796f, -0.9849516f, 0.811102f, -0.8691257f, -0.9432272f, -0.7645537f, -0.5886661f))
        assertToken(out, 1, floatArrayOf(-0.4664947f, -0.2951682f, 0.9992323f, -0.7897885f, -0.8542269f, -0.9301718f, -0.7392356f, -0.5683015f))
        assertToken(out, 2, leer)
    }

    @Test
    fun folgebildBestehtNurAusLeerpunkten() {
        val out = embed(floatArrayOf(0.7f, 0.2f), floatArrayOf(0.4f, 0.9f), intArrayOf(-1, -1))
        for (t in 0..2) assertToken(out, t, leer)
    }

    @Test
    fun falscheEingabenWerdenAbgewiesen() {
        assertThrows(IllegalArgumentException::class.java) { embed(floatArrayOf(0.5f), floatArrayOf(0.5f), intArrayOf(4)) }
        assertThrows(IllegalArgumentException::class.java) {
            encoder.embed(floatArrayOf(0.5f), floatArrayOf(0.5f), intArrayOf(1), FloatArray(3 * d))
        }
        assertEquals(3 * d, embed(floatArrayOf(0.5f, 0f), floatArrayOf(0.5f, 0f), intArrayOf(1, -1)).size)
    }
}
