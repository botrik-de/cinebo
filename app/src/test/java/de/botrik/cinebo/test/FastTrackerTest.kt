package de.botrik.cinebo.test

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Wo die Schnellverfolgung im nächsten Bild hintippt: die Mitte der Maske, immer auf dem Objekt. */
class FastTrackerTest {

    private val n = 8

    private fun mask(cells: List<Pair<Int, Int>>) = FloatArray(n * n) { -1f }.also { m ->
        for ((x, y) in cells) m[y * n + x] = 1f
    }

    @Test
    fun leereMaskeHatKeineMitte() {
        assertNull(maskCenter(FloatArray(n * n) { -1f }, n))
    }

    @Test
    fun quadratLiefertSeineMitte() {
        // Schwerpunkt (2,5 | 4,5); alle vier Felder gleich weit weg, das erste zählt.
        val c = maskCenter(mask(listOf(2 to 4, 3 to 4, 2 to 5, 3 to 5)), n)!!
        assertEquals(2.5f / n, c[0], 1e-6f)
        assertEquals(4.5f / n, c[1], 1e-6f)
    }

    @Test
    fun beiEinemBogenLiegtDerPunktAufDerMaske() {
        // Ein „C“: Der Schwerpunkt (2 | 3,5) liegt in der Lücke. Genommen wird das nächste
        // Feld der Maske, (1 | 3), nicht der Schwerpunkt selbst.
        val c = (1..4).map { it to 1 } + (2..5).map { 1 to it } + (1..4).map { it to 6 }
        val center = maskCenter(mask(c), n)!!
        assertEquals(1.5f / n, center[0], 1e-6f)
        assertEquals(3.5f / n, center[1], 1e-6f)
    }
}
