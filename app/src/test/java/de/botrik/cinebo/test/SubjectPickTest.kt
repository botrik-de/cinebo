package de.botrik.cinebo.test

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/**
 * Wahl des ganzen Objekts beim Antippen ([SubjectPick]) und die Abstandstransformation dazu.
 * Die Szenen sind den Fällen aus der Offline-Prüfung mit EdgeTAM nachgebaut: Person mit Mütze,
 * Hund in der Wiese, Becher mit einem Streifen Hintergrund, Person hinter einem Tisch. Tiefe ist
 * inverse Tiefe (größer = näher); jede Szene hat einen Boden mit Verlauf, damit die Tiefenspanne
 * des Bildes (P95 − P5) wie bei echten Bildern ist.
 */
class SubjectPickTest {

    private val n = SubjectPick.N

    /** Maske als Logits: +8 in [inside], −8 sonst. */
    private fun logits(inside: (Int, Int) -> Boolean) = FloatArray(n * n) { k -> if (inside(k % n, k / n)) 8f else -8f }

    private fun rect(x0: Int, y0: Int, x1: Int, y1: Int): (Int, Int) -> Boolean = { x, y -> x in x0 until x1 && y in y0 until y1 }

    private fun stack(vararg masks: FloatArray): FloatArray = FloatArray(masks.size * n * n).also { out ->
        masks.forEachIndexed { i, m -> System.arraycopy(m, 0, out, i * n * n, n * n) }
    }

    /** Hintergrund weit weg (1), unten ein Boden von 1 bis 4 (näher), darüber [layers] (später gewinnt). */
    private fun scene(vararg layers: Pair<(Int, Int) -> Boolean, Float>): FloatArray = FloatArray(n * n) { k ->
        val x = k % n
        val y = k / n
        var v = if (y >= 200) 1f + 3f * (y - 200) / 55f else 1f
        for ((where, depth) in layers) if (where(x, y)) v = depth
        v
    }

    private fun u(x: Int) = (x + 0.5f) / n

    @Test
    fun abstandstransformationWieRohesNachrechnen() {
        val size = 40
        val rnd = Random(7)
        repeat(5) { round ->
            val m = BooleanArray(size * size) { rnd.nextDouble() < if (round == 0) 0.002 else 0.08 }
            for (feature in listOf(true, false)) {
                val got = FloatArray(size * size)
                Edt.squared(m, feature, got, size)
                for (k in 0 until size * size) {
                    var best = Float.MAX_VALUE
                    for (j in 0 until size * size) {
                        if (m[j] != feature) continue
                        val dx = (k % size - j % size).toFloat()
                        val dy = (k / size - j / size).toFloat()
                        best = minOf(best, dx * dx + dy * dy)
                    }
                    if (best == Float.MAX_VALUE) assertTrue("ohne Ziel unendlich", got[k] > 1e19f)
                    else assertEquals("Zelle $k, Runde $round, Ziel $feature", best, got[k], 0f)
                }
            }
        }
    }

    @Test
    fun personStattMuetzeUndErneutTippenFuerTeile() {
        val person = rect(80, 40, 176, 256)
        val grid = scene(person to 2.5f)
        val hat = logits(rect(100, 40, 156, 70))
        val bit = logits(rect(120, 50, 136, 60))
        // Die IoU-Schätzung mag die Mütze am liebsten, wie bei Meta.
        val ious = floatArrayOf(0.09f, 0.98f, 0.31f)
        val c = SubjectPick.evaluate(stack(logits(person), hat, bit), 3, ious, grid, u(128), u(55))
        assertEquals(1, SubjectPick.bestIou(c))
        assertTrue(c.all { it.plausible })
        assertEquals("die Person, nicht die Mütze", 0, SubjectPick.choose(c))
        // Gleich noch einmal getippt: der nächstkleinere Teil, dann das kleine Stück, dann wieder alles.
        assertEquals(1, SubjectPick.choose(c, 1))
        assertEquals(2, SubjectPick.choose(c, 2))
        assertEquals(0, SubjectPick.choose(c, 3))
    }

    @Test
    fun wieseUmDenHundEnthaeltDenHundNicht() {
        val dog = rect(60, 90, 196, 190)
        val lawnArea = rect(0, 60, 256, 200)
        val grid = scene(lawnArea to 1.5f, dog to 3f)
        val lawn = logits { x, y -> lawnArea(x, y) && !dog(x, y) }
        val c = SubjectPick.evaluate(stack(lawn, logits(dog), logits(rect(120, 110, 124, 114))), 3, floatArrayOf(0.02f, 0.07f, 0.01f), grid, u(122), u(112))
        assertTrue("die Wiese enthält den Hund nicht", c[0].nested < 0.6)
        assertFalse(c[0].plausible)
        assertEquals(1, SubjectPick.choose(c))
    }

    @Test
    fun becherMitStreifenHintergrundFliegtRaus() {
        // Becher nah (4) vor einer Person (2); zwischen beiden schaut die Wand (1) durch. Eine Maske
        // hat den Streifen Wand dabei: in sich nah und fern gemischt, innen am Rand weiter weg als außen.
        val cup = rect(110, 120, 150, 190)
        val strip = rect(80, 90, 110, 190)
        val grid = scene(rect(40, 0, 220, 200) to 2f, strip to 1f, cup to 4f)
        val withStrip = logits { x, y -> cup(x, y) || strip(x, y) }
        val c = SubjectPick.evaluate(stack(withStrip, logits(cup), logits(rect(120, 130, 130, 140))), 3, floatArrayOf(0.34f, 0.84f, 0.22f), grid, u(130), u(150))
        assertTrue("große Tiefenspanne", c[0].spread > 0.3)
        assertTrue("innen am Rand weiter weg als außen", c[0].contrast < 0.0)
        assertFalse(c[0].plausible)
        assertTrue("der Becher allein ist glaubwürdig", c[1].plausible)
        assertEquals(1, SubjectPick.choose(c))
    }

    @Test
    fun personHinterTischBleibtGlaubwuerdig() {
        // Nahaufnahme: Person (2) über die ganze Bildbreite hinter einem Tisch (3,5), der ihren
        // unteren Teil verdeckt. Ihre Umgebung unten ist näher, sie selbst aber eben. Sie darf
        // nicht auf ihr Hemd zurückfallen.
        val person = rect(0, 20, 256, 150)
        val table = rect(0, 150, 256, 256)
        val grid = scene(person to 2f, table to 3.5f)
        val shirt = logits(rect(60, 60, 200, 140))
        val c = SubjectPick.evaluate(stack(logits(person), shirt), 2, floatArrayOf(0.3f, 0.9f), grid, u(128), u(100))
        assertTrue("Außenring zum großen Teil näher", c[0].behind > 0.3)
        assertTrue("in sich eben", c[0].spread <= 0.3)
        assertTrue(c[0].plausible)
        assertEquals(0, SubjectPick.choose(c))
    }

    @Test
    fun zuGrossIstHintergrund() {
        val grid = scene()
        val huge = logits { _, y -> y < 230 }
        val obj = logits(rect(100, 230, 140, 250))
        val c = SubjectPick.evaluate(stack(huge, obj), 2, floatArrayOf(0.5f, 0.4f), grid, u(120), u(240))
        assertTrue(c[0].area > 0.85)
        assertFalse(c[0].plausible)
        assertEquals(1, SubjectPick.choose(c))
    }

    @Test
    fun maskeFernVomTippIstNichtGemeint() {
        val grid = scene(rect(20, 20, 80, 80) to 3f, rect(150, 150, 190, 180) to 3f)
        val far = logits(rect(20, 20, 80, 80))
        val near = logits(rect(150, 150, 190, 180))
        val c = SubjectPick.evaluate(stack(far, near), 2, floatArrayOf(0.9f, 0.5f), grid, u(170), u(170))
        assertTrue(c[0].tapDistance > 4.0)
        assertEquals(0.0, c[1].tapDistance, 0.0)
        assertEquals(1, SubjectPick.choose(c))
    }

    @Test
    fun keineGlaubwuerdigeHeisstWieMeta() {
        // Beide Masken weit weg vom Tipp: Rückfall auf die beste Schätzung, auch beim zweiten Tipp.
        val grid = scene()
        val a = logits(rect(10, 10, 40, 40))
        val b = logits(rect(10, 10, 30, 30))
        val c = SubjectPick.evaluate(stack(a, b), 2, floatArrayOf(0.2f, 0.6f), grid, u(200), u(200))
        assertFalse(c[0].plausible)
        assertFalse(c[1].plausible)
        assertEquals(1, SubjectPick.choose(c))
        assertEquals(1, SubjectPick.choose(c, 1))
    }

    @Test
    fun winzigeMaskeIstNichtGlaubwuerdig() {
        val c = SubjectPick.evaluate(stack(logits(rect(10, 10, 11, 13))), 1, floatArrayOf(0.9f), scene(), u(10), u(11))
        assertEquals(-1.0, c[0].contrast, 0.0)
        assertFalse(c[0].plausible)
        assertEquals(0, SubjectPick.choose(c))
    }

    @Test
    fun tiefeAufsMaskenrasterWieEingepasst() {
        // Tiefenbild 10 × 10, Inhalt in der mittleren Hälfte (links 0,25 bis rechts 0,75).
        val size = 10
        val depth = FloatArray(size * size) { k -> (k % size) + 100f * (k / size) }
        val grid = SubjectPick.depthGrid(depth, size, 0.25f, 0f, 0.75f, 1f)
        // Zelle 0 (links oben) liegt bei x = 0,25 + 0,5/256·0,5 → Spalte 2, Zeile 0.
        assertEquals(2f, grid[0], 0f)
        // Letzte Zelle rechts unten: x ≈ 0,749 → Spalte 7, y ≈ 0,998 → Zeile 9.
        assertEquals(907f, grid[n * n - 1], 0f)
        // Mitte: Spalte 5, Zeile 5.
        assertEquals(505f, grid[(n / 2) * n + n / 2], 0f)
    }

    @Test
    fun tiefeMitNanOderUnendlichTaugtNicht() {
        val grid = scene()
        assertTrue(SubjectPick.depthUsable(grid))
        assertFalse(SubjectPick.depthUsable(grid.copyOf().also { it[100] = Float.NaN }))
        assertFalse(SubjectPick.depthUsable(grid.copyOf().also { it[7] = Float.POSITIVE_INFINITY }))
    }

    @Test
    fun medianUndPerzentilWieNumpy() {
        val g = floatArrayOf(5f, 1f, 4f, 2f)
        assertEquals(3.0, SubjectPick.median(g) { true }, 0.0)
        // Index 0 und 2: Werte 5 und 4, Median 4,5.
        assertEquals(4.5, SubjectPick.median(g) { it != 1 && it != 3 }, 0.0)
        assertTrue(SubjectPick.median(g) { false }.isNaN())
        // numpy.percentile([1, 2, 4, 5], 75) = 4,25; 5 % = 1,15.
        assertEquals(4.25, SubjectPick.percentile(g.copyOf(), 0.75, presorted = false), 1e-9)
        assertEquals(1.15, SubjectPick.percentile(g.copyOf(), 0.05, presorted = false), 1e-6)
    }
}
