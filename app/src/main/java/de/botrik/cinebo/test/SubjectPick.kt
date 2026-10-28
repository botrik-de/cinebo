package de.botrik.cinebo.test

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Wählt beim Antippen unter den drei Masken des Decoders die, die das ganze Objekt trifft.
 *
 * Der Decoder schlägt zu einem Tipp drei Masken vor: meist das ganze Objekt, einen Teil und
 * ein kleines Stück. Metas Videoverfolgung nimmt die mit der besten IoU-Schätzung; bei einem
 * bunten Objekt ist das oft nur die angetippte Farbfläche (Mütze statt Person, Trikot statt
 * Radfahrer). Hier zählt die größte Maske, die nicht nach Hintergrund aussieht:
 * - höchstens 85 % des Bildes,
 * - sie enthält mindestens 60 % der nächstkleineren Maske (das Ganze enthält seine Teile; die
 *   Wiese um einen Hund enthält den Hund nicht),
 * - der Tipp liegt höchstens 4 Zellen von ihr entfernt,
 * - und sie mischt nicht Nahes mit Fernem: Ist die Tiefenspanne in der Maske groß (über 30 %
 *   der Spanne des Bildes), darf sie weder innen am Rand weiter weg sein als ihre Umgebung
 *   (Tiefenkontrast < 0, etwa ein Becher samt Streifen Hintergrund) noch zu mehr als 30 % ihres
 *   Außenrings deutlich dahinter liegen (etwa die Wiese um einen Hund). Eine Person hinter einem
 *   Tisch ist in sich eben und bleibt glaubwürdig.
 * „Deutlich“ und „Spanne“ messen sich an der Tiefenspanne des ganzen Bildes (P95 − P5), weil
 * Depth Anything Tiefe nur bis auf Maßstab und Verschiebung schätzt. Sieht keine Maske
 * glaubwürdig aus, gilt die beste IoU-Schätzung wie im Original. Tippt man gleich noch einmal
 * auf dieselbe Stelle, nimmt [choose] mit [choose]`(rank = 1)` die nächstkleinere glaubwürdige.
 *
 * Offline geprüft an 16 Tipps auf andersfarbige Teile in Hochkant-Ausschnitten der EdgeTAM-
 * Beispielvideos (Metas EdgeTAM, Tiefenmodell der App): mit der besten Schätzung 0,51 mittlere
 * Übereinstimmung mit dem ganzen Objekt, mit dieser Regel 0,76 = jeweils die beste der drei;
 * über 16 Folgebilder verfolgt 0,62 gegen 0,97. Mit 16 Gegenproben (Tipps auf Hintergrund,
 * Nachbarn in gleicher Tiefe, Nahaufnahmen, Personen hinter Tisch oder Geländer) zusammen
 * 0,52 gegen 0,71, in 30 von 32 Fällen die beste der drei.
 *
 * Alles im 256er-Raster der Masken (Zeile 0 oben); die Tiefe ist inverse Tiefe von Depth
 * Anything (größer = näher), auf dasselbe Raster abgetastet ([depthGrid]). Ränder und Ringe
 * kommen aus einer exakten euklidischen Abstandstransformation, wie die Scheiben im Python-Test.
 */
internal object SubjectPick {

    const val N = MASK_SIZE
    private const val MAX_AREA = 0.85
    private const val MIN_NESTED = 0.6
    private const val NEST_MIN_AREA = 0.002
    private const val MAX_TAP_DISTANCE = 4.0
    private const val CLOSER = 0.10
    private const val MAX_BEHIND = 0.3
    private const val MAX_SPREAD = 0.3

    /** Merkmale einer Maske für die Wahl und den Bericht. */
    data class Candidate(
        val area: Double,
        val iou: Float,
        /** (Median innen am Rand − Median Band außen) / Tiefenspanne des Bildes; −1 bei winziger Maske. */
        val contrast: Double,
        /** Anteil des Außenrings, der deutlich näher liegt als die Maske innen am Rand. */
        val behind: Double,
        /** Tiefenspanne in der Maske (P75 − P25) / Tiefenspanne des Bildes. */
        val spread: Double,
        /** Abstand des Tipps zur Maske in Zellen (0 = drin). */
        val tapDistance: Double,
        /** Anteil der nächstkleineren Maske, den diese enthält (1 ohne kleinere). */
        val nested: Double,
    ) {
        /**
         * Glaubwürdig: nicht zu groß, enthält die nächstkleinere Maske, liegt am Tipp, und mischt
         * nicht Nahes mit Fernem, das hinter seiner Umgebung liegt. Die Tiefe zählt nur bei großer
         * Spanne in der Maske: Eine Person hinter einem Tisch hat ihre nähere Umgebung vor sich
         * (Kontrast negativ, Außenring näher), ist in sich aber eben; ein Becher samt Streifen
         * Hintergrund dagegen nicht.
         */
        val plausible: Boolean
            get() = area <= MAX_AREA && nested >= MIN_NESTED && tapDistance <= MAX_TAP_DISTANCE &&
                !(spread > MAX_SPREAD && (behind > MAX_BEHIND || contrast < 0.0))
    }

    /**
     * Tiefe je Zelle des Maskenrasters: Mitte der Zelle im aufrechten Bild, in das eingepasste
     * Tiefenbild ([size]² Werte, Zeile 0 oben) übertragen, nächster Bildpunkt. Lage des Bildes im
     * Tiefenquadrat: [left], [top], [right], [bottom] (0 bis 1).
     */
    fun depthGrid(depth: FloatArray, size: Int, left: Float, top: Float, right: Float, bottom: Float, out: FloatArray = FloatArray(N * N)): FloatArray {
        require(depth.size >= size * size && out.size == N * N)
        val xs = IntArray(N) { ((left + (it + 0.5f) / N * (right - left)) * size).toInt().coerceIn(0, size - 1) }
        val ys = IntArray(N) { ((top + (it + 0.5f) / N * (bottom - top)) * size).toInt().coerceIn(0, size - 1) }
        for (y in 0 until N) {
            val row = ys[y] * size
            for (x in 0 until N) out[y * N + x] = depth[row + xs[x]]
        }
        return out
    }

    /**
     * Ob die Tiefe für die Wahl taugt: alle Werte endlich. NaN oder Unendlich (etwa ein Überlauf auf
     * dem KI-Chip) schalten sonst still die Tiefenprüfung ab; dann wählt die App wie Meta.
     */
    fun depthUsable(grid: FloatArray): Boolean = grid.all { it.isFinite() }

    /**
     * Merkmale aller Masken; [masks]: je Maske N² Logits (größer 0 heißt Objekt) hintereinander,
     * [grid]: Tiefe je Zelle, ([tapU], [tapV]): Tipp von 0 bis 1 im aufrechten Bild.
     */
    fun evaluate(masks: FloatArray, count: Int, ious: FloatArray, grid: FloatArray, tapU: Float, tapV: Float): List<Candidate> {
        require(masks.size >= count * N * N && grid.size == N * N && ious.size >= count)
        val sorted = grid.copyOf().also { it.sort() }
        val norm = max(percentile(sorted, 0.95, presorted = true) - percentile(sorted, 0.05, presorted = true), 1e-6)
        val sx = (tapU * N).toInt().coerceIn(0, N - 1)
        val sy = (tapV * N).toInt().coerceIn(0, N - 1)
        val bits = Array(count) { i -> BooleanArray(N * N) { k -> masks[i * N * N + k] > 0f } }
        val areas = IntArray(count) { i -> bits[i].count { it } }
        // Reihenfolge nach Fläche (bei Gleichstand der kleinere Index zuerst), für „die nächstkleinere“.
        val order = (0 until count).sortedWith(compareByDescending<Int> { areas[it] }.thenBy { it })
        val inside = FloatArray(N * N)
        val outside = FloatArray(N * N)
        return List(count) { i ->
            val m = bits[i]
            val area = areas[i] / (N * N).toDouble()
            if (areas[i] < 4) {
                Candidate(area, ious[i], -1.0, 1.0, 0.0, 999.0, 0.0)
            } else {
                // Quadrierte Abstände: innen zur nächsten Zelle außerhalb, außen zur nächsten Maskenzelle.
                Edt.squared(m, false, inside)
                Edt.squared(m, true, outside)
                val pos = order.indexOf(i)
                var nested = 1.0
                if (pos + 1 < count) {
                    val j = order[pos + 1]
                    if (areas[j] / (N * N).toDouble() > NEST_MIN_AREA) {
                        var both = 0
                        for (k in 0 until N * N) if (bits[j][k] && m[k]) both++
                        nested = both / maxOf(areas[j], 1).toDouble()
                    }
                }
                Candidate(
                    area = area,
                    iou = ious[i],
                    contrast = contrast(m, grid, inside, outside, norm),
                    behind = behind(m, grid, inside, outside, norm),
                    spread = spread(m, grid, norm),
                    tapDistance = sqrt(outside[sy * N + sx].toDouble()),
                    nested = nested,
                )
            }
        }
    }

    /**
     * Index der gewählten Maske: die größte glaubwürdige, beim [rank]-ten erneuten Tipp an derselben
     * Stelle die entsprechend kleinere (reihum); ohne glaubwürdige die beste IoU-Schätzung.
     */
    fun choose(candidates: List<Candidate>, rank: Int = 0): Int {
        val plausible = candidates.indices.filter { candidates[it].plausible }
            .sortedWith(compareByDescending<Int> { candidates[it].area }.thenBy { it })
        if (plausible.isNotEmpty()) return plausible[Math.floorMod(rank, plausible.size)]
        return bestIou(candidates)
    }

    fun bestIou(candidates: List<Candidate>): Int {
        var best = 0
        for (i in 1 until candidates.size) if (candidates[i].iou > candidates[best].iou) best = i
        return best
    }

    /**
     * (innen − außen) / Spanne: innen der Median im Randring der Maske (bis 4 Zellen vom Rand),
     * außen der Median im Band 2 bis 8 Zellen außerhalb. Positiv: Die Maske liegt vor ihrer Umgebung.
     */
    private fun contrast(m: BooleanArray, g: FloatArray, inside: FloatArray, outside: FloatArray, norm: Double): Double {
        var di = median(g) { m[it] && inside[it] <= 16f }
        if (di.isNaN()) di = median(g) { m[it] }
        val dout = median(g) { !m[it] && outside[it] > 4f && outside[it] <= 64f }
        if (dout.isNaN()) return 0.0
        return (di - dout) / norm
    }

    /** Anteil des Außenrings (bis 3 Zellen), der um mehr als 10 % der Spanne näher liegt als die Maske innen am Rand. */
    private fun behind(m: BooleanArray, g: FloatArray, inside: FloatArray, outside: FloatArray, norm: Double): Double {
        var di = median(g) { m[it] && inside[it] <= 9f }
        if (di.isNaN()) di = median(g) { m[it] }
        var ring = 0
        var closer = 0
        for (k in g.indices) {
            if (!m[k] && outside[k] <= 9f) {
                ring++
                if ((g[k] - di) / norm > CLOSER) closer++
            }
        }
        return if (ring == 0) 0.0 else closer / ring.toDouble()
    }

    /** (P75 − P25) der Tiefe in der Maske, geteilt durch die Spanne des Bildes. */
    private fun spread(m: BooleanArray, g: FloatArray, norm: Double): Double {
        val values = values(g) { m[it] }
        if (values.isEmpty()) return 0.0
        values.sort()
        return (percentile(values, 0.75, presorted = true) - percentile(values, 0.25, presorted = true)) / norm
    }

    /** Median der Werte, für die [take] gilt (bei gerader Anzahl Mittel der beiden mittleren, wie numpy); NaN ohne Werte. */
    internal inline fun median(g: FloatArray, take: (Int) -> Boolean): Double {
        val values = values(g, take)
        if (values.isEmpty()) return Double.NaN
        values.sort()
        val n = values.size
        return if (n % 2 == 1) values[n / 2].toDouble() else (values[n / 2 - 1].toDouble() + values[n / 2].toDouble()) / 2.0
    }

    internal inline fun values(g: FloatArray, take: (Int) -> Boolean): FloatArray {
        var n = 0
        for (k in g.indices) if (take(k)) n++
        val values = FloatArray(n)
        var j = 0
        for (k in g.indices) if (take(k)) values[j++] = g[k]
        return values
    }

    /** Perzentil wie numpy (lineare Interpolation zwischen den Nachbarn); [values] wird sortiert, falls nötig. */
    internal fun percentile(values: FloatArray, q: Double, presorted: Boolean): Double {
        require(values.isNotEmpty())
        if (!presorted) values.sort()
        val p = (values.size - 1) * q
        val lo = floor(p).toInt()
        val hi = minOf(lo + 1, values.size - 1)
        return values[lo] + (values[hi] - values[lo]) * (p - lo)
    }
}

/**
 * Exakte euklidische Abstandstransformation (Felzenszwalb und Huttenlocher) auf dem n × n-Raster:
 * quadrierter Abstand jeder Zelle zur nächsten Zelle, deren Wert [feature] ist. Zellen außerhalb
 * des Rasters zählen nicht. Gibt es keine solche Zelle, bleibt der Abstand unendlich (1e20).
 */
internal object Edt {
    private const val INF = 1e20f

    fun squared(m: BooleanArray, feature: Boolean, out: FloatArray, n: Int = SubjectPick.N) {
        require(m.size == n * n && out.size == n * n)
        for (k in 0 until n * n) out[k] = if (m[k] == feature) 0f else INF
        val f = FloatArray(n)
        val d = FloatArray(n)
        val v = IntArray(n)
        val z = FloatArray(n + 1)
        // Spalten, dann Zeilen.
        for (x in 0 until n) {
            for (y in 0 until n) f[y] = out[y * n + x]
            pass(f, d, v, z, n)
            for (y in 0 until n) out[y * n + x] = d[y]
        }
        for (y in 0 until n) {
            for (x in 0 until n) f[x] = out[y * n + x]
            pass(f, d, v, z, n)
            for (x in 0 until n) out[y * n + x] = d[x]
        }
    }

    /** Eindimensionale Transformation: untere Hülle der Parabeln (q − p)² + f[p]. */
    private fun pass(f: FloatArray, d: FloatArray, v: IntArray, z: FloatArray, n: Int) {
        var k = 0
        v[0] = 0
        z[0] = -INF
        z[1] = INF
        for (q in 1 until n) {
            var s = intersect(f, q, v[k])
            // z[0] ist −∞: Spätestens bei k = 0 endet die Schleife.
            while (s <= z[k]) {
                k--
                s = intersect(f, q, v[k])
            }
            k++
            v[k] = q
            z[k] = s
            z[k + 1] = INF
        }
        k = 0
        for (q in 0 until n) {
            while (z[k + 1] < q) k++
            val p = v[k]
            d[q] = (q - p).toFloat() * (q - p) + f[p]
        }
    }

    /** Wo sich die Parabeln von [q] und [p] schneiden. */
    private fun intersect(f: FloatArray, q: Int, p: Int): Float =
        ((f[q] + q.toFloat() * q) - (f[p] + p.toFloat() * p)) / (2f * q - 2f * p)
}
