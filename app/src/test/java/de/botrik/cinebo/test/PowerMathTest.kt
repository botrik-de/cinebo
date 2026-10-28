package de.botrik.cinebo.test

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Einheiten der Strommessung (Hersteller melden µA oder mA, mV oder V, mit beliebigem Vorzeichen),
 * der Trend im Dauertest und der Stand für den Beendigungsgrund.
 */
class PowerMathTest {

    @Test
    fun stromInMikroUndMilliampere() {
        assertEquals(1.5, PowerMath.amps(-1_500_000), 1e-9) // µA, entladen (Android-Vorzeichen)
        assertEquals(1.5, PowerMath.amps(1_500_000), 1e-9) // µA, anderes Vorzeichen
        assertEquals(1.5, PowerMath.amps(-1500), 1e-9) // mA
        assertEquals(0.35, PowerMath.amps(350), 1e-9) // mA, wenig Last
        assertEquals(0.02, PowerMath.amps(20_000), 1e-9) // ab 20 000 als µA
        assertTrue(PowerMath.amps(0).isNaN())
        assertTrue(PowerMath.amps(Int.MIN_VALUE).isNaN())
    }

    @Test
    fun spannungInMillivoltOderVolt() {
        assertEquals(3.87, PowerMath.volts(3870), 1e-9)
        assertEquals(4.0, PowerMath.volts(4), 1e-9)
        assertEquals(3.87, PowerMath.volts(3_870_000), 1e-9) // µV
        assertTrue(PowerMath.volts(0).isNaN())
        assertTrue(PowerMath.volts(-1).isNaN())
    }

    @Test
    fun stromAusDemLadungszaehler() {
        // 8 333 µAh in 20 s sind 1,5 A.
        assertEquals(1.5, PowerMath.ampsFromCharge(5_000_000, 4_991_667, 20.0), 1e-3)
        assertTrue(PowerMath.ampsFromCharge(5_000_000, 5_000_000, 20.0).isNaN())
        assertTrue(PowerMath.ampsFromCharge(5_000_000, 5_000_100, 20.0).isNaN())
        assertTrue(PowerMath.ampsFromCharge(5_000_000, 4_000_000, 0.0).isNaN())
    }

    @Test
    fun leistungAusDemEnergiezaehler() {
        // 10 mWh in 6 s sind 6 W.
        assertEquals(6.0, PowerMath.wattsFromEnergy(10_000_000_000, 9_990_000_000, 6.0), 1e-9)
        assertTrue("führt das Handy nicht", PowerMath.wattsFromEnergy(Long.MIN_VALUE, Long.MIN_VALUE, 6.0).isNaN())
        assertTrue(PowerMath.wattsFromEnergy(0, 0, 6.0).isNaN())
        assertTrue("steigt", PowerMath.wattsFromEnergy(9_990_000_000, 10_000_000_000, 6.0).isNaN())
        assertTrue(PowerMath.wattsFromEnergy(10_000_000_000, 9_990_000_000, 0.0).isNaN())
    }

    @Test
    fun trendDerAkkutemperatur() {
        val s = doubleArrayOf(0.0, 60.0, 120.0, 180.0)
        assertEquals(1.0, Trend.slopePerMinute(s, doubleArrayOf(30.0, 31.0, 32.0, 33.0)), 1e-9)
        assertEquals(-0.5, Trend.slopePerMinute(s, doubleArrayOf(40.0, 39.5, 39.0, 38.5)), 1e-9)
        // Fehlende Werte fallen weg.
        assertEquals(1.0, Trend.slopePerMinute(s, doubleArrayOf(30.0, Double.NaN, 32.0, 33.0)), 1e-9)
        assertTrue(Trend.slopePerMinute(s, doubleArrayOf(30.0, Double.NaN, Double.NaN, Double.NaN)).isNaN())
        assertTrue("ohne Zeitspanne", Trend.slopePerMinute(doubleArrayOf(5.0, 5.0), doubleArrayOf(30.0, 31.0)).isNaN())
        assertEquals(2.0, Trend.mean(doubleArrayOf(1.0, Double.NaN, 3.0)), 1e-9)
        assertTrue(Trend.mean(doubleArrayOf(Double.NaN)).isNaN())
    }

    @Test
    fun standFuerDenBeendigungsgrund() {
        val line = StateText.line(
            "v" to "0.3.16", "test" to "kidauer", "ph" to "abkühlen2", "t" to "3:30", "st" to 2,
            "bat" to 39.46, "W" to Double.NaN, "ki" to 5.0, "leer" to "", "nichts" to null,
        )
        assertEquals("v=0.3.16 test=kidauer ph=abkuehlen2 t=3:30 st=2 bat=39.5 ki=5.0", line)
        assertEquals("Waerme 39,5 C ?", StateText.ascii("Wärme 39,5 °C €"))
        val long = StateText.line("x" to "a".repeat(300))
        assertEquals(StateText.MAX_BYTES, long.length)
        assertTrue(long.all { it.code in 32..126 })
    }
}
