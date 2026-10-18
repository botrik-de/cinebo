package de.botrik.cinebo.test

import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import org.json.JSONObject

/** Sammelt, was die Kamera während der Aufnahme pro Bild meldet. */
class FrameStats {
    private var frames = 0L
    private var firstTs = 0L
    private var lastTs = 0L
    private var maxGapNs = 0L
    private var exposureSumNs = 0L
    private var exposureMaxNs = 0L
    private var isoSum = 0L
    private var aeRange = ""

    @Synchronized
    fun reset() {
        frames = 0; firstTs = 0; lastTs = 0; maxGapNs = 0
        exposureSumNs = 0; exposureMaxNs = 0; isoSum = 0; aeRange = ""
    }

    @Synchronized
    fun add(r: TotalCaptureResult) {
        val ts = r.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
        if (frames == 0L) {
            firstTs = ts
        } else {
            val gap = ts - lastTs
            if (gap > maxGapNs) maxGapNs = gap
        }
        lastTs = ts
        frames++
        r.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let {
            exposureSumNs += it
            if (it > exposureMaxNs) exposureMaxNs = it
        }
        r.get(CaptureResult.SENSOR_SENSITIVITY)?.let { isoSum += it }
        r.get(CaptureResult.CONTROL_AE_TARGET_FPS_RANGE)?.let { aeRange = "${it.lower}-${it.upper}" }
    }

    /** Bilder bisher sowie Zeitstempel des ersten und letzten Bildes in Nanosekunden. */
    @Synchronized
    fun snapshot(): Triple<Long, Long, Long> = Triple(frames, firstTs, lastTs)

    @Synchronized
    fun toJson(): JSONObject {
        val o = JSONObject()
        o.put("bilder", frames)
        if (frames > 1 && lastTs > firstTs) {
            val durS = (lastTs - firstTs) / 1e9
            o.put("fps", r2((frames - 1) / durS))
            o.put("groessteLueckeMs", r2(maxGapNs / 1e6))
            o.put("belichtungMittelMs", r2(exposureSumNs.toDouble() / frames / 1e6))
            o.put("belichtungMaxMs", r2(exposureMaxNs / 1e6))
            o.put("isoMittel", isoSum / frames)
        }
        if (aeRange.isNotEmpty()) o.put("aeBildrate", aeRange)
        return o
    }
}
