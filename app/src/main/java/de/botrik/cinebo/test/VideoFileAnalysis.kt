package de.botrik.cinebo.test

import android.media.MediaExtractor
import android.media.MediaFormat
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.roundToInt

/** Zählt in der fertigen MP4-Datei jedes Bild und findet Lücken. */
object VideoFileAnalysis {

    /** Lücken in der ersten Sekunde gelten als Startlücke. */
    const val START_WINDOW_S = 1.0

    fun analyze(file: File, requestedFps: Int): JSONObject {
        val o = JSONObject()
        o.put("angefragteFps", requestedFps)
        o.put("dateiMB", r1(file.length() / 1e6))
        val ex = MediaExtractor()
        try {
            ex.setDataSource(file.absolutePath)
            var videoTrack = -1
            for (i in 0 until ex.trackCount) {
                val f = ex.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: ""
                if (mime.startsWith("video/") && videoTrack < 0) {
                    videoTrack = i
                    o.put("format", mime)
                    o.put("breite", f.getInteger(MediaFormat.KEY_WIDTH))
                    o.put("hoehe", f.getInteger(MediaFormat.KEY_HEIGHT))
                    if (f.containsKey(MediaFormat.KEY_FRAME_RATE)) {
                        o.put("angegebeneFps", f.getNumber(MediaFormat.KEY_FRAME_RATE)?.toString() ?: "")
                    }
                    if (f.containsKey(MediaFormat.KEY_ROTATION)) o.put("drehung", f.getInteger(MediaFormat.KEY_ROTATION))
                } else if (mime.startsWith("audio/")) {
                    val audio = JSONObject().put("format", mime)
                    if (f.containsKey(MediaFormat.KEY_SAMPLE_RATE)) audio.put("hz", f.getInteger(MediaFormat.KEY_SAMPLE_RATE))
                    if (f.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) audio.put("kanaele", f.getInteger(MediaFormat.KEY_CHANNEL_COUNT))
                    o.put("ton", audio)
                }
            }
            if (videoTrack < 0) {
                o.put("fehler", "keine Videospur")
                return o
            }

            ex.selectTrack(videoTrack)
            val times = ArrayList<Long>()
            while (true) {
                val t = ex.sampleTime
                if (t < 0) break
                times.add(t)
                if (!ex.advance()) break
            }
            times.sort()

            val n = times.size
            o.put("bilder", n)
            if (n >= 3) {
                val durS = (times[n - 1] - times[0]) / 1e6
                val intervals = LongArray(n - 1) { times[it + 1] - times[it] }
                val sorted = intervals.sorted()
                val median = sorted[sorted.size / 2].toDouble()
                val limit = 1.5 * median
                val gaps = intervals.count { it > limit }
                val lost = intervals.sumOf { if (it > limit) ((it / median).roundToInt() - 1).coerceAtLeast(1) else 0 }
                o.put("dauerS", r1(durS))
                o.put("fps", r2((n - 1) / durS))
                o.put("abstandMedianMs", r2(median / 1000.0))
                o.put("abstandMaxMs", r2(sorted[sorted.size - 1] / 1000.0))
                o.put("luecken", gaps)
                o.put("verloreneBilder", lost)
                // Wo die Lücken liegen: Eine Lücke in der ersten Sekunde entsteht beim Start
                // der Aufnahme und ist im Video nicht zu sehen; später zählt jede.
                val where = JSONArray()
                var lostAfterStart = 0
                for (i in intervals.indices) {
                    val gap = intervals[i]
                    if (gap <= limit) continue
                    val missing = ((gap / median).roundToInt() - 1).coerceAtLeast(1)
                    val atS = (times[i] - times[0]) / 1e6
                    if (atS >= START_WINDOW_S) lostAfterStart += missing
                    if (where.length() < 30) {
                        where.put(JSONObject().put("s", r2(atS)).put("ms", r1(gap / 1000.0)).put("bilder", missing))
                    }
                }
                o.put("lueckenBei", where)
                o.put("verloreneBilderNachStart", lostAfterStart)
                o.put("mbits", r1(file.length() * 8.0 / durS / 1e6))
            }
        } finally {
            ex.release()
        }
        return o
    }
}
