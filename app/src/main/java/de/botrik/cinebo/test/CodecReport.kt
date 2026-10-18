package de.botrik.cinebo.test

import android.media.MediaCodecInfo
import android.media.MediaCodecInfo.CodecProfileLevel
import android.media.MediaCodecInfo.VideoCapabilities.PerformancePoint
import android.media.MediaCodecList
import org.json.JSONArray
import org.json.JSONObject

/** Welche Video-Encoder das Handy hat und was sie bei 4K schaffen. */
object CodecReport {

    private val formats = listOf("video/avc" to "H.264", "video/hevc" to "H.265")

    fun build(): JSONArray {
        val out = JSONArray()
        for (info in MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos) {
            if (!info.isEncoder) continue
            for ((mime, label) in formats) {
                if (info.supportedTypes.none { it.equals(mime, ignoreCase = true) }) continue
                try {
                    out.put(describe(info, mime, label))
                } catch (e: Exception) {
                    out.put(JSONObject().put("name", info.name).put("format", label).put("fehler", e.toString()))
                }
            }
        }
        return out
    }

    private fun describe(info: MediaCodecInfo, mime: String, label: String): JSONObject {
        val caps = info.getCapabilitiesForType(mime)
        val o = JSONObject()
        o.put("name", info.name)
        o.put("format", label)
        o.put("hardware", info.isHardwareAccelerated)
        val video = caps.videoCapabilities ?: return o
        o.put("4k30", video.areSizeAndRateSupported(3840, 2160, 30.0))
        o.put("4k60", video.areSizeAndRateSupported(3840, 2160, 60.0))
        try {
            video.getAchievableFrameRatesFor(3840, 2160)?.let {
                o.put("4kGemesseneFps", "${r1(it.lower)} bis ${r1(it.upper)}")
            }
        } catch (_: Exception) {
        }
        video.supportedPerformancePoints?.let { points ->
            o.put("leistung4k30", points.any { it.covers(PerformancePoint.UHD_30) })
            o.put("leistung4k60", points.any { it.covers(PerformancePoint.UHD_60) })
        }
        o.put("maxMbits", video.bitrateRange.upper / 1_000_000)
        if (mime == "video/hevc") {
            o.put("10bit", caps.profileLevels.any { it.profile == CodecProfileLevel.HEVCProfileMain10 })
        }
        return o
    }
}
