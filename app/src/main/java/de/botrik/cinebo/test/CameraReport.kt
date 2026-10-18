package de.botrik.cinebo.test

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.params.DynamicRangeProfiles
import android.media.MediaRecorder
import android.os.Build
import android.util.Size
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Liest aus, was Android fremden Apps über jede Kamera verrät. */
object CameraReport {

    private val levelNames = mapOf(0 to "LIMITED", 1 to "FULL", 2 to "LEGACY", 3 to "LEVEL_3", 4 to "EXTERNAL")

    private val capabilityNames = mapOf(
        0 to "BACKWARD_COMPATIBLE", 1 to "MANUAL_SENSOR", 2 to "MANUAL_POST_PROCESSING", 3 to "RAW",
        4 to "PRIVATE_REPROCESSING", 5 to "READ_SENSOR_SETTINGS", 6 to "BURST_CAPTURE", 7 to "YUV_REPROCESSING",
        8 to "DEPTH_OUTPUT", 9 to "CONSTRAINED_HIGH_SPEED_VIDEO", 10 to "MOTION_TRACKING", 11 to "LOGICAL_MULTI_CAMERA",
        12 to "MONOCHROME", 13 to "SECURE_IMAGE_DATA", 14 to "SYSTEM_CAMERA", 15 to "OFFLINE_PROCESSING",
        16 to "ULTRA_HIGH_RESOLUTION_SENSOR", 17 to "REMOSAIC_REPROCESSING", 18 to "DYNAMIC_RANGE_TEN_BIT",
        19 to "STREAM_USE_CASE", 20 to "COLOR_SPACE_PROFILES"
    )

    private val sceneModeNames = mapOf(0 to "AUS", 1 to "BOKEH_FOTO", 2 to "BOKEH_VIDEO")
    private val extensionNames = mapOf(0 to "AUTO", 1 to "GESICHT_RETUSCHE", 2 to "BOKEH", 3 to "HDR", 4 to "NACHT")
    private val videoStabNames = mapOf(0 to "AUS", 1 to "VIDEO", 2 to "VORSCHAU_UND_VIDEO")
    private val antibandingNames = mapOf(0 to "AUS", 1 to "50HZ", 2 to "60HZ", 3 to "AUTO")

    private val probeSizes = listOf(Size(3840, 2160), Size(3840, 1644), Size(1920, 1080), Size(4096, 3072))

    fun build(context: Context): JSONObject {
        val cm = context.getSystemService(CameraManager::class.java)
        val root = JSONObject()
        root.put("geraet", JSONObject().apply {
            put("hersteller", Build.MANUFACTURER)
            put("modell", Build.MODEL)
            put("android", Build.VERSION.RELEASE)
            put("sdk", Build.VERSION.SDK_INT)
            put("chipHersteller", Build.SOC_MANUFACTURER)
            put("chip", Build.SOC_MODEL)
        })

        val ids = ArrayList<String>()
        for (id in cm.cameraIdList) {
            if (id !in ids) ids.add(id)
            try {
                for (p in cm.getCameraCharacteristics(id).physicalCameraIds) {
                    if (p !in ids) ids.add(p)
                }
            } catch (_: Exception) {
            }
        }

        val cameras = JSONArray()
        for (id in ids) {
            try {
                cameras.put(describe(cm, id))
            } catch (e: Exception) {
                cameras.put(JSONObject().put("id", id).put("fehler", e.toString()))
            }
        }
        root.put("kameras", cameras)
        return root
    }

    private fun describe(cm: CameraManager, id: String): JSONObject {
        val c = cm.getCameraCharacteristics(id)
        val o = JSONObject()
        o.put("id", id)
        o.put(
            "richtung",
            when (c.get(CameraCharacteristics.LENS_FACING)) {
                CameraCharacteristics.LENS_FACING_BACK -> "hinten"
                CameraCharacteristics.LENS_FACING_FRONT -> "vorne"
                else -> "extern"
            }
        )
        o.put("stufe", levelNames[c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL) ?: -1] ?: "?")

        val physical = c.physicalCameraIds
        if (physical.isNotEmpty()) o.put("linsen", JSONArray(physical.toList()))

        val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        val sensor = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        if (focal != null) {
            o.put("brennweiteMm", JSONArray(focal.map { r2(it.toDouble()) }))
            if (sensor != null) {
                val diag = sqrt((sensor.width * sensor.width + sensor.height * sensor.height).toDouble())
                if (diag > 0.0) o.put("kleinbildMm", JSONArray(focal.map { (it * 43.27 / diag).roundToInt() }))
            }
        }
        if (sensor != null) o.put("sensorMm", "${r2(sensor.width.toDouble())} x ${r2(sensor.height.toDouble())}")
        c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.let { a ->
            o.put("blende", JSONArray(a.map { r2(it.toDouble()) }))
        }
        c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)?.let { d ->
            o.put("naheGrenzeCm", if (d > 0f) (100f / d).roundToInt().toString() else "Fixfokus")
        }
        c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)?.let { o.put("ois", it.contains(1)) }
        c.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)?.let { m ->
            o.put("stabilisierung", JSONArray(m.map { videoStabNames[it] ?: it.toString() }))
        }
        c.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)?.let {
            o.put("zoom", "${r2(it.lower.toDouble())} bis ${r2(it.upper.toDouble())}")
        }
        c.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM)?.let { o.put("digitalzoomMax", r2(it.toDouble())) }

        c.get(CameraCharacteristics.CONTROL_AVAILABLE_EXTENDED_SCENE_MODE_CAPABILITIES)?.let { caps ->
            val arr = JSONArray()
            for (cap in caps) {
                val s = cap.maxStreamingSize
                val z = cap.zoomRatioRange
                arr.put(
                    JSONObject()
                        .put("modus", sceneModeNames[cap.mode] ?: cap.mode.toString())
                        .put("maxGroesse", "${s.width} x ${s.height}")
                        .put("zoom", "${r2(z.lower.toDouble())} bis ${r2(z.upper.toDouble())}")
                )
            }
            o.put("bokehModi", arr)
        }

        c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.let { r ->
            o.put("bildraten", JSONArray(r.map { "${it.lower}-${it.upper}" }))
        }
        c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_ANTIBANDING_MODES)?.let { m ->
            o.put("flimmerschutz", JSONArray(m.map { antibandingNames[it] ?: it.toString() }))
        }

        c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)?.let { map ->
            val videoSizes = map.getOutputSizes(MediaRecorder::class.java)?.toSet() ?: emptySet()
            val fpsBySize = JSONObject()
            for (s in probeSizes) {
                val key = "${s.width}x${s.height}"
                if (s in videoSizes) {
                    val d = map.getOutputMinFrameDuration(MediaRecorder::class.java, s)
                    fpsBySize.put(key, if (d > 0) r1(1e9 / d).toString() else "unbekannt")
                } else {
                    fpsBySize.put(key, "nicht angeboten")
                }
            }
            o.put("maxFpsJeGroesse", fpsBySize)
            val slowMo = JSONArray()
            map.highSpeedVideoSizes?.forEach { s ->
                val ranges = map.getHighSpeedVideoFpsRangesFor(s).joinToString(", ") { "${it.lower}-${it.upper}" }
                slowMo.put("${s.width}x${s.height}: $ranges")
            }
            if (slowMo.length() > 0) o.put("zeitlupe", slowMo)
        }

        c.get(CameraCharacteristics.REQUEST_AVAILABLE_DYNAMIC_RANGE_PROFILES)?.let { p ->
            o.put("dynamikumfang", JSONArray(p.supportedProfiles.map { dynamicRangeName(it) }))
        }
        c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.let { caps ->
            o.put("faehigkeiten", JSONArray(caps.map { capabilityNames[it] ?: it.toString() }))
        }

        try {
            val ext = cm.getCameraExtensionCharacteristics(id).supportedExtensions
            o.put("herstellerModi", JSONArray(ext.map { extensionNames[it] ?: it.toString() }))
        } catch (_: Exception) {
        }
        return o
    }

    private fun dynamicRangeName(profile: Long): String = when (profile) {
        DynamicRangeProfiles.STANDARD -> "SDR"
        DynamicRangeProfiles.HLG10 -> "HLG10"
        DynamicRangeProfiles.HDR10 -> "HDR10"
        DynamicRangeProfiles.HDR10_PLUS -> "HDR10+"
        else -> "Dolby Vision ($profile)"
    }
}
