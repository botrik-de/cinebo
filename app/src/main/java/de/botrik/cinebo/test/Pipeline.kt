package de.botrik.cinebo.test

import android.content.Context
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES30
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.util.Size
import android.view.Surface
import androidx.camera.core.CameraEffect
import androidx.camera.core.SurfaceOutput
import androidx.camera.core.SurfaceProcessor
import androidx.camera.core.SurfaceRequest
import androidx.core.util.Consumer
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.util.concurrent.Executor
import kotlin.math.max
import kotlin.math.roundToInt

/** Was die KI der Grafik-Pipeline über ihren Takt verrät. */
internal interface AiDemand {
    /** Wann die KI voraussichtlich das nächste Bild nimmt (System.nanoTime); 0 = unbekannt. */
    val nextTakeNs: Long

    /** Die KI wartet gerade auf ein Bild. */
    val waiting: Boolean

    /** Rechnet die KI beim nächsten Durchgang die Tiefe? Sonst spart sich die Pipeline das Tiefenbild. */
    val depthNeeded: Boolean
}

/** Nimmt die Masken der KI an (vom KI-Thread). */
internal interface MaskSink {
    /**
     * [probabilities] (256 × 256, Zeile 0 oben) gehört zum Kamerabild [frameNs]; wird sofort
     * kopiert. [generation] zählt, wie oft die Verfolgung neu angesetzt hat (Antippen): Masken
     * verschiedener Motive mischt die Pipeline nicht.
     */
    fun onMask(frameNs: Long, probabilities: ByteArray, present: Boolean, generation: Int)
}

/** CameraX-Effekt mit unserem Prozessor: ein Kamerastrom für Vorschau (und Video). */
internal class PipelineEffect(targets: Int, processor: BokehProcessor) :
    CameraEffect(targets, processor.executor, processor, Consumer { processor.onEffectError(it) })

/**
 * Die Grafik-Pipeline aus Stufe 3, als Effekt in CameraX: Jedes Kamerabild geht hier durch,
 * bevor CameraX es an Vorschau und Videoaufnahme gibt.
 *
 * - Für die Unschärfe hält sie die letzten Bilder in einem Ring und gibt jedes erst
 *   [requestedDelay] Bilder später aus. Bis dahin hat die KI die Masken zu den KI-Bildern davor
 *   und danach gerechnet; das Bild bekommt die Mischung beider nach der Zeit. So sind Bild und
 *   Maske verschweißt. Die Unschärfe entsteht nur aus Hintergrund (kein Lichthof ums Motiv).
 *   Die Ausgabe trägt den Zeitstempel der Kamera; die Zeitachse der Aufnahme samt Ton stimmt
 *   deshalb trotz der Verzögerung.
 * - Für die KI rendert sie bei Bedarf ein aufrechtes Bild (1024² gestreckt für EdgeTAM, 518²
 *   eingepasst für die Tiefe) und liest es als Gleitkommazahlen direkt in den [FramePool].
 * - Ohne Unschärfe ([strength] 0) reicht sie jedes Bild sofort durch.
 *
 * Für jede Bindung an CameraX einen neuen Prozessor nehmen und ihn nach dem Lösen mit
 * [release] freigeben. Alles läuft auf einem eigenen Thread mit einem OpenGL-ES-3-Kontext.
 */
internal class BokehProcessor(
    private val context: Context,
    /** 0 = keine Unschärfe und keine Verzögerung, 1 = volle Unschärfe. */
    private val strength: Float,
    /** Um so viele Bilder wartet die Ausgabe auf die Masken der KI. */
    private val requestedDelay: Int = DEFAULT_DELAY,
    /** Bekommt die Bilder für die KI; null = keine KI. */
    private val aiPool: FramePool? = null,
    aiDemand: AiDemand? = null,
    /** Nach so vielen KI-Bildern ein Vorschaubild für den Bericht (0 = keins). */
    private val thumbnailAfter: Int = THUMBNAIL_AFTER,
    staticMask: ByteArray? = null,
) : SurfaceProcessor, MaskSink {

    /** Wer die KI-Bilder abnimmt (Takt, Tiefenbild); lässt sich für den Strombedarf mitten im Lauf wechseln. */
    @Volatile var aiDemand: AiDemand? = aiDemand

    /**
     * Für den Strombedarf: eine feste Maske (256² Wahrscheinlichkeiten) für jedes Bild statt der
     * KI. Die Unschärfe läuft dann voll, ohne dass die KI etwas rechnet; Masken der KI zählen
     * nicht. Null = normal. Lässt sich mitten im Lauf ein- und ausschalten.
     */
    @Volatile var staticMask: ByteArray? = staticMask

    /** Ob gerade die feste Maske im Maskenring liegt (nur Pipeline-Thread). */
    private var staticShown = false

    // Hohe Priorität wie bei der Bildausgabe: Wer hier zu spät kommt, verliert Bilder.
    private val thread = HandlerThread("pipeline", Process.THREAD_PRIORITY_URGENT_DISPLAY).apply { start() }
    private val handler = Handler(thread.looper)

    /**
     * Auf diesem Thread läuft alles; CameraX ruft den Prozessor darüber auf. Ist der Thread nach
     * dem Freigeben schon beendet, läuft ein später Auftrag direkt; alle Einstiege sind dann
     * harmlos, und CameraX wartet nie vergeblich auf eine Antwort.
     */
    val executor: Executor = Executor { r -> if (!handler.post(r)) r.run() }

    val stats = PipelineStats()

    // ---------------------------------------------------- Zustand (nur auf dem Pipeline-Thread)

    @Volatile private var releaseRequested = false
    private var glDone = false
    private val forceRelease = Runnable { finishRelease(force = true) }
    private var display: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglConfig: EGLConfig? = null
    private var pbuffer: EGLSurface = EGL14.EGL_NO_SURFACE
    private var vao = 0
    private lateinit var copyOes: Program
    private lateinit var aiFrame: Program
    private lateinit var pack: Program
    private lateinit var downOes: Program
    private lateinit var down: Program
    private lateinit var weigh: Program
    private lateinit var gauss: Program
    private lateinit var composite: Program
    private var floatPacking = false
    private val gaussTaps = PipelineMath.gaussTaps(BLUR_SIGMA)

    private class Input(val request: SurfaceRequest, val st: SurfaceTexture, val surface: Surface, val oes: Int, val size: Size) {
        /** Drehung vom gespeicherten Puffer zum aufrechten Bild laut CameraX; −1 = noch unbekannt. */
        @Volatile var rotation = -1
        var released = false
        /** Zuletzt gemeldete Lage (st-Drehung, Rest bis aufrecht); ändert sie sich, beginnt der Fluss neu. */
        var orientationKey = ""
    }

    private class Output(val target: SurfaceOutput, val egl: EGLSurface, val size: Size) {
        val matrix = FloatArray(16)
    }

    private val inputs = ArrayList<Input>()
    private var current: Input? = null
    private val outputs = ArrayList<Output>()
    private val stMatrix = FloatArray(16)

    // Ring der verzögerten Bilder im st-Raum: volles Bild und seine Fassung in einem Achtel der Größe.
    private class Slot(val tex: Int, val fbo: Int, val smallTex: Int, val smallFbo: Int) {
        var ts = 0L
    }

    private var ring: List<Slot> = emptyList()
    private var ringSize: Size? = null
    private var ringFailedFor: Size? = null
    private var smallW = 1
    private var smallH = 1
    private var delay = 0
    /** Zuletzt gezeigte Stärke der Unschärfe: Sie steigt höchstens um [BLUR_RAMP] je Bild, nie sprunghaft. */
    private var shownBlur = 0f
    /** Zeitstempel des ersten Bildes seit dem letzten Neubeginn; ältere Masken gehören nicht mehr dazu. */
    private var flowStartNs = Long.MAX_VALUE
    private val queue = ArrayDeque<Slot>()
    private val freeSlots = ArrayDeque<Slot>()
    // Zwischenstufen: Viertel (Verkleinern), Hintergrund und Gauß-Zwischenschritt (Achtel).
    private var quarterTex = 0
    private var quarterFbo = 0
    private var bgTex = 0
    private var bgFbo = 0
    private var bgTmpTex = 0
    private var bgTmpFbo = 0

    // Bilder für die KI.
    private var encTex = 0
    private var encFbo = 0
    private var depthTex = 0
    private var depthFbo = 0
    private var encPackTex = 0
    private var encPackFbo = 0
    private var depthPackTex = 0
    private var depthPackFbo = 0
    private var aiTargetsFailed = false
    private val encLayout = PipelineMath.PackLayout(FrameSlot.ENC, ENC_PACK_WIDTH)
    private val depthLayout = PipelineMath.PackLayout(FrameSlot.DEPTH, DEPTH_PACK_WIDTH)
    private var rgbaRead: ByteBuffer? = null
    private var rgbaBytes: ByteArray? = null
    private var rgbFloats: FloatArray? = null
    private var thumbnailDone = false

    // Masken der KI: Ring aus R8-Texturen mit dem Zeitstempel ihres Kamerabilds.
    private val maskTex = IntArray(MASKS)
    private val maskNs = LongArray(MASKS)
    private val maskPresent = BooleanArray(MASKS)
    private val maskGeneration = IntArray(MASKS)
    private var maskCount = 0
    private var maskNext = 0
    private val pendingLock = Any()
    private val pending = ArrayDeque<PendingMask>()
    private val maskUpload: ByteBuffer = ByteBuffer.allocateDirect(MASK_SIZE * MASK_SIZE)

    private class PendingMask(val ns: Long, val bytes: ByteArray, val present: Boolean, val generation: Int)

    // ---------------------------------------------------- SurfaceProcessor

    override fun onInputSurface(request: SurfaceRequest) {
        if (releaseRequested) {
            request.willNotProvideSurface()
            return
        }
        val input: Input
        try {
            ensureGl()
            val oes = createOesTexture()
            val res = request.resolution
            val st = SurfaceTexture(oes)
            st.setDefaultBufferSize(res.width, res.height)
            input = Input(request, st, Surface(st), oes, res)
        } catch (e: Throwable) {
            stats.error("Eingang", e)
            request.willNotProvideSurface()
            return
        }
        inputs.add(input)
        current = input
        resetFlow()
        stats.input(input.size)
        request.setTransformationInfoListener(executor) { info ->
            input.rotation = info.rotationDegrees
            stats.rotation(info.rotationDegrees, info.cropRect.width(), info.cropRect.height(), info.isMirroring)
        }
        input.st.setOnFrameAvailableListener({ onFrame(input) }, handler)
        request.provideSurface(input.surface, executor) {
            // CameraX braucht den Eingang nicht mehr.
            releaseInput(input)
            if (releaseRequested) finishRelease(force = false)
        }
    }

    override fun onOutputSurface(surfaceOutput: SurfaceOutput) {
        if (releaseRequested) {
            surfaceOutput.close()
            return
        }
        try {
            ensureGl()
            var output: Output? = null
            // CameraX bittet über dieses Ereignis, die Ausgabe nicht mehr zu benutzen und zu schließen.
            val surface = surfaceOutput.getSurface(executor) {
                val o = output
                if (o != null) removeOutput(o) else surfaceOutput.close()
            }
            val egl = EGL14.eglCreateWindowSurface(display, eglConfig, surface, intArrayOf(EGL14.EGL_NONE), 0)
            check(egl != null && egl != EGL14.EGL_NO_SURFACE) { "Ausgabefläche: EGL-Fehler 0x${Integer.toHexString(EGL14.eglGetError())}" }
            output = Output(surfaceOutput, egl, surfaceOutput.size)
            outputs.add(output)
            stats.output(surfaceOutput.size, surfaceOutput.targets)
        } catch (e: Throwable) {
            stats.error("Ausgabe", e)
            surfaceOutput.close()
        }
    }

    /** Fehler, die CameraX über den Effekt meldet. */
    fun onEffectError(e: Throwable) {
        stats.error("CameraX", e)
    }

    override fun onMask(frameNs: Long, probabilities: ByteArray, present: Boolean, generation: Int) {
        val copy = probabilities.copyOf(MASK_SIZE * MASK_SIZE)
        synchronized(pendingLock) {
            pending.addLast(PendingMask(frameNs, copy, present, generation))
            while (pending.size > MAX_PENDING_MASKS) pending.removeFirst()
        }
    }

    /** Ob die Pipeline den Hintergrund unscharf macht (sonst reicht sie die Bilder unverändert durch). */
    val blurs: Boolean get() = strength > 0f

    /**
     * Gibt alles frei; danach ist der Prozessor nicht mehr nutzbar. Nach dem Lösen von CameraX
     * aufrufen. Wie bei CameraX' eigenem Prozessor endet OpenGL erst, wenn CameraX den Eingang
     * zurückgegeben hat (spätestens nach 3 s), damit die Kamera nie in eine freigegebene Fläche
     * schreibt.
     */
    fun release() {
        releaseRequested = true
        if (handler.post { finishRelease(force = false) }) handler.postDelayed(forceRelease, RELEASE_TIMEOUT_MS)
    }

    private fun finishRelease(force: Boolean) {
        if (glDone || !releaseRequested) return
        if (inputs.isNotEmpty()) {
            if (!force) return
            stats.note("Freigabe ohne Rückgabe des Eingangs durch CameraX")
        }
        for (input in inputs.toList()) releaseInput(input)
        for (output in outputs.toList()) removeOutput(output)
        try {
            if (display != EGL14.EGL_NO_DISPLAY) {
                EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                if (pbuffer != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, pbuffer)
                EGL14.eglDestroyContext(display, eglContext)
                EGL14.eglReleaseThread()
                // Android zählt eglInitialize/eglTerminate je Prozess; CameraX' Kontext bleibt heil.
                EGL14.eglTerminate(display)
            }
        } catch (e: Throwable) {
            stats.error("OpenGL freigeben", e)
        }
        display = EGL14.EGL_NO_DISPLAY
        glDone = true
        // Nur die Zeitgrenze zurücknehmen; was CameraX schon geschickt hat, läuft noch durch.
        handler.removeCallbacks(forceRelease)
        thread.quitSafely()
    }

    // ---------------------------------------------------- Bild für Bild

    private fun onFrame(input: Input) {
        if (glDone || input.released) return
        val t0 = System.nanoTime()
        try {
            makeCurrent(pbuffer)
            input.st.updateTexImage()
        } catch (e: Throwable) {
            stats.error("updateTexImage", e)
            return
        }
        // Bilder eines abgelösten Eingangs oder nach der Freigabe nur abholen, damit die Kamera nicht hängt.
        if (releaseRequested || input !== current) return
        val ts = input.st.timestamp
        input.st.getTransformMatrix(stMatrix)
        stats.frameIn(ts)
        if (flowStartNs == Long.MAX_VALUE) flowStartNs = ts
        try {
            uploadMasks()
            // Alles hier liegt im st-Raum: Die Matrix der SurfaceTexture dreht den quer
            // gespeicherten Kamerapuffer schon in die natürliche Lage des Geräts. Aufrecht ist
            // der st-Raum noch um den Rest gedreht, den CameraX darüber hinaus verlangt.
            val stRotation = PipelineMath.stRotation(stMatrix)
            val rotationDegrees = if (input.rotation >= 0) input.rotation else stRotation
            val upright = PipelineMath.uprightFromSt(rotationDegrees, stMatrix)
            val (sw, sh) = PipelineMath.stSize(input.size.width, input.size.height, stMatrix)
            val key = "$stRotation/$upright"
            if (key != input.orientationKey) {
                // Gespeicherte Bilder und Masken passen nicht mehr zu einer neuen Lage.
                if (input.orientationKey.isNotEmpty()) {
                    resetFlow()
                    stats.note("Lage geändert: st $stRotation°, bis aufrecht $upright°")
                }
                input.orientationKey = key
                stats.orientation(stRotation, PipelineMath.stMirrored(stMatrix), upright, sw, sh)
            }
            // Erst das Bild für Vorschau und Aufnahme, dann das KI-Bild: Das Lesen des KI-Bildes
            // wartet auf die Grafikeinheit und soll die Ausgabe nicht aufhalten.
            val tOut = System.nanoTime()
            if (outputs.isNotEmpty()) {
                if (strength > 0f && ensureRing(Size(sw, sh))) {
                    pushRing(input, ts, sw, sh)
                    while (queue.size > delay) emit(queue.removeFirst(), upright)
                } else {
                    drawDirect(input, ts)
                }
            }
            val outMs = (System.nanoTime() - tOut) / 1e6
            // Ein GL-Fehler der Ausgabe gehört zur Ausgabe, nicht zum folgenden KI-Bild.
            val glError = GLES30.glGetError()
            if (glError != GLES30.GL_NO_ERROR) {
                drainGlErrors()
                stats.outputFailed("GL-Fehler 0x${Integer.toHexString(glError)}")
            }
            val aiMs = if (aiPool != null) maybeAiFrame(input, ts, sw, sh, upright) else Double.NaN
            stats.frameDone((System.nanoTime() - t0) / 1e6, outMs, aiMs)
        } catch (e: Throwable) {
            stats.error("Bild", e)
        }
    }

    /** Ohne Unschärfe: Kamerabild direkt in jede Ausgabe, mit seinem Zeitstempel. */
    private fun drawDirect(input: Input, ts: Long) {
        var failure: String? = null
        for (o in outputs.toList()) {
            try {
                o.target.updateTransformMatrix(o.matrix, stMatrix)
                makeCurrent(o.egl)
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
                GLES30.glViewport(0, 0, o.size.width, o.size.height)
                copyOes.use()
                bindOes(0, input.oes)
                copyOes.int("uCam", 0)
                copyOes.mat("uTex", o.matrix)
                drawTriangle()
                present(o, ts)?.let { failure = it }
            } catch (e: Throwable) {
                failure = "${e.javaClass.simpleName}: ${e.message}"
            }
        }
        failure?.let { stats.outputFailed(it) }
        stats.frameOut(ts, null, blurred = false, subject = false, stale = false, gapNs = -1L)
    }

    /** Kamerabild in den Ring (st-Raum, [w] × [h]) und seine Fassung in einem Achtel der Größe dazu. */
    private fun pushRing(input: Input, ts: Long, w: Int, h: Int) {
        val slot = freeSlots.removeFirstOrNull() ?: queue.removeFirst().also { stats.dropped() }
        try {
            slot.ts = ts
            // Scharfes Bild: vUv ist die Koordinate im st-Raum; die Matrix der SurfaceTexture führt
            // zur Textur. Der Ring hat die Größe des st-Raums, so bleibt jeder Bildpunkt erhalten.
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, slot.fbo)
            GLES30.glViewport(0, 0, w, h)
            copyOes.use()
            bindOes(0, input.oes)
            copyOes.int("uCam", 0)
            copyOes.mat("uTex", stMatrix)
            drawTriangle()
            // Verkleinert: auf ein Viertel (4 × 4 Pixel aus der Kamera gemittelt), dann auf ein Achtel.
            // Die Unschärfe daraus entsteht erst bei der Ausgabe, mit der Maske dieses Zeitpunkts.
            val (qw, qh) = quarter(w, h)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, quarterFbo)
            GLES30.glViewport(0, 0, qw, qh)
            downOes.use()
            bindOes(0, input.oes)
            downOes.int("uCam", 0)
            downOes.mat("uTex", stMatrix)
            downOes.vec2("uStep", 1f / w, 1f / h)
            drawTriangle()
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, slot.smallFbo)
            GLES30.glViewport(0, 0, smallW, smallH)
            down.use()
            bind2d(0, quarterTex)
            down.int("uSrc", 0)
            drawTriangle()
            queue.addLast(slot)
        } catch (e: Throwable) {
            freeSlots.addLast(slot)
            throw e
        }
    }

    /**
     * Ein verzögertes Bild mit der Maske seines Zeitpunkts ausgeben; [upright]: Drehung st-Raum →
     * aufrecht. Der Platz im Ring kommt in jedem Fall zurück, auch wenn eine Ausgabe scheitert:
     * Sonst stünde das Bild für immer.
     */
    private fun emit(slot: Slot, upright: Int) {
        try {
            var pick = if (staticShown) {
                if (maskCount > 0) PipelineMath.MaskPick(0, 0, 0f, 0L) else null
            } else {
                PipelineMath.pickMasks(maskNs, maskCount, slot.ts, MAX_MASK_DISTANCE_NS)
            }
            var fade = 1f
            var stale = false
            if (pick == null) {
                // Stockt die KI, gilt die nächste Maske weiter, mit nachlassender Unschärfe: Das
                // Bild springt nie hart zwischen unscharf und scharf.
                pick = PipelineMath.pickMasks(maskNs, maskCount, slot.ts, Long.MAX_VALUE)
                if (pick != null) {
                    fade = PipelineMath.maskFade(pick.distanceNs)
                    stale = true
                }
            }
            // Nach dem Antippen eines neuen Motivs nicht zwischen altem und neuem Motiv überblenden.
            if (pick != null && pick.a != pick.b && maskGeneration[pick.a] != maskGeneration[pick.b]) {
                pick = if (pick.mix < 0.5f) pick.copy(b = pick.a, mix = 0f) else pick.copy(a = pick.b, mix = 0f)
            }
            val presentA = pick != null && maskPresent[pick.a]
            val presentB = pick != null && maskPresent[pick.b]
            // Nach oben höchstens um BLUR_RAMP je Bild (≈ 0,27 s bis voll), etwa wenn die KI nach
            // einem Stocken wieder Masken liefert; nach unten folgt sie gleich (Masken und Alter
            // blenden dort schon weich über).
            val target = if (pick != null && (presentA || presentB)) strength * fade else 0f
            val blur = if (target > shownBlur) minOf(target, shownBlur + BLUR_RAMP) else target
            shownBlur = blur
            val bufToMask = PipelineMath.bufferToUpright(upright)
            if (pick != null && blur > 0f) renderBackground(slot, pick, presentA, presentB, bufToMask)
            val stInverse = PipelineMath.invert(stMatrix) ?: PipelineMath.identity()
            var failure: String? = null
            for (o in outputs.toList()) {
                try {
                    o.target.updateTransformMatrix(o.matrix, stMatrix)
                    // Ausgabe-Koordinate → st-Raum: Die Ausgabematrix führt zur Textur, die Umkehrung
                    // der SurfaceTexture-Matrix von dort zurück in den st-Raum, in dem der Ring liegt.
                    val outToBuf = PipelineMath.multiply(stInverse, o.matrix)
                    makeCurrent(o.egl)
                    GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
                    GLES30.glViewport(0, 0, o.size.width, o.size.height)
                    composite.use()
                    // Texturen vor jedem Zeichnen neu binden (ein Zwischenschritt kann Einheiten belegt haben).
                    bind2d(0, slot.tex)
                    bind2d(1, bgTex)
                    bind2d(2, slot.smallTex)
                    bind2d(3, maskTex[pick?.a ?: 0])
                    bind2d(4, maskTex[pick?.b ?: 0])
                    composite.int("uSharp", 0)
                    composite.int("uBg", 1)
                    composite.int("uSmall", 2)
                    composite.int("uMaskA", 3)
                    composite.int("uMaskB", 4)
                    composite.mat("uOutToBuf", outToBuf)
                    composite.mat("uBufToMask", bufToMask)
                    composite.float("uMix", pick?.mix ?: 0f)
                    composite.vec2("uPresent", if (presentA) 1f else 0f, if (presentB) 1f else 0f)
                    composite.float("uStrength", blur)
                    drawTriangle()
                    present(o, slot.ts)?.let { failure = it }
                } catch (e: Throwable) {
                    failure = "${e.javaClass.simpleName}: ${e.message}"
                }
            }
            failure?.let { stats.outputFailed(it) }
            val gapNs = if (pick != null && pick.a != pick.b) maskNs[pick.b] - maskNs[pick.a] else -1L
            stats.frameOut(slot.ts, pick, blurred = true, subject = blur > 0f, stale = stale, gapNs = gapNs)
        } finally {
            // Nichts bleibt gebunden, was ein späterer Schritt als Ziel beschreibt.
            for (unit in 0 until 5) bind2d(unit, 0)
            freeSlots.addLast(slot)
        }
    }

    /**
     * Unschärfe nur aus Hintergrund für [slot]: verkleinertes Bild mal Anteil Hintergrund
     * (weigh.frag), dann Gauß waagerecht und senkrecht. Ergebnis (rgb · w, w) in [bgTex];
     * composite.frag teilt durch w.
     */
    private fun renderBackground(slot: Slot, pick: PipelineMath.MaskPick, presentA: Boolean, presentB: Boolean, bufToMask: FloatArray) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, bgFbo)
        GLES30.glViewport(0, 0, smallW, smallH)
        weigh.use()
        bind2d(0, slot.smallTex)
        bind2d(1, maskTex[pick.a])
        bind2d(2, maskTex[pick.b])
        weigh.int("uSmall", 0)
        weigh.int("uMaskA", 1)
        weigh.int("uMaskB", 2)
        weigh.mat("uBufToMask", bufToMask)
        weigh.float("uMix", pick.mix)
        weigh.vec2("uPresent", if (presentA) 1f else 0f, if (presentB) 1f else 0f)
        drawTriangle()
        gaussPass(bgTex, bgTmpFbo, 1f / smallW, 0f)
        gaussPass(bgTmpTex, bgFbo, 0f, 1f / smallH)
    }

    private fun gaussPass(src: Int, dstFbo: Int, dx: Float, dy: Float) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, dstFbo)
        GLES30.glViewport(0, 0, smallW, smallH)
        gauss.use()
        bind2d(0, src)
        gauss.int("uSrc", 0)
        gauss.vec2("uDir", dx, dy)
        gauss.floats("uWeight", gaussTaps.first)
        gauss.floats("uOffset", gaussTaps.second)
        drawTriangle()
    }

    /** Bild mit dem Zeitstempel der Kamera abgeben; null = geklappt, sonst was schiefging. */
    private fun present(o: Output, ts: Long): String? {
        EGLExt.eglPresentationTimeANDROID(display, o.egl, ts)
        if (EGL14.eglSwapBuffers(display, o.egl)) return null
        return "eglSwapBuffers: EGL-Fehler 0x${Integer.toHexString(EGL14.eglGetError())}"
    }

    // ---------------------------------------------------- Bilder für die KI

    /**
     * Bereitet ein Bild für die KI vor, wenn sie bald eins nimmt; gibt die Zeit in ms zurück
     * (NaN = keins). Der st-Raum ist [sw] × [sh] groß und steht um [upright] gedreht aufrecht.
     */
    private fun maybeAiFrame(input: Input, ts: Long, sw: Int, sh: Int, upright: Int): Double {
        val pool = aiPool ?: return Double.NaN
        val demand = aiDemand ?: return Double.NaN
        if (!PipelineMath.shouldPrepare(System.nanoTime(), demand.nextTakeNs, demand.waiting, LEAD_MS)) return Double.NaN
        if (!ensureAiTargets()) return Double.NaN
        val slot = pool.acquire() ?: return Double.NaN
        val start = System.nanoTime()
        try {
            val withDepth = demand.depthNeeded
            val (uw, uh) = PipelineMath.uprightSize(sw, sh, upright)
            val uprightToTex = PipelineMath.multiply(stMatrix, PipelineMath.uprightToBuffer(upright))
            drawAiFrame(input, encFbo, FrameSlot.ENC, uprightToTex, FULL, uw, uh)
            val box = PipelineMath.letterbox(uw, uh, FrameSlot.DEPTH)
            if (withDepth) drawAiFrame(input, depthFbo, FrameSlot.DEPTH, uprightToTex, box, uw, uh)
            if (floatPacking) {
                packAndRead(encTex, encPackFbo, encLayout, slot.enc)
                if (withDepth) packAndRead(depthTex, depthPackFbo, depthLayout, slot.depth)
            } else {
                readAsFloats(encFbo, FrameSlot.ENC, slot.enc)
                if (withDepth) readAsFloats(depthFbo, FrameSlot.DEPTH, slot.depth)
            }
            slot.hasDepth = withDepth
            slot.depthContent.set(box[0], box[1], box[2], box[3])
            slot.timestampNs = ts
            slot.width = uw
            slot.height = uh
            slot.rotation = upright
            slot.prepMs = (System.nanoTime() - start) / 1e6
            // Einmal je Lauf ein kleines Bild dessen, was die KI bekommt, zurückgerechnet aus den
            // gelesenen Zahlen: So zeigt der Bericht Ausrichtung und Farben des ganzen Weges. Hier
            // nur die Bildpunkte (schnell); das JPEG entsteht auf einem eigenen Thread, damit die
            // Aufnahme kein Bild verliert.
            if (thumbnailAfter > 0 && !thumbnailDone && stats.aiFrames() + 1 >= thumbnailAfter) {
                thumbnailDone = true
                try {
                    val size = FrameSlot.THUMBNAIL
                    val pixels = slot.thumbnailPixels(size)
                    Thread({
                        try {
                            stats.thumbnail(FrameSlot.jpegBase64(pixels, size))
                        } catch (e: Throwable) {
                            stats.error("Vorschaubild", e)
                        }
                    }, "vorschaubild").start()
                } catch (e: Throwable) {
                    stats.error("Vorschaubild", e)
                }
            }
            stats.aiFrame(withDepth)
            pool.publish(slot)
            return slot.prepMs
        } catch (e: Throwable) {
            pool.giveBack(slot)
            stats.error("KI-Bild", e)
            return Double.NaN
        }
    }

    private fun drawAiFrame(input: Input, fbo: Int, size: Int, uprightToTex: FloatArray, content: FloatArray, uw: Int, uh: Int) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glViewport(0, 0, size, size)
        aiFrame.use()
        bindOes(0, input.oes)
        aiFrame.int("uCam", 0)
        aiFrame.mat("uUprightToTex", uprightToTex)
        aiFrame.vec4("uContent", content[0], content[1], content[2], content[3])
        aiFrame.vec2("uTarget", size.toFloat(), size.toFloat())
        aiFrame.vec2("uSource", uw.toFloat(), uh.toFloat())
        drawTriangle()
    }

    /** RGBA8-Bild als Gleitkommazahlen RGB packen und direkt in [out] lesen (Eingang des Modells). */
    private fun packAndRead(src: Int, packFbo: Int, layout: PipelineMath.PackLayout, out: ByteBuffer) {
        check(out.capacity() == layout.bytes) { "Puffer ${out.capacity()} statt ${layout.bytes} Bytes" }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, packFbo)
        GLES30.glViewport(0, 0, layout.width, layout.height)
        pack.use()
        bind2d(0, src)
        pack.int("uSrc", 0)
        pack.int("uSize", layout.size)
        pack.int("uWidth", layout.width)
        pack.int("uTotal", layout.total)
        drawTriangle()
        out.clear()
        GLES30.glReadPixels(0, 0, layout.width, layout.fullRows, GLES30.GL_RGBA, GLES30.GL_FLOAT, out)
        if (layout.rest > 0) {
            // Angefangene letzte Zeile: nur so viele Texel, wie der Eingang noch braucht.
            val tail = out.duplicate().apply { position(layout.fullRows * layout.width * 16) }.slice()
            GLES30.glReadPixels(0, layout.fullRows, layout.rest, 1, GLES30.GL_RGBA, GLES30.GL_FLOAT, tail)
        }
        out.rewind()
        checkGl("Lesen")
    }

    /** Rückfall ohne Gleitkomma-Rendern: RGBA8 lesen und auf der CPU umrechnen. */
    private fun readAsFloats(fbo: Int, size: Int, out: ByteBuffer) {
        val n = size * size
        val read = rgbaRead?.takeIf { it.capacity() >= n * 4 }
            ?: ByteBuffer.allocateDirect(FrameSlot.ENC * FrameSlot.ENC * 4).also { rgbaRead = it }
        val bytes = rgbaBytes ?: ByteArray(FrameSlot.ENC * FrameSlot.ENC * 4).also { rgbaBytes = it }
        val floats = rgbFloats ?: FloatArray(FrameSlot.ENC * FrameSlot.ENC * 3).also { rgbFloats = it }
        read.clear()
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glReadPixels(0, 0, size, size, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, read)
        checkGl("Lesen RGBA")
        read.rewind()
        read.get(bytes, 0, n * 4)
        var j = 0
        for (i in 0 until n) {
            floats[j] = (bytes[i * 4].toInt() and 0xFF) * INV255
            floats[j + 1] = (bytes[i * 4 + 1].toInt() and 0xFF) * INV255
            floats[j + 2] = (bytes[i * 4 + 2].toInt() and 0xFF) * INV255
            j += 3
        }
        out.rewind()
        out.asFloatBuffer().put(floats, 0, n * 3)
        out.rewind()
    }

    // ---------------------------------------------------- Masken

    private fun uploadMasks() {
        val fixed = staticMask
        if (fixed == null && staticShown) {
            // Zurück zur KI: Die feste Maske gehört nicht zu ihren Bildern.
            staticShown = false
            maskCount = 0
            maskNext = 0
        }
        if (fixed != null) {
            // Feste Maske: einmal hochladen, danach gilt sie für jedes Bild; Masken der KI fallen weg.
            synchronized(pendingLock) { pending.clear() }
            if (!staticShown || maskCount == 0) {
                staticShown = true
                maskUpload.clear()
                maskUpload.put(fixed, 0, MASK_SIZE * MASK_SIZE)
                maskUpload.rewind()
                GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTex[0])
                GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, MASK_SIZE, MASK_SIZE, GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, maskUpload)
                maskNs[0] = 0L
                maskPresent[0] = true
                maskGeneration[0] = 0
                maskCount = 1
                maskNext = 1
            }
            return
        }
        while (true) {
            val m = synchronized(pendingLock) { pending.removeFirstOrNull() } ?: return
            // Masken zu Bildern vor dem letzten Neubeginn (neuer Eingang, neue Lage) passen nicht mehr.
            if (m.ns < flowStartNs) continue
            val i = maskNext
            maskNext = (maskNext + 1) % MASKS
            if (maskCount < MASKS) maskCount++
            maskUpload.clear()
            maskUpload.put(m.bytes, 0, MASK_SIZE * MASK_SIZE)
            maskUpload.rewind()
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTex[i])
            GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, MASK_SIZE, MASK_SIZE, GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, maskUpload)
            maskNs[i] = m.ns
            maskPresent[i] = m.present
            maskGeneration[i] = m.generation
            stats.maskIn()
        }
    }

    // ---------------------------------------------------- Aufräumen

    private fun releaseInput(input: Input) {
        if (!input.released) {
            input.released = true
            try {
                input.request.clearTransformationInfoListener()
                input.st.setOnFrameAvailableListener(null)
                input.st.release()
                input.surface.release()
                if (display != EGL14.EGL_NO_DISPLAY) {
                    makeCurrent(pbuffer)
                    GLES30.glDeleteTextures(1, intArrayOf(input.oes), 0)
                }
            } catch (e: Throwable) {
                stats.error("Eingang freigeben", e)
            }
            if (current === input) {
                current = null
                resetFlow()
            }
        }
        inputs.remove(input)
    }

    private fun removeOutput(output: Output) {
        if (!outputs.remove(output)) return
        try {
            if (display != EGL14.EGL_NO_DISPLAY) {
                makeCurrent(pbuffer)
                EGL14.eglDestroySurface(display, output.egl)
            }
        } catch (e: Throwable) {
            stats.error("Ausgabe freigeben", e)
        }
        output.target.close()
    }

    /** Neuer Eingang oder neue Lage: alte Bilder und Masken gelten nicht mehr. */
    private fun resetFlow() {
        freeSlots.addAll(queue)
        queue.clear()
        maskCount = 0
        maskNext = 0
        shownBlur = 0f
        flowStartNs = Long.MAX_VALUE
        synchronized(pendingLock) { pending.clear() }
    }

    // ---------------------------------------------------- OpenGL

    /**
     * EGL und alle Programme einrichten. Scheitert ein Schritt, wird alles wieder abgebaut, damit
     * ein neuer Versuch sauber beginnt und kein halb eingerichteter Zustand bleibt.
     */
    private fun ensureGl() {
        if (display != EGL14.EGL_NO_DISPLAY) return
        check(!glDone) { "Pipeline schon freigegeben" }
        val dpy = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        check(dpy != EGL14.EGL_NO_DISPLAY) { "Kein EGL-Display" }
        val version = IntArray(2)
        check(EGL14.eglInitialize(dpy, version, 0, version, 1)) { "EGL startet nicht: 0x${Integer.toHexString(EGL14.eglGetError())}" }
        display = dpy
        try {
            val attrs = intArrayOf(
                EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8, EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_DEPTH_SIZE, 0, EGL14.EGL_STENCIL_SIZE, 0,
                EGL14.EGL_RENDERABLE_TYPE, EGLExt.EGL_OPENGL_ES3_BIT_KHR,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT or EGL14.EGL_PBUFFER_BIT,
                EGLExt.EGL_RECORDABLE_ANDROID, 1,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val count = IntArray(1)
            check(EGL14.eglChooseConfig(dpy, attrs, 0, configs, 0, 1, count, 0) && count[0] > 0) { "Keine passende EGL-Konfiguration" }
            eglConfig = configs[0]
            val ctx = EGL14.eglCreateContext(dpy, eglConfig, EGL14.EGL_NO_CONTEXT, intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE), 0)
            check(ctx != null && ctx != EGL14.EGL_NO_CONTEXT) { "Kein OpenGL-ES-3-Kontext: 0x${Integer.toHexString(EGL14.eglGetError())}" }
            eglContext = ctx
            val pb = EGL14.eglCreatePbufferSurface(dpy, eglConfig, intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
            check(pb != null && pb != EGL14.EGL_NO_SURFACE) { "Keine Pbuffer-Fläche: 0x${Integer.toHexString(EGL14.eglGetError())}" }
            pbuffer = pb
            makeCurrent(pbuffer)
            val vaos = IntArray(1)
            GLES30.glGenVertexArrays(1, vaos, 0)
            vao = vaos[0]
            GLES30.glBindVertexArray(vao)
            GLES30.glClearColor(0f, 0f, 0f, 0f)
            copyOes = Program("copy_oes.frag")
            aiFrame = Program("ai_frame.frag")
            pack = Program("pack.frag")
            downOes = Program("down_oes.frag")
            down = Program("down.frag")
            weigh = Program("weigh.frag")
            gauss = Program("gauss.frag")
            composite = Program("composite.frag")
            createMaskTextures()
            val glVersion = GLES30.glGetString(GLES30.GL_VERSION) ?: "?"
            val extensions = GLES30.glGetString(GLES30.GL_EXTENSIONS) ?: ""
            floatPacking = glVersion.contains("OpenGL ES 3.2") || extensions.contains("GL_EXT_color_buffer_float")
            stats.gl(GLES30.glGetString(GLES30.GL_RENDERER) ?: "?", glVersion)
            checkGl("Start")
        } catch (e: Throwable) {
            try {
                EGL14.eglMakeCurrent(dpy, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                if (pbuffer != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(dpy, pbuffer)
                if (eglContext != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(dpy, eglContext)
                EGL14.eglTerminate(dpy)
            } catch (_: Throwable) {
            }
            pbuffer = EGL14.EGL_NO_SURFACE
            eglContext = EGL14.EGL_NO_CONTEXT
            eglConfig = null
            display = EGL14.EGL_NO_DISPLAY
            throw e
        }
    }

    private fun makeCurrent(surface: EGLSurface) {
        check(EGL14.eglMakeCurrent(display, surface, surface, eglContext)) { "eglMakeCurrent: Fehler 0x${Integer.toHexString(EGL14.eglGetError())}" }
    }

    /**
     * Ring für [requestedDelay] Bilder Verzögerung samt Zwischenstufen. Reicht der Grafikspeicher
     * nicht, geht es mit weniger Bildern weiter; klappt auch das nicht, ohne Unschärfe (steht im
     * Bericht).
     */
    private fun ensureRing(size: Size): Boolean {
        if (ringSize == size && ring.isNotEmpty()) return true
        if (ringFailedFor == size) return false
        freeRing()
        val (qw, qh) = quarter(size.width, size.height)
        val (ew, eh) = eighth(size.width, size.height)
        drainGlErrors()
        val q = allocate(qw, qh, GLES30.GL_RGBA8)
        // Der gewichtete Hintergrund braucht Halbfloat, sonst zeigen sich Stufen am Motivrand.
        var bg = allocate(ew, eh, GLES30.GL_RGBA16F)
        var tmp = allocate(ew, eh, GLES30.GL_RGBA16F)
        if (bg == null || tmp == null) {
            bg?.let { deleteTarget(it) }
            tmp?.let { deleteTarget(it) }
            bg = allocate(ew, eh, GLES30.GL_RGBA8)
            tmp = allocate(ew, eh, GLES30.GL_RGBA8)
            stats.note("Unschärfe mit 8 Bit statt Halbfloat")
        }
        if (q == null || bg == null || tmp == null) {
            listOfNotNull(q, bg, tmp).forEach { deleteTarget(it) }
            ringFailedFor = size
            stats.note("Kein Grafikspeicher für die Unschärfe: ohne Unschärfe weiter")
            return false
        }
        quarterTex = q.first
        quarterFbo = q.second
        bgTex = bg.first
        bgFbo = bg.second
        bgTmpTex = tmp.first
        bgTmpFbo = tmp.second
        smallW = ew
        smallH = eh
        val wanted = requestedDelay.coerceIn(1, MAX_DELAY)
        for (d in listOf(wanted, (wanted + 1) / 2, 2).distinct().filter { it <= wanted }) {
            val slots = ArrayList<Slot>()
            var ok = true
            for (i in 0..d) {
                val sharp = allocate(size.width, size.height, GLES30.GL_RGBA8)
                val small = if (sharp != null) allocate(ew, eh, GLES30.GL_RGBA8) else null
                if (sharp == null || small == null) {
                    sharp?.let { deleteTarget(it) }
                    ok = false
                    break
                }
                slots.add(Slot(sharp.first, sharp.second, small.first, small.second))
            }
            if (!ok) {
                for (s in slots) {
                    deleteTarget(s.tex to s.fbo)
                    deleteTarget(s.smallTex to s.smallFbo)
                }
                stats.note("Grafikspeicher reicht nicht für $d Bilder Verzögerung")
                continue
            }
            ring = slots
            freeSlots.clear()
            freeSlots.addAll(slots)
            queue.clear()
            delay = d
            ringSize = size
            stats.ring(d, slots.size, slots.size * (size.width.toLong() * size.height + ew.toLong() * eh) * 4)
            return true
        }
        freeScratch()
        ringFailedFor = size
        stats.note("Kein Grafikspeicher für die Verzögerung: ohne Unschärfe weiter")
        return false
    }

    /**
     * Textur mit Framebuffer; null, wenn das Format nicht geht oder der Grafikspeicher nicht
     * reicht. Das Löschen danach belegt den Speicher wirklich, so fällt ein Mangel gleich auf.
     */
    private fun allocate(w: Int, h: Int, internal: Int): Pair<Int, Int>? {
        val tex = texture(w, h, internal, linear = true)
        if (GLES30.glGetError() != GLES30.GL_NO_ERROR) {
            GLES30.glDeleteTextures(1, intArrayOf(tex), 0)
            drainGlErrors()
            return null
        }
        val fbo = try {
            framebuffer(tex)
        } catch (e: Throwable) {
            GLES30.glDeleteTextures(1, intArrayOf(tex), 0)
            drainGlErrors()
            return null
        }
        GLES30.glViewport(0, 0, w, h)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        if (GLES30.glGetError() != GLES30.GL_NO_ERROR) {
            deleteTarget(tex to fbo)
            drainGlErrors()
            return null
        }
        return tex to fbo
    }

    private fun deleteTarget(t: Pair<Int, Int>) {
        if (t.second != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(t.second), 0)
        if (t.first != 0) GLES30.glDeleteTextures(1, intArrayOf(t.first), 0)
    }

    private fun freeScratch() {
        deleteTarget(quarterTex to quarterFbo)
        deleteTarget(bgTex to bgFbo)
        deleteTarget(bgTmpTex to bgTmpFbo)
        quarterTex = 0
        quarterFbo = 0
        bgTex = 0
        bgFbo = 0
        bgTmpTex = 0
        bgTmpFbo = 0
    }

    private fun freeRing() {
        if (ring.isEmpty()) return
        for (s in ring) {
            deleteTarget(s.tex to s.fbo)
            deleteTarget(s.smallTex to s.smallFbo)
        }
        freeScratch()
        ring = emptyList()
        ringSize = null
        queue.clear()
        freeSlots.clear()
    }

    /** Ziele für die KI-Bilder; false, wenn sie sich nicht anlegen ließen (dann ohne KI-Bilder, steht im Bericht). */
    private fun ensureAiTargets(): Boolean {
        if (encFbo != 0 && depthFbo != 0) return true
        if (aiTargetsFailed) return false
        try {
            drainGlErrors()
            encTex = texture(FrameSlot.ENC, FrameSlot.ENC, GLES30.GL_RGBA8, linear = false)
            encFbo = framebuffer(encTex)
            depthTex = texture(FrameSlot.DEPTH, FrameSlot.DEPTH, GLES30.GL_RGBA8, linear = false)
            depthFbo = framebuffer(depthTex)
            checkGl("KI-Ziele")
        } catch (e: Throwable) {
            // Einmal scheitern reicht: nicht bei jedem Bild neu versuchen und Speicher verlieren.
            deleteTarget(encTex to encFbo)
            deleteTarget(depthTex to depthFbo)
            encTex = 0
            encFbo = 0
            depthTex = 0
            depthFbo = 0
            aiTargetsFailed = true
            drainGlErrors()
            stats.error("KI-Ziele", e)
            return false
        }
        if (floatPacking) {
            try {
                drainGlErrors()
                encPackTex = texture(encLayout.width, encLayout.height, GLES30.GL_RGBA32F, linear = false)
                checkGl("Packziel")
                encPackFbo = framebuffer(encPackTex)
                depthPackTex = texture(depthLayout.width, depthLayout.height, GLES30.GL_RGBA32F, linear = false)
                checkGl("Packziel")
                depthPackFbo = framebuffer(depthPackTex)
            } catch (e: Throwable) {
                // Gleitkomma-Rendern geht doch nicht: auf der CPU umrechnen.
                floatPacking = false
                deleteTarget(encPackTex to encPackFbo)
                deleteTarget(depthPackTex to depthPackFbo)
                encPackTex = 0
                encPackFbo = 0
                depthPackTex = 0
                depthPackFbo = 0
                drainGlErrors()
                stats.note("Gleitkomma-Rendern nicht möglich: ${e.message}")
            }
        }
        stats.aiTargets(floatPacking, (FrameSlot.ENC * FrameSlot.ENC + FrameSlot.DEPTH * FrameSlot.DEPTH) * 4L +
            if (floatPacking) (encLayout.width * encLayout.height + depthLayout.width * depthLayout.height) * 16L else 0L)
        return true
    }

    private fun createMaskTextures() {
        GLES30.glGenTextures(MASKS, maskTex, 0)
        val clear = ByteBuffer.allocateDirect(MASK_SIZE * MASK_SIZE)
        for (t in maskTex) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t)
            GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_R8, MASK_SIZE, MASK_SIZE)
            params(GLES30.GL_TEXTURE_2D, linear = true)
            clear.rewind()
            GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, MASK_SIZE, MASK_SIZE, GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, clear)
        }
        checkGl("Masken")
    }

    private fun createOesTexture(): Int {
        val t = IntArray(1)
        GLES30.glGenTextures(1, t, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, t[0])
        params(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, linear = true)
        checkGl("OES-Textur")
        return t[0]
    }

    private fun texture(w: Int, h: Int, internal: Int, linear: Boolean): Int {
        val t = IntArray(1)
        GLES30.glGenTextures(1, t, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t[0])
        GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, internal, w, h)
        params(GLES30.GL_TEXTURE_2D, linear)
        return t[0]
    }

    private fun params(target: Int, linear: Boolean) {
        val f = if (linear) GLES30.GL_LINEAR else GLES30.GL_NEAREST
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_MIN_FILTER, f)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_MAG_FILTER, f)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
    }

    private fun framebuffer(texture: Int): Int {
        val f = IntArray(1)
        GLES30.glGenFramebuffers(1, f, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, f[0])
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, texture, 0)
        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glDeleteFramebuffers(1, f, 0)
            throw IllegalStateException("Framebuffer unvollständig: 0x${Integer.toHexString(status)}")
        }
        return f[0]
    }

    private fun bindOes(unit: Int, texture: Int) {
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + unit)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texture)
    }

    private fun bind2d(unit: Int, texture: Int) {
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + unit)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
    }

    /** Ein Dreieck über den ganzen Ausschnitt (siehe fullscreen.vert). */
    private fun drawTriangle() {
        GLES30.glBindVertexArray(vao)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
    }

    private fun checkGl(where: String) {
        val e = GLES30.glGetError()
        if (e != GLES30.GL_NO_ERROR) {
            drainGlErrors()
            throw IllegalStateException("$where: GL-Fehler 0x${Integer.toHexString(e)}")
        }
    }

    private fun drainGlErrors() {
        repeat(8) { if (GLES30.glGetError() == GLES30.GL_NO_ERROR) return }
    }

    /** Ein Shader-Programm aus assets/shader mit gemerkten Uniform-Stellen. */
    private inner class Program(fragment: String) {
        private val id: Int
        private val locations = HashMap<String, Int>()

        init {
            val vs = compile(GLES30.GL_VERTEX_SHADER, "fullscreen.vert")
            val fs = compile(GLES30.GL_FRAGMENT_SHADER, fragment)
            val p = GLES30.glCreateProgram()
            GLES30.glAttachShader(p, vs)
            GLES30.glAttachShader(p, fs)
            GLES30.glLinkProgram(p)
            val ok = IntArray(1)
            GLES30.glGetProgramiv(p, GLES30.GL_LINK_STATUS, ok, 0)
            check(ok[0] != 0) { "$fragment bindet nicht: ${GLES30.glGetProgramInfoLog(p)}" }
            GLES30.glDeleteShader(vs)
            GLES30.glDeleteShader(fs)
            id = p
        }

        private fun compile(type: Int, name: String): Int {
            val src = context.assets.open("shader/$name").bufferedReader().use { it.readText() }
            val s = GLES30.glCreateShader(type)
            GLES30.glShaderSource(s, src)
            GLES30.glCompileShader(s)
            val ok = IntArray(1)
            GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
            check(ok[0] != 0) { "$name übersetzt nicht: ${GLES30.glGetShaderInfoLog(s)}" }
            return s
        }

        fun use() = GLES30.glUseProgram(id)

        private fun loc(name: String): Int = locations.getOrPut(name) { GLES30.glGetUniformLocation(id, name) }

        fun int(name: String, v: Int) = GLES30.glUniform1i(loc(name), v)
        fun float(name: String, v: Float) = GLES30.glUniform1f(loc(name), v)
        fun vec2(name: String, x: Float, y: Float) = GLES30.glUniform2f(loc(name), x, y)
        fun vec4(name: String, x: Float, y: Float, z: Float, w: Float) = GLES30.glUniform4f(loc(name), x, y, z, w)
        fun mat(name: String, m: FloatArray) = GLES30.glUniformMatrix4fv(loc(name), 1, false, m, 0)
        fun floats(name: String, v: FloatArray) = GLES30.glUniform1fv(loc(name), v.size, v, 0)
    }

    companion object {
        /**
         * 12 Bilder ≈ 400 ms: So lange braucht es, bis zu jedem Bild die KI-Masken davor und
         * danach da sind, bei 5 KI-Durchgängen pro Sekunde (Abstand 200 ms, Bildalter ≈ 100 ms,
         * Rechenzeit ≈ 120 ms; jedes Kamerabild kommt erst gut 50 ms nach seinem Zeitstempel hier
         * an). In 0.3.15 reichten 10 Bilder bei 7,5 Durchgängen pro Sekunde; seither rechnet die
         * KI wegen der Wärme seltener (siehe PipelineMath.minIntervalMs).
         */
        const val DEFAULT_DELAY = 12
        const val MAX_DELAY = 14
        /** Breite der Unschärfe in Pixeln des Achtel-Bildes (≈ 28 Pixel in 4K). */
        const val BLUR_SIGMA = 3.5
        /** Bis 0,4 s Abstand gilt eine Maske als aktuell; ältere lassen die Unschärfe ausklingen. */
        const val MAX_MASK_DISTANCE_NS = 400_000_000L
        /** So viel Unschärfe kommt je Bild höchstens dazu: von null bis voll in 8 Bildern. */
        const val BLUR_RAMP = 0.125f
        /** So viel früher, als die KI das nächste Bild nimmt, bereitet die Pipeline es vor. */
        const val LEAD_MS = 45.0
        const val MASKS = 8
        const val MAX_PENDING_MASKS = 4
        const val ENC_PACK_WIDTH = 768
        const val DEPTH_PACK_WIDTH = 1024
        /** Das 20. KI-Bild: Belichtung und Weißabgleich sind dann eingependelt. */
        const val THUMBNAIL_AFTER = 20
        const val RELEASE_TIMEOUT_MS = 3000L
        private const val INV255 = 1f / 255f
        private val FULL = floatArrayOf(0f, 0f, 1f, 1f)

        private fun quarter(w: Int, h: Int) = max(1, w / 4) to max(1, h / 4)
        private fun eighth(w: Int, h: Int) = max(1, w / 8) to max(1, h / 8)
    }
}

/** Zahlen der Grafik-Pipeline für den Bericht; von mehreren Threads benutzt. */
internal class PipelineStats {
    private val lock = Any()
    private val info = JSONObject()
    private val notes = JSONArray()
    private var framesIn = 0
    private var framesOut = 0
    private var blurredOut = 0
    private var withSubject = 0
    private var withoutMask = 0
    private var staleMask = 0
    private var mixed = 0
    private var gaps = 0
    private var lostEstimate = 0
    private var dropped = 0
    private var outputFailures = 0
    private var lastOutputFailure = ""
    private var lastIn = 0L
    private var lastOut = 0L
    private var aiFrames = 0
    private var aiFramesWithoutDepth = 0
    private var masksIn = 0
    private var outOfOrder = 0
    private var thumbnail: String? = null
    private val intervals = Series()
    private val frameMs = Series()
    private val outMs = Series()
    private val aiMs = Series()
    private val oneSidedMs = Series()
    private val maskGapMs = Series()
    // Abschnitt für den Verlauf (etwa alle 30 s).
    private val windowFrameMs = Series()
    private var windowOutputFailures = 0
    private var windowIn = 0
    private var windowOut = 0
    private var windowGaps = 0
    private var windowBlurred = 0
    private var windowMixed = 0
    private var windowWithMask = 0
    private var windowStale = 0

    private class Series {
        var values = DoubleArray(512)
        var size = 0
        fun add(v: Double) {
            if (v.isNaN()) return
            if (size == values.size) values = values.copyOf(size * 2)
            values[size++] = v
        }
        fun clear() {
            size = 0
        }
        fun json(): JSONObject? {
            if (size == 0) return null
            val t = Timing(values.copyOf(size))
            return JSONObject().put("median", r1(t.median)).put("p90", r1(t.p90)).put("max", r1(t.max))
        }
    }

    /**
     * Zähler von vorn, etwa beim Start der Aufnahme; was die Pipeline über Gerät und Lage weiß,
     * Hinweise und das Vorschaubild bleiben.
     */
    fun resetCounters() = synchronized(lock) {
        framesIn = 0
        framesOut = 0
        blurredOut = 0
        withSubject = 0
        withoutMask = 0
        staleMask = 0
        mixed = 0
        gaps = 0
        lostEstimate = 0
        dropped = 0
        outputFailures = 0
        lastIn = 0L
        lastOut = 0L
        aiFrames = 0
        aiFramesWithoutDepth = 0
        masksIn = 0
        outOfOrder = 0
        for (series in listOf(intervals, frameMs, outMs, aiMs, oneSidedMs, maskGapMs, windowFrameMs)) series.clear()
        clearWindow()
    }

    private fun clearWindow() {
        windowFrameMs.clear()
        windowOutputFailures = 0
        windowIn = 0
        windowOut = 0
        windowGaps = 0
        windowBlurred = 0
        windowMixed = 0
        windowWithMask = 0
        windowStale = 0
    }

    fun gl(renderer: String, version: String) = synchronized(lock) {
        info.put("gpu", renderer).put("opengl", version)
    }

    fun input(size: Size) = synchronized(lock) { info.put("eingang", "${size.width}x${size.height}") }

    fun rotation(degrees: Int, cropW: Int, cropH: Int, mirrored: Boolean) = synchronized(lock) {
        info.put("drehung", degrees).put("ausschnitt", "${cropW}x$cropH")
        if (mirrored) info.put("gespiegelt", true)
    }

    /** Lage des Bildes: Drehung in der Matrix der SurfaceTexture, Rest bis aufrecht, Größe des st-Raums. */
    fun orientation(stRotation: Int, stMirrored: Boolean, upright: Int, w: Int, h: Int) = synchronized(lock) {
        info.put("drehungInSt", stRotation).put("drehungBisAufrecht", upright).put("stRaum", "${w}x$h")
        if (stMirrored) info.put("stGespiegelt", true)
    }

    fun output(size: Size, targets: Int) = synchronized(lock) {
        val list = info.optJSONArray("ausgaben") ?: JSONArray().also { info.put("ausgaben", it) }
        val what = mutableListOf<String>()
        if (targets and CameraEffect.PREVIEW != 0) what += "Vorschau"
        if (targets and CameraEffect.VIDEO_CAPTURE != 0) what += "Video"
        if (targets and CameraEffect.IMAGE_CAPTURE != 0) what += "Foto"
        list.put("${size.width}x${size.height} (${what.joinToString(" + ")})")
    }

    fun ring(delay: Int, slots: Int, bytes: Long) = synchronized(lock) {
        info.put("verzoegerungBilder", delay).put("ringBilder", slots).put("ringMB", (bytes / 1e6).roundToInt())
    }

    fun aiTargets(floatPacking: Boolean, bytes: Long) = synchronized(lock) {
        info.put("kiBildAlsGleitkomma", floatPacking).put("kiZieleMB", (bytes / 1e6).roundToInt())
    }

    fun frameIn(ts: Long) = synchronized(lock) {
        framesIn++
        windowIn++
        if (lastIn != 0L) {
            val d = (ts - lastIn) / 1e6
            intervals.add(d)
            // Mehr als anderthalb Bildabstände (bei 30 fps) heißt: Bilder fehlen.
            if (d > GAP_MS) {
                gaps++
                windowGaps++
                lostEstimate += max(1, (d / FRAME_MS).roundToInt() - 1)
            }
        }
        lastIn = ts
    }

    /**
     * Zeit auf dem Pipeline-Thread für ein Bild, davon für die Ausgabe und fürs KI-Bild. Das ist
     * Uhrzeit, keine reine Rechenzeit: Warten auf die Grafikeinheit (Lesen des KI-Bildes) und auf
     * freie Puffer (Abgabe des Bildes) zählt mit. Dauert ein Bild länger als ein Bildabstand,
     * droht ein verlorenes Bild.
     */
    fun frameDone(totalMs: Double, outputMs: Double, aiFrameMs: Double) = synchronized(lock) {
        frameMs.add(totalMs)
        windowFrameMs.add(totalMs)
        outMs.add(outputMs)
        aiMs.add(aiFrameMs)
    }

    /**
     * Ein Bild ging hinaus. [blurred]: über den Weg mit Unschärfe (Ring und Masken); [subject]:
     * der Hintergrund wurde unscharf; [stale]: die nächste Maske war älter als 0,4 s; [gapNs]:
     * Abstand der beiden gemischten Masken (−1 = nur eine).
     */
    fun frameOut(ts: Long, pick: PipelineMath.MaskPick?, blurred: Boolean, subject: Boolean, stale: Boolean, gapNs: Long) = synchronized(lock) {
        framesOut++
        windowOut++
        if (lastOut != 0L && ts <= lastOut) outOfOrder++
        lastOut = ts
        if (!blurred) return@synchronized
        blurredOut++
        windowBlurred++
        if (subject) withSubject++
        if (pick == null) {
            withoutMask++
            return@synchronized
        }
        windowWithMask++
        if (stale) {
            staleMask++
            windowStale++
        }
        if (pick.a != pick.b) {
            mixed++
            windowMixed++
            if (gapNs >= 0) maskGapMs.add(gapNs / 1e6)
        } else {
            oneSidedMs.add(pick.distanceNs / 1e6)
        }
    }

    fun dropped() = synchronized(lock) { dropped++ }

    /** Ein Bild ließ sich nicht (in alle Ausgaben) abgeben; einmal je Bild gezählt. */
    fun outputFailed(what: String) = synchronized(lock) {
        outputFailures++
        windowOutputFailures++
        lastOutputFailure = what.take(200)
    }

    fun aiFrame(withDepth: Boolean) = synchronized(lock) {
        aiFrames++
        if (!withDepth) aiFramesWithoutDepth++
    }

    fun aiFrames(): Int = synchronized(lock) { aiFrames }

    fun maskIn() = synchronized(lock) { masksIn++ }

    fun thumbnail(base64: String) = synchronized(lock) { thumbnail = base64 }

    fun note(text: String) = synchronized(lock) {
        if (notes.length() < MAX_NOTES) notes.put(text.take(300))
    }

    fun error(where: String, e: Throwable) = note("$where: ${e.javaClass.simpleName}: ${e.message}")

    /** Ausgegebene Bilder bisher (für die Anzeige). */
    fun framesOut(): Int = synchronized(lock) { framesOut }

    /** Abschnitt für den Verlauf (etwa alle 30 s). */
    fun closeWindow(): JSONObject = synchronized(lock) {
        val o = JSONObject().put("bilderEin", windowIn).put("bilderAus", windowOut).put("luecken", windowGaps)
        windowFrameMs.json()?.let { o.put("jeBildMs", it) }
        if (windowOutputFailures > 0) o.put("ausgabeFehler", windowOutputFailures)
        if (windowWithMask > 0) {
            o.put("gemischtProzent", Math.round(windowMixed * 100.0 / windowWithMask))
            if (windowStale > 0) o.put("alteMaskeProzent", Math.round(windowStale * 100.0 / windowWithMask))
        }
        if (windowBlurred > windowWithMask) o.put("ohneMaske", windowBlurred - windowWithMask)
        clearWindow()
        o
    }

    fun toJson(): JSONObject = synchronized(lock) {
        val o = JSONObject(info.toString())
        o.put("bilderEin", framesIn).put("bilderAus", framesOut).put("luecken", gaps)
            .put("fehlendeBilderGeschaetzt", lostEstimate)
        if (dropped > 0) o.put("verworfen", dropped)
        if (outOfOrder > 0) o.put("zeitRueckwaerts", outOfOrder)
        if (outputFailures > 0) o.put("ausgabeFehler", outputFailures).put("letzterAusgabeFehler", lastOutputFailure)
        if (aiFrames > 0 || masksIn > 0) {
            o.put("kiBilder", aiFrames).put("masken", masksIn)
            if (aiFramesWithoutDepth > 0) o.put("kiBilderOhneTiefe", aiFramesWithoutDepth)
        }
        if (blurredOut > 0) {
            // mitUnschaerfeWeg: Bilder über Ring und Masken; mitMotiv: davon mit unscharfem
            // Hintergrund; ohneMaske: noch gar keine KI-Maske; mitAlterMaske: nächste Maske älter
            // als 0,4 s (Unschärfe klingt aus); gemischtProzent: Anteil mit Masken davor und danach.
            val withMask = blurredOut - withoutMask
            o.put("mitUnschaerfeWeg", blurredOut).put("mitMotiv", withSubject).put("ohneMaske", withoutMask)
                .put("mitAlterMaske", staleMask)
            if (withMask > 0) o.put("gemischtProzent", Math.round(mixed * 100.0 / withMask))
        }
        val ms = JSONObject()
        intervals.json()?.let { ms.put("bildabstand", it) }
        frameMs.json()?.let { ms.put("jeBild", it) }
        outMs.json()?.let { ms.put("ausgabe", it) }
        aiMs.json()?.let { ms.put("kiBild", it) }
        // Nur eine Maske in der Nähe: wie weit sie vom Bild weg war. Zwei Masken: wie weit sie auseinanderlagen.
        oneSidedMs.json()?.let { ms.put("maskeEinseitig", it) }
        maskGapMs.json()?.let { ms.put("maskenLuecke", it) }
        o.put("ms", ms)
        if (notes.length() > 0) o.put("hinweise", JSONArray(notes.toString()))
        thumbnail?.let { o.put("kiBildJpegBase64", it) }
        o
    }

    private companion object {
        const val FRAME_MS = 1000.0 / 30.0
        const val GAP_MS = FRAME_MS * 1.5
        const val MAX_NOTES = 12
    }
}
