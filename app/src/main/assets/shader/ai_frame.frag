#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
// Bild für die KI: aufrecht, auf die Zielgröße verkleinert, im Inhaltsrechteck (sonst schwarz).
// Zeile 0 des Ziels ist die oberste Bildzeile; so liegt sie nach glReadPixels vorn im Speicher.
// Verkleinert wird mit mehreren bilinearen Abgriffen je Zielpixel (wie ein Box-Filter über den
// Fußabdruck), damit beim Verkleinern aus 4K keine Muster entstehen.
precision highp float;
precision highp int;

uniform samplerExternalOES uCam;
uniform mat4 uUprightToTex;   // aufrechte Koordinate (u nach rechts, v nach unten, 0 bis 1) → Textur
uniform vec4 uContent;        // Inhalt im Ziel: links, oben, rechts, unten (0 bis 1)
uniform vec2 uTarget;         // Zielgröße in Pixeln
uniform vec2 uSource;         // Quellbild aufrecht in Pixeln (Breite, Höhe)

out vec4 outColor;

void main() {
    vec2 f = gl_FragCoord.xy / uTarget;
    if (f.x < uContent.x || f.x > uContent.z || f.y < uContent.y || f.y > uContent.w) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec2 size = uContent.zw - uContent.xy;
    vec2 uv = (f - uContent.xy) / size;
    // Quellpixel je Zielpixel; so viele Abgriffe je Achse (höchstens 8).
    vec2 footprint = uSource / (uTarget * size);
    ivec2 n = clamp(ivec2(ceil(footprint - 0.001)), ivec2(1), ivec2(8));
    vec2 stepUv = footprint / vec2(n) / uSource;
    vec2 origin = uv - 0.5 * stepUv * vec2(n - ivec2(1));
    vec3 sum = vec3(0.0);
    for (int j = 0; j < n.y; j++) {
        for (int i = 0; i < n.x; i++) {
            vec2 p = origin + stepUv * vec2(float(i), float(j));
            sum += texture(uCam, (uUprightToTex * vec4(p, 0.0, 1.0)).xy).rgb;
        }
    }
    outColor = vec4(sum / float(n.x * n.y), 1.0);
}
