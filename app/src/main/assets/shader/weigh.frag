#version 300 es
// Hintergrund für die Unschärfe ohne Lichthof: das verkleinerte Kamerabild, gewichtet mit seinem
// Anteil Hintergrund (1 − Motiv) laut der Maske dieses Zeitpunkts, vormultipliziert (rgb · w, w).
// Gauß darüber und Teilen durch das mitverschmierte Gewicht (composite.frag) ergibt eine
// Unschärfe nur aus Hintergrund: Die Farben des Motivs laufen nicht in den Hintergrund aus.
precision highp float;

uniform sampler2D uSmall;     // Kamerabild verkleinert, st-Raum
uniform sampler2D uMaskA;     // Maske vor diesem Bild: Wahrscheinlichkeit 0 bis 1, Zeile 0 oben
uniform sampler2D uMaskB;     // Maske nach diesem Bild
uniform mat4 uBufToMask;      // st-Raum → aufrechte Maskenkoordinate
uniform float uMix;           // 0 = nur A, 1 = nur B
uniform vec2 uPresent;        // je Maske: 1 = Motiv gefunden, 0 = keins

in vec2 vUv;
out vec4 outColor;

void main() {
    vec2 m = (uBufToMask * vec4(vUv, 0.0, 1.0)).xy;
    float pa = mix(1.0, texture(uMaskA, m).r, uPresent.x);
    float pb = mix(1.0, texture(uMaskB, m).r, uPresent.y);
    float w = 1.0 - smoothstep(0.3, 0.7, mix(pa, pb, uMix));
    outColor = vec4(texture(uSmall, vUv).rgb * w, w);
}
