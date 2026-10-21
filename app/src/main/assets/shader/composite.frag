#version 300 es
// Fertiges Bild: das scharfe Kamerabild, wo die Maske das Motiv zeigt, sonst die Unschärfe aus
// reinem Hintergrund (siehe weigh.frag). Die Maske für diesen Zeitpunkt mischt die KI-Masken
// davor (A) und danach (B).
precision highp float;

uniform sampler2D uSharp;     // Kamerabild im st-Raum (Ring)
uniform sampler2D uBg;        // Hintergrund verkleinert, gewichtet und unscharf: (rgb · w, w)
uniform sampler2D uSmall;     // Kamerabild verkleinert, für Stellen ganz ohne Hintergrund in der Nähe
uniform sampler2D uMaskA;     // Maske vor diesem Bild: Wahrscheinlichkeit 0 bis 1, Zeile 0 oben
uniform sampler2D uMaskB;     // Maske nach diesem Bild
uniform mat4 uOutToBuf;       // Ausgabe-Koordinate → st-Raum
uniform mat4 uBufToMask;      // st-Raum → aufrechte Maskenkoordinate
uniform float uMix;           // 0 = nur A, 1 = nur B
uniform vec2 uPresent;        // je Maske: 1 = Motiv gefunden, 0 = keins (dann bleibt alles scharf)
uniform float uStrength;      // 0 = keine Unschärfe, 1 = volle (schon mit dem Alter der Maske verrechnet)

in vec2 vUv;
out vec4 outColor;

void main() {
    vec2 b = (uOutToBuf * vec4(vUv, 0.0, 1.0)).xy;
    vec3 sharp = texture(uSharp, b).rgb;
    // Normierte Faltung: Summe der gewichteten Farben durch die Summe der Gewichte. Wo fast kein
    // Hintergrund in der Nähe ist (tief im Motiv), übernimmt das verkleinerte Bild; dort ist die
    // Unschärfe ohnehin aus.
    vec4 bg = texture(uBg, b);
    vec3 soft = (bg.rgb + 0.002 * texture(uSmall, b).rgb) / (bg.a + 0.002);
    vec2 m = (uBufToMask * vec4(b, 0.0, 1.0)).xy;
    float pa = mix(1.0, texture(uMaskA, m).r, uPresent.x);
    float pb = mix(1.0, texture(uMaskB, m).r, uPresent.y);
    float w = (1.0 - smoothstep(0.3, 0.7, mix(pa, pb, uMix))) * uStrength;
    outColor = vec4(mix(sharp, soft, w), 1.0);
}
