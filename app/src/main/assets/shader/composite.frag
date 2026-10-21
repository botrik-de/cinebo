#version 300 es
// Fertiges Bild: scharfes Kamerabild, wo die Maske das Motiv zeigt, sonst die unscharfe
// Fassung. Die Maske für diesen Zeitpunkt mischt die KI-Masken davor (A) und danach (B).
precision highp float;

uniform sampler2D uSharp;     // Kamerabild in Pufferausrichtung (Ring)
uniform sampler2D uBlur;      // dasselbe Bild verkleinert und unscharf
uniform sampler2D uMaskA;     // Maske vor diesem Bild: Wahrscheinlichkeit 0 bis 1, Zeile 0 oben
uniform sampler2D uMaskB;     // Maske nach diesem Bild
uniform mat4 uOutToBuf;       // Ausgabe-Koordinate → Pufferkoordinate
uniform mat4 uBufToMask;      // Pufferkoordinate → aufrechte Maskenkoordinate
uniform float uMix;           // 0 = nur A, 1 = nur B
uniform vec2 uPresent;        // je Maske: 1 = Objekt gefunden, 0 = keins (dann bleibt alles scharf)
uniform float uStrength;      // 0 = keine Unschärfe, 1 = volle

in vec2 vUv;
out vec4 outColor;

void main() {
    vec2 b = (uOutToBuf * vec4(vUv, 0.0, 1.0)).xy;
    vec3 sharp = texture(uSharp, b).rgb;
    vec3 soft = texture(uBlur, b).rgb;
    vec2 m = (uBufToMask * vec4(b, 0.0, 1.0)).xy;
    float pa = mix(1.0, texture(uMaskA, m).r, uPresent.x);
    float pb = mix(1.0, texture(uMaskB, m).r, uPresent.y);
    float p = mix(pa, pb, uMix);
    float w = (1.0 - smoothstep(0.3, 0.7, p)) * uStrength;
    outColor = vec4(mix(sharp, soft, w), 1.0);
}
