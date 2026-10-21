#version 300 es
// Gaußsche Unschärfe in einer Richtung, für alle vier Kanäle (auch das Gewicht in alpha, siehe
// weigh.frag). 13 Pixel breit, über 7 bilineare Abgriffe: Mitte plus je drei Paare, deren
// Gewichte und Abstände die App vorab ausrechnet (uWeight, uOffset).
precision highp float;
precision highp sampler2D;   // Halbfloat-Texturen mit voller Genauigkeit lesen

uniform sampler2D uSrc;
uniform vec2 uDir;          // ein Pixel in Abtastrichtung
uniform float uWeight[4];   // [0] Mitte, [1..3] Paare
uniform float uOffset[4];   // Abstand der Paare in Pixeln ([0] bleibt 0)

in vec2 vUv;
out vec4 outColor;

void main() {
    vec4 c = texture(uSrc, vUv) * uWeight[0];
    for (int i = 1; i < 4; i++) {
        vec2 d = uDir * uOffset[i];
        c += (texture(uSrc, vUv + d) + texture(uSrc, vUv - d)) * uWeight[i];
    }
    outColor = c;
}
