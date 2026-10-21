#version 300 es
// Gaußsche Unschärfe in einer Richtung. 13 Pixel breit, über 7 bilineare Abgriffe: Mitte plus
// je drei Paare, deren Gewichte und Abstände der App vorab ausrechnet (uWeight, uOffset).
precision highp float;

uniform sampler2D uSrc;
uniform vec2 uDir;          // ein Pixel in Abtastrichtung
uniform float uWeight[4];   // [0] Mitte, [1..3] Paare
uniform float uOffset[4];   // Abstand der Paare in Pixeln ([0] bleibt 0)

in vec2 vUv;
out vec4 outColor;

void main() {
    vec3 c = texture(uSrc, vUv).rgb * uWeight[0];
    for (int i = 1; i < 4; i++) {
        vec2 d = uDir * uOffset[i];
        c += (texture(uSrc, vUv + d).rgb + texture(uSrc, vUv - d).rgb) * uWeight[i];
    }
    outColor = vec4(c, 1.0);
}
