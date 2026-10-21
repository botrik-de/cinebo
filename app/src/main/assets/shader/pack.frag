#version 300 es
// Packt ein quadratisches RGBA8-Bild als Gleitkommazahlen RGB hintereinander (NHWC, 0 bis 1):
// Texel x, y der Ausgabe hält die Zahlen 4·(y·uWidth + x) bis +3. Nach glReadPixels(GL_FLOAT)
// liegt im Speicher genau der Eingang des Modells.
precision highp float;
precision highp int;

uniform highp sampler2D uSrc;   // Zeile 0 = oberste Bildzeile
uniform int uSize;              // Kantenlänge des Quadrats
uniform int uWidth;             // Breite der Ausgabe in Texeln
uniform int uTotal;             // Anzahl der Zahlen: uSize · uSize · 3

out vec4 outColor;

float channel(int f) {
    if (f >= uTotal) return 0.0;
    int p = f / 3;
    int c = f - p * 3;
    vec4 v = texelFetch(uSrc, ivec2(p % uSize, p / uSize), 0);
    return c == 0 ? v.r : (c == 1 ? v.g : v.b);
}

void main() {
    ivec2 t = ivec2(gl_FragCoord.xy);
    int base = (t.y * uWidth + t.x) * 4;
    outColor = vec4(channel(base), channel(base + 1), channel(base + 2), channel(base + 3));
}
