#version 300 es
// Ein Dreieck, das den ganzen Bildausschnitt bedeckt; ohne Eckpunkt-Puffer.
// vUv läuft über den sichtbaren Bereich von 0 bis 1 (unten links 0, 0).
out vec2 vUv;

void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
