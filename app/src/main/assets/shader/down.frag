#version 300 es
// Um den Faktor 2 verkleinern: ein bilinearer Abgriff in der Mitte mittelt 2 × 2 Pixel.
precision highp float;

uniform sampler2D uSrc;

in vec2 vUv;
out vec4 outColor;

void main() {
    outColor = vec4(texture(uSrc, vUv).rgb, 1.0);
}
