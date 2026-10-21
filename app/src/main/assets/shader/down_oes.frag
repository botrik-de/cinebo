#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
// Kamerabild um den Faktor 4 verkleinern (4 × 4 Pixel gemittelt) für die Unschärfe.
// vUv ist die Pufferkoordinate des Zielpixels; uStep ist ein Quellpixel in Pufferkoordinaten.
precision highp float;

uniform samplerExternalOES uCam;
uniform mat4 uTex;
uniform vec2 uStep;

in vec2 vUv;
out vec4 outColor;

vec3 tap(vec2 b) {
    return texture(uCam, (uTex * vec4(b, 0.0, 1.0)).xy).rgb;
}

void main() {
    // Jeder bilineare Abgriff auf einer Pixelkante mittelt 2 × 2 Pixel; vier davon ergeben 4 × 4.
    vec3 s = tap(vUv + vec2(-1.0, -1.0) * uStep) + tap(vUv + vec2(1.0, -1.0) * uStep)
           + tap(vUv + vec2(-1.0, 1.0) * uStep) + tap(vUv + vec2(1.0, 1.0) * uStep);
    outColor = vec4(s * 0.25, 1.0);
}
