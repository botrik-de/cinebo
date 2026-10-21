#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
// Kamerabild unverändert übernehmen: in den Ring (uTex = Matrix der SurfaceTexture, dann ist
// vUv die Pufferkoordinate) oder direkt in die Ausgabe (uTex = Matrix der Ausgabe).
precision highp float;

uniform samplerExternalOES uCam;
uniform mat4 uTex;

in vec2 vUv;
out vec4 outColor;

void main() {
    vec2 tc = (uTex * vec4(vUv, 0.0, 1.0)).xy;
    outColor = vec4(texture(uCam, tc).rgb, 1.0);
}
