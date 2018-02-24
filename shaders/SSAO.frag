#version 330

uniform sampler2D DepthTexture;
uniform sampler2D RandomPlanes;
uniform int ScreenWidth;
uniform int ScreenHeight;

in vec4 vertNormal;

out vec4 outColor;

const int RANDOM_PLANES_TEXTURE_SIDE = 4;
const int SAMPLES_COUNT = 8;

void main() {
    const vec4 samples[SAMPLES_COUNT] = vec4[SAMPLES_COUNT](
        vec4(-0.5, -0.5, -0.5, 0),
        vec4(-0.5, -0.5,  0.5, 0),
        vec4(-0.5,  0.5, -0.5, 0),
        vec4(-0.5,  0.5,  0.5, 0),
        vec4( 0.5, -0.5, -0.5, 0),
        vec4( 0.5, -0.5,  0.5, 0),
        vec4( 0.5,  0.5, -0.5, 0),
        vec4( 0.5,  0.5,  0.5, 0)
    );

    const float zFar = 10000, zNear = 0.0001;
    const float radius = 10;
    vec2 screenSize = vec2(ScreenWidth, ScreenHeight);
    float depth = texture(DepthTexture, gl_FragCoord.xy / screenSize).x;
    float z = zFar * zNear / (depth * (zFar - zNear) - zFar);
    float visibleSamples = 0;
    ivec2 randomPlaneindex = ivec2(gl_FragCoord.xy) % RANDOM_PLANES_TEXTURE_SIDE;
    vec3 randomPlane = texture(RandomPlanes, vec2(randomPlaneindex) / RANDOM_PLANES_TEXTURE_SIDE).xyz;

    for (int i = 0; i < SAMPLES_COUNT; ++i) {
        vec3 reflectedSample = reflect(samples[i].xyz, randomPlane);
        float sampleDepth = texture(DepthTexture, (gl_FragCoord.xy + reflectedSample.xy * radius) / screenSize).x;
        float zSample = zFar * zNear / (sampleDepth * (zFar - zNear) - zFar);

        visibleSamples += float(zSample <= z);
    }
    outColor = vec4(visibleSamples / SAMPLES_COUNT) + 0.3;
}
