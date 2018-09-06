#version 330

uniform vec4 diffuseColor;
uniform vec4 lightPosAndSide;
uniform vec4 lightColor;
uniform vec4 emissionColor;
uniform vec4 lightNormal;
uniform sampler2D shadowMap;

out vec4 outColor;
in vec4 quadColor;
in vec4 vertPosition;
in vec4 lightSpacePosition;
in vec4 vertNormal;

vec4 saturate(vec4 v) {
    return max(min(v, vec4(1)), vec4(0));
}

void main() {
    vec3 lightProj = lightSpacePosition.xyz / lightSpacePosition.w;
    lightProj = lightProj * 0.5 + 0.5;

    vec3 lightPoint = vertPosition.xyz;
    lightPoint.y = max(lightPosAndSide.y, lightPoint.y);
    lightPoint.xz = min(max(lightPoint.xz, lightPosAndSide.xz), lightPosAndSide.xz + lightPosAndSide.w);
    vec3 L = normalize(lightPoint - vertPosition.xyz);
    vec4 diffuse = pow(diffuseColor, vec4(2.2)) * max(dot(L, normalize(vertNormal.xyz)), 0) * max(dot(-L, lightNormal.xyz), 0);
    int samplesCount = 4;
    float lighting = samplesCount * samplesCount;
    if (lightProj.x < 1 - 1e-3 && lightProj.x > 1e-3 && lightProj.y < 1 - 1e-3 && lightProj.y > 1e-3)
    {
        for (int i = 0; i < samplesCount; ++i) {
            for (int j = 0; j < samplesCount; ++j) {
                vec2 bias = vec2(i, j) / 1024;
                if (texture(shadowMap, lightProj.xy + bias).x < lightProj.z - 1e-2) {
                    lighting -= 1;
                }
            }
        }
    }
    lighting /= samplesCount * samplesCount;
	outColor = pow(saturate(quadColor + diffuse * lighting * 0.000001 + emissionColor * 0.000001), vec4(1 / 2.2));
}