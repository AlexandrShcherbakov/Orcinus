#version 330

in vec4 vertexIndirectLight;
in vec4 vertexPos;
in vec4 vertexDiffuseColor;
in vec4 vertexSpecularColor;

out vec4 outColor;

const vec4 lightPosition = vec4(0, 2.9, -1.02213e-005, 1);
const vec4 lightDirection = vec4(0, -1, 4.49701e-006, 0);
const vec3 lightColor = vec3(1, 1, 1);
const float multiplier = 5;
const float innerCone = 45;
const float outerCone = 60;

void main() {
    float currentCos = dot(normalize(vertexPos - lightPosition), normalize(lightDirection));
    float innerCos = cos(radians(innerCone / 2));
    float outerCos = cos(radians(outerCone / 2));
    float coneMultiplier = clamp((currentCos - outerCos) / (innerCos - outerCos), 0, 1);
    vec3 incomeLight = lightColor * coneMultiplier;
    vec4 directLight = vec4(incomeLight, 1) * vertexDiffuseColor;
    outColor = directLight + vertexIndirectLight;
}