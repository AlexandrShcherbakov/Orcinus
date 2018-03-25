#version 330

in vec4 vertexIndirectLight;
in vec4 vertexPos;
in vec4 vertexDiffuseColor;
in vec4 vertexSpecularColor;
in vec4 vertexNormal;

out vec4 outColor;

const vec4 lightPosition = vec4(0, 2.9, -1.02213e-005, 1);
const vec4 lightDirection = vec4(0, -1, 4.49701e-006, 0);
const vec3 lightColor = vec3(1, 1, 1);
const float multiplier = 5;
const float innerCone = 45;
const float outerCone = 60;

uniform vec4 cameraPosition;

void main() {
    vec4 lightToPoint = normalize(vertexPos - lightPosition);
    vec4 pointToEye = normalize(cameraPosition - vertexPos);
    float currentCos = dot(lightToPoint, normalize(lightDirection));
    float innerCos = cos(radians(innerCone / 2));
    float outerCos = cos(radians(outerCone / 2));
    float coneMultiplier = clamp((currentCos - outerCos) / (innerCos - outerCos), 0, 1);
    float lambertTerm = max(dot(normalize(vertexNormal), -lightToPoint), 0);
    vec3 incomeLight = lightColor * coneMultiplier;
    vec4 diffuseColor = vertexDiffuseColor * lambertTerm / 3.14;
    float specularTerm = pow(dot(pointToEye, reflect(lightToPoint, normalize(vertexNormal))), vertexSpecularColor.w);
    vec4 specularColor = specularTerm * vertexSpecularColor;
    vec4 directLight = vec4(incomeLight, 1) * (diffuseColor + specularColor);
    outColor = pow(directLight + vertexIndirectLight, vec4(1 / 2.2));
}