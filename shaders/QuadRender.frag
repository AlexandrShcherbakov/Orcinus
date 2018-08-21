#version 330

uniform vec4 diffuseColor;
uniform vec4 lightPosAndSide;
uniform vec4 lightColor;
uniform vec4 emissionColor;
uniform vec4 lightNormal;

out vec4 outColor;
in vec4 quadColor;
in vec4 vertPosition;
in vec4 vertNormal;

vec4 saturate(vec4 v) {
    return max(min(v, vec4(1)), vec4(0));
}

void main() {
    vec3 lightPoint = vertPosition.xyz;
    lightPoint.y = max(lightPosAndSide.y, lightPoint.y);
    lightPoint.xz = min(max(lightPoint.xz, lightPosAndSide.xz), lightPosAndSide.xz + lightPosAndSide.w);
    vec3 L = normalize(lightPoint - vertPosition.xyz);
    vec4 diffuse = pow(diffuseColor, vec4(2.2)) * max(dot(L, normalize(vertNormal.xyz)), 0) * max(dot(-L, lightNormal.xyz), 0);
	outColor = pow(saturate(quadColor + diffuse + emissionColor), vec4(1 / 2.2));
}