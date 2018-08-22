#version 330

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 color;
layout(location = 2) in vec4 normal;

uniform mat4 CameraMatrix;
uniform mat4 shadowMapMatrix;

out vec4 quadColor;
out vec4 vertPosition;
out vec4 vertNormal;
out vec4 lightSpacePosition;

void main() {
	gl_Position = CameraMatrix * pos;
	vertPosition = pos;
	vertNormal = normal;
	quadColor = color;
	lightSpacePosition = shadowMapMatrix * pos;
}