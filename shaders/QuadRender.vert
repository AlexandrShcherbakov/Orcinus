#version 330

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 color;
layout(location = 2) in vec4 normal;

uniform mat4 CameraMatrix;
out vec4 quadColor;
out vec4 vertPosition;
out vec4 vertNormal;

void main() {
	gl_Position = CameraMatrix * pos;
	vertPosition = pos;
	vertNormal = normal;
	quadColor = color;
}