#version 330

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 color;

uniform mat4 CameraMatrix;

out vec4 quadColor;

void main() {
	gl_Position = CameraMatrix * pos;
	quadColor = color;
}