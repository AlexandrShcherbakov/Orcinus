#version 330

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 uv;

uniform mat4 CameraMatrix;

out vec4 quadColor;
out vec2 vert_uv;

void main() {
	gl_Position = CameraMatrix * pos;
	quadColor = color;
	vert_uv = uv;
}