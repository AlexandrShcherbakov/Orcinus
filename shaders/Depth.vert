#version 330

layout(location = 0) in vec4 pos;

uniform mat4 cameraMatrix;

void main() {
	gl_Position = cameraMatrix * (pos + vec4(0, 0, 0, 0));
}