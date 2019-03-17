#version 330

layout(location = 0) in vec4 point;

uniform mat4 CameraMatrix;

void main() {
	gl_Position = CameraMatrix * point;
}