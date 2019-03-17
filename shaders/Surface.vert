#version 330

layout(location = 0) in vec4 point;
layout(location = 1) in vec4 color;

out vec4 vertexColor;

uniform mat4 CameraMatrix;

void main() {
    vertexColor = color;
	gl_Position = CameraMatrix * point;
}
