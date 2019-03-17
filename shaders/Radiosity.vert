#version 330

layout(location = 0) in vec4 point;
layout(location = 1) in vec4 indirectLight;
layout(location = 2) in vec4 diffuseColor;
layout(location = 3) in vec4 normal;

out vec4 vertexIndirectLight;
out vec4 vertexPos;
out vec4 vertexDiffuseColor;
out vec4 vertexNormal;

uniform mat4 CameraMatrix;

void main() {
    vertexIndirectLight = indirectLight;
    vertexDiffuseColor = diffuseColor;
    vertexNormal = normal;
    vertexPos = point;
	gl_Position = CameraMatrix * point;
}
