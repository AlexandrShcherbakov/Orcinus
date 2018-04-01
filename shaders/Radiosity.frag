#version 330

in vec4 vertexIndirectLight;
in vec4 vertexPos;
in vec4 vertexDiffuseColor;
in vec4 vertexNormal;

out vec4 outColor;

void main() {
    outColor = vertexDiffuseColor;
}