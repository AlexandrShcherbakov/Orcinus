#version 450

layout(location = 0) in vec4 pos;
layout(location = 1) in vec4 direct;
layout(location = 2) in vec2 uv;

uniform mat4 CameraMatrix;

out vec2 vert_uv;
out vec4 indirect;
out vec4 out_direct;

layout(std430, binding = 0) buffer layoutName
{
	vec4 indirect_light[];
};

void main() {
	gl_Position = CameraMatrix * pos;
	vert_uv = uv;
	indirect = indirect_light[gl_VertexID / 4];
	out_direct = direct;
}