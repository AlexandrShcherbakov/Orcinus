#version 450

uniform sampler2D Tex;

out vec4 outColor;
in vec4 indirect;
in vec2 vert_uv;

vec4 saturate(vec4 v) {
    return max(min(v, vec4(1)), vec4(0));
}

void main() {
    outColor = pow(texture(Tex, vert_uv), vec4(2.2)) * vec4(indirect.xyz, 1);
	if (outColor.a < 0.5) {
	    discard;
	}
	outColor = pow(outColor, vec4(1 / 2.2));
}