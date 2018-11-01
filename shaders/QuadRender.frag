#version 330

uniform sampler2D Tex;

out vec4 outColor;
in vec4 quadColor;
in vec2 vert_uv;

vec4 saturate(vec4 v) {
    return max(min(v, vec4(1)), vec4(0));
}

void main() {
	outColor = quadColor;//pow(saturate(quadColor), vec4(1 / 2.2));
	outColor = texture(Tex, vert_uv);
	if (outColor.a < 0.5) {
	    discard;
	}
}