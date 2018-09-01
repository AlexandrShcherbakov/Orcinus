#version 330

out vec4 outColor;
in vec4 quadColor;

vec4 saturate(vec4 v) {
    return max(min(v, vec4(1)), vec4(0));
}

void main() {
	outColor = pow(saturate(quadColor), vec4(1 / 2.2));
}