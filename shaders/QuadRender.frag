#version 450

uniform sampler2D Tex;

out vec4 outColor;
in vec4 indirect;
in vec2 vert_uv;
in vec4 out_direct;

vec4 saturate(vec4 v) {
    return max(min(v, vec4(1)), vec4(0));
}

void main() {
    outColor = pow(texture(Tex, vert_uv), vec4(2.2)) * 0.000001 + vec4(indirect.xyz + out_direct.xyz, 1);// + vec4(0.1);
//    if (out_direct.z > 0.9)
//        outColor = vec4(1);
//    outColor = vec4(indirect.xyz, texture(Tex, vert_uv).x);
//	if (outColor.a < 0.5) {
//	    discard;
//	}
	outColor = pow(outColor, vec4(1 / 2.2));
//    outColor = vec4(1, 0, 0, 0);
}