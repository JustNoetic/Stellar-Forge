#version 460
layout(push_constant) uniform Push { vec4 color; vec4 parameters; } pc;
void main() {
    vec2 position = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2)*2.0-1.0;
    gl_Position = vec4(position, pc.parameters.x+position.x*pc.parameters.y, 1.0);
}
