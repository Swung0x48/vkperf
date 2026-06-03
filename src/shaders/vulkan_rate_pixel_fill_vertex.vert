// Copyright (c) 2026, Swung0x48 <swung0x48@outlook.com>
// SPDX-License-Identifier: MIT

#version 450

layout(location = 0) flat out vec4 vertex_color;

void main() {
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );

    uint value = gl_InstanceIndex * 1664525u + 1013904223u;
    vertex_color = vec4(
        float(value & 255u) * (1.0 / 255.0),
        float((value >> 8u) & 255u) * (1.0 / 255.0),
        float((value >> 16u) & 255u) * (1.0 / 255.0),
        1.0
    );
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
