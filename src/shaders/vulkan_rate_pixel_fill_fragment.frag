// Copyright (c) 2026, Swung0x48 <swung0x48@outlook.com>
// SPDX-License-Identifier: MIT

#version 450

layout(location = 0) flat in vec4 fragment_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = fragment_color;
}
