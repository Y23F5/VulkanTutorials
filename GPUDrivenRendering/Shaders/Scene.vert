#version 460
#extension GL_EXT_scalar_block_layout : enable

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view;
    mat4 proj;
} camera;

struct CullingDatum {
    vec3 center;
    vec3 halfExtent;
};

layout(set = 0, binding = 1, scalar) readonly buffer CullingData {
    CullingDatum data[];
} culling;

layout(set = 0, binding = 2, scalar) readonly buffer RenderData {
    vec4 colors[];
} render;

layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec3 outColor;

void main() {
    uint idx = gl_InstanceIndex;
    CullingDatum d = culling.data[idx];
    vec3 scale = d.halfExtent * 2.0;
    vec3 worldPos = inPosition * scale + d.center;
    gl_Position = camera.proj * camera.view * vec4(worldPos, 1.0);
    outColor = render.colors[idx].rgb;
}
