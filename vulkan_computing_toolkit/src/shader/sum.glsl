#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(binding=0) uniform UBO {
    uint len;
};

layout(std430, binding=1) buffer SSBO_1 {
    float in_vec_1[];
};

layout(std430, binding=2) buffer SSBO_2 {
    float in_vec_2[];
};


void main() 
{
    if (gl_GlobalInvocationID.x < len) {
        in_vec_1[gl_GlobalInvocationID.x] += in_vec_2[gl_GlobalInvocationID.x];
    }
}
