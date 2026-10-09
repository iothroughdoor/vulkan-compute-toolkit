#version 450
#pragma shader_stage(compute)

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding=0) uniform UBO {
    uint len;
};

layout(std430, binding=1) buffer SSBO {
    float in_vec[];
};


void main() 
{
    if (gl_GlobalInvocationID.x < len) {
        in_vec[gl_GlobalInvocationID.x] += 1;
    }
}
