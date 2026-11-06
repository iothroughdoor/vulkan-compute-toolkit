#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(binding=0) uniform UBO {
    uint len;
};

layout(std430, binding=1) buffer SSBO {
    float in_vec[];
};


void main() 
{
    debugPrintfEXT("the global incovation id is %i\n", gl_GlobalInvocationID.x);
    if (gl_GlobalInvocationID.x < len) {
        in_vec[gl_GlobalInvocationID.x] += 1;
    }
}
