#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() 
{
    debugPrintfEXT("Hello World!\nMy global invocation ID is (%i, %i, %i)\n"
                 , gl_GlobalInvocationID.x
                 , gl_GlobalInvocationID.y
                 , gl_GlobalInvocationID.z);
}
