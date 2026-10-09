// CONDITION: gl_WorkGroupSize.x / gl_SubgroupSize <= gl_SubgroupSize
// meaning we only need two reduction steps
#version 450
#pragma shader_stage(compute)
#extension GL_KHR_shader_subgroup_shuffle_relative : enable

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform UBO {
    uint len;
};
layout(std430, binding = 1) buffer SSBO_1 {
    int vec[];
};
layout(std430, binding = 2) buffer SSBO_2 {
    int sum;
};

shared int scratch_array[gl_WorkGroupSize.x];

void main()
{
    // load into shared mem
    scratch_array[gl_LocalInvocationIndex] = vec[gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex];

    // reduction tree
    int val = 0;
    for (uint delta = gl_SubgroupSize/2; delta > 0; delta /= 2) {
        val = scratch_array[gl_LocalInvocationIndex];
        scratch_array[gl_LocalInvocationIndex] += subgroupShuffleDown(val, delta);
    }

    memoryBarrierShared();

    // first subgroup collects
    if (gl_LocalInvocationIndex < gl_SubgroupSize && gl_LocalInvocationIndex > 0) {
        for (uint i = 1; i < gl_WorkGroupSize.x / gl_SubgroupSize; i++) {
            scratch_array[i] = scratch_array[i * gl_SubgroupSize];
        }
    }

    memoryBarrierShared();

    if (gl_LocalInvocationIndex < gl_SubgroupSize) {
        for (uint delta = gl_WorkGroupSize.x / gl_SubgroupSize / 2; delta > 0; delta /= 2) {
            val = scratch_array[gl_LocalInvocationIndex];
            scratch_array[gl_LocalInvocationIndex] += subgroupShuffleDown(val, delta);
        }
    }

    memoryBarrierShared();

    // leader writes back
    if (gl_LocalInvocationIndex == 0) {
        atomicAdd(sum, scratch_array[0]);
    }
}

