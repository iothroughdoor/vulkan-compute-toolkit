# Vulkan Computing Toolkit
A toolkit aiming at making Vulkan a little simpler to use for GPGPU computations.

This project is in its early stages as indicated by its version.

## Dependencies 
- vulkan driver 
- GLSL compiler
- vulkan-devel, vulkan-validation-layers packages of your OS

## Principles
1) The concepts of vulkan are thoroughly designed, we should not invent our own. We merely want
to bundle certain standard steps into macro steps / concepts such that they are faster (and easier)
to use.
2) We profile before we optimize.

## Constraints on shaders
- Shader code for this toolkit MUST have constant_ids for the local group size exposed via
```GLSL 
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
```
in order for the toolkit being able to set the local group size.
