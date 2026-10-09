# Vulkan Computing Toolkit

## Dependencies 
- vulkan driver 
- vulkan-devel, vulkan-validation-layers packages of your OS

## Principles
1) The concepts of vulkan are thoroughly designed, we should not invent our own. We merely want
to bundle certain standard steps into macro steps / concepts such that they are faster (and easier)
to use.
2) We profile before we optimize
