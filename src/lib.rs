mod device_loading;
mod workload;
mod shader;

pub use device_loading::Dispatcher;
pub use workload::{DeviceTransferable, DeviceVariable, Kernel, KernelArgInfo};
pub use shader::*;

use ash::{Device, Entry, Instance, vk};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use thiserror::Error;

const ENGINE_NAME: &CStr = c"Vulkan Computing Toolkit";
const ENGINE_VERSION: u32 = 0;
const API_MAJOR: u32 = 0;
const API_MINOR: u32 = 1;
const API_PATCH: u32 = 3;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Creating vulkan instance failed: {0}")]
    VkInstanceCreation(vk::Result),
    #[error("Creating logical device failed: {0}")]
    DeviceCreation(vk::Result),
    #[error("Enumerating physical devices failed: {0}")]
    PhysicalDeviceEnumeration(vk::Result),
    #[error("No physical device with transfer and computation queue found")]
    NoSuitablePhysicalDevice,
    #[error("Creating the descriptor pool failed: {0}")]
    Creation(vk::Result),
}

#[derive(Debug, Error)]
pub enum ComputeContextError {
    #[error("CString creation failed")]
    CStringCreation,
    #[error("Creating descriptor set layout failed: {0}")]
    DescriptorSetLayoutCreation(vk::Result),
    #[error("Creating pipeline failed: {0}")]
    PipelineCreation(vk::Result),
    #[error("Creating pipeline layout failed: {0}")]
    PipelineLayoutCreation(vk::Result),
    #[error("Creating command pool failed: {0}")]
    CmdPoolCreation(vk::Result),
    #[error("Allocating descriptor set failed: {0}")]
    DescriptorSetAllocation(vk::Result),
    #[error("Freeing descriptor set failed: {0}")]
    FreeingDescriptorSet(vk::Result),
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] Error),
}

#[derive(Clone)]
pub struct ComputeContext {
    dev: Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    comp_queue_fam_idx: u32,
    comp_queue_fam_props: vk::QueueFamilyProperties,
    descr_pool: vk::DescriptorPool,
}

impl ComputeContext {
    const DESCRIPTOR_COUNT: u32 = 10;

    pub fn new(app_name: &str, app_version: u32) -> Result<ComputeContext, ComputeContextError> {
        let vk_instance = create_vk_instance(
            &CString::new(app_name).map_err(|_| ComputeContextError::CStringCreation)?,
            app_version,
        )?;
        let (physical_device, comp_queue_fam_idx, comp_queue_fam_props) =
            select_physical_device(&vk_instance)?;
        let dev = create_logical_device(
            &vk_instance,
            physical_device,
            comp_queue_fam_idx,
            comp_queue_fam_props.queue_count,
        )?;
        let mem_props =
            unsafe { vk_instance.get_physical_device_memory_properties(physical_device) };
        let descr_pool = create_descriptor_pool(&dev, Self::DESCRIPTOR_COUNT)?;
        Ok(ComputeContext {
            dev,
            mem_props,
            comp_queue_fam_idx,
            comp_queue_fam_props,
            descr_pool,
        })
    }

    fn create_command_pool(
        &self,
        flags: vk::CommandPoolCreateFlags,
    ) -> Result<vk::CommandPool, ComputeContextError> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(flags)
            .queue_family_index(self.comp_queue_fam_idx);
        unsafe {
            self.dev
                .create_command_pool(&command_pool_create_info, None)
        }
        .map_err(|e| ComputeContextError::CmdPoolCreation(e))
    }

    fn create_descriptor_set_layout(
        &self,
        bindings: &[u32],
        descriptor_types: &[vk::DescriptorType],
    ) -> Result<vk::DescriptorSetLayout, ComputeContextError> {
        let descriptor_set_layout_bindings = bindings
            .iter()
            .zip(descriptor_types.iter())
            .map(|(b, d)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(*b)
                    .descriptor_type(*d)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<vk::DescriptorSetLayoutBinding>>();
        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);
        unsafe {
            self.dev
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .map_err(|e| ComputeContextError::DescriptorSetLayoutCreation(e))
        }
    }

    fn allocate_descriptor_set(
        &self,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, ComputeContextError> {
        let layouts = [descriptor_set_layout];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descr_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe {
            self.dev
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
        }
        .map_err(|e| ComputeContextError::DescriptorSetAllocation(e))?;

        Ok(descriptor_sets[0])
    }

    fn free_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
    ) -> Result<(), ComputeContextError> {
        unsafe {
            self.dev
                .free_descriptor_sets(self.descr_pool, &[descriptor_set])
        }
        .map_err(|e| ComputeContextError::FreeingDescriptorSet(e))?;

        Ok(())
    }

    fn create_pipeline(
        &self,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        shader_module: vk::ShaderModule,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), ComputeContextError> {
        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(descriptor_set_layouts);
        let pipeline_layout = unsafe {
            self.dev
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .map_err(|e| ComputeContextError::PipelineLayoutCreation(e))?
        };

        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let pipeline_create_infos = [vk::ComputePipelineCreateInfo::default()
            .layout(pipeline_layout)
            .stage(shader_stage_create_info)];
        let pipelines = unsafe {
            self.dev
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .map_err(|(_, e)| ComputeContextError::PipelineCreation(e))?
        };

        unsafe {
            self.dev.destroy_shader_module(shader_module, None);
        }

        Ok((pipelines[0], pipeline_layout))
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        unsafe {
            self.dev.destroy_descriptor_pool(self.descr_pool, None);
        }
    }
}

fn create_vk_instance(app_name: &CStr, app_version: u32) -> Result<Instance, Error> {
    let layer_names: Vec<*const c_char> = [c"VK_LAYER_KHRONOS_validation"]
        .iter()
        .map(|name| name.as_ptr())
        .collect();

    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(app_version)
        .engine_name(ENGINE_NAME)
        .engine_version(ENGINE_VERSION)
        .api_version(vk::make_api_version(API_MAJOR, API_MINOR, API_PATCH, 0));

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_names);

    let entry = Entry::linked();
    unsafe {
        entry
            .create_instance(&create_info, None)
            .map_err(|e| Error::VkInstanceCreation(e))
    }
}

fn select_physical_device(
    instance: &Instance,
) -> Result<(vk::PhysicalDevice, u32, vk::QueueFamilyProperties), Error> {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| Error::PhysicalDeviceEnumeration(e))?
    };
    unsafe {
        physical_devices
            .iter()
            .find_map(|pd| {
                instance
                    .get_physical_device_queue_family_properties(*pd)
                    .iter()
                    .enumerate()
                    .find_map(|(idx, info)| {
                        if info
                            .queue_flags
                            .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
                        {
                            Some((*pd, idx as u32, *info))
                        } else {
                            None
                        }
                    })
            })
            .ok_or(Error::NoSuitablePhysicalDevice)
    }
}

fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    queue_count: u32,
) -> Result<Device, Error> {
    let queue_prios = vec![1.0; queue_count as usize];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_prios);

    let queue_infos = &[queue_info];
    let device_create_info = vk::DeviceCreateInfo::default().queue_create_infos(queue_infos);
    unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .map_err(|e| Error::DeviceCreation(e))
    }
}

fn find_memory_type_index(
    buffer_required_memory_type: u32,
    requested_memory_properties: &vk::MemoryPropertyFlags,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Option<u32> {
    for i in 0..device_memory_properties.memory_type_count {
        if buffer_required_memory_type & (1 << i) != 0
            && device_memory_properties.memory_types[i as usize]
                .property_flags
                .intersects(*requested_memory_properties)
        {
            return Some(i);
        }
    }
    None
}

fn create_descriptor_pool(dev: &Device, desc_count: u32) -> Result<vk::DescriptorPool, Error> {
    let descriptor_pool_size_storage = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(desc_count); // bound by max_sets, but in principle this can be lower
    let descriptor_pool_size_uniform = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(desc_count); // bound by max_sets, but in principle this can be lower
    let descriptor_pool_sizes = [descriptor_pool_size_storage, descriptor_pool_size_uniform];
    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&descriptor_pool_sizes)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(desc_count); // if multiple compute shaders are active operating on multiple objects,
    // we need a descriptor set for every such shader / object
    let descriptor_pool = unsafe {
        dev.create_descriptor_pool(&descriptor_pool_create_info, None)
            .map_err(|e| Error::Creation(e))?
    };
    Ok(descriptor_pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
    impl<T: Copy> DeviceTransferable for Vec<T> {
        fn cpy_to(&self, ptr: *mut std::ffi::c_void, offset: usize) {
            let mem_typed = unsafe {
                std::slice::from_raw_parts_mut::<T>(ptr.add(offset).cast(), self.len() as usize)
            };
            mem_typed.copy_from_slice(self);
        }

        fn cpy_from(&mut self, ptr: *const std::ffi::c_void, offset: usize) {
            let mem_typed = unsafe {
                std::slice::from_raw_parts::<T>(ptr.add(offset).cast(), self.len() as usize)
            };
            self.copy_from_slice(mem_typed);
        }

        fn size(&self) -> usize {
            std::mem::size_of::<T>() * self.len()
        }
    }
    */

    #[test]
    fn smoke_cctx_creation() {
        ComputeContext::new("App", 0).expect("Compute context creation failed");
    }

    #[test]
    fn smoke_kernel_creation_inv_shader() {
        let cctx = ComputeContext::new("App", 0).expect("Compute context creation failed");
        let shader_dummy = &[0u32];
        let data = vec![1.0, 2.0, 3.0];
        let arg_infos = vec![KernelArgInfo {
            is_uniformly_readonly: true,
            size: (std::mem::size_of::<f32>() * data.len())
                .try_into()
                .expect("Cast from usize to u64 not possible"),
        }];
        if let Ok(_kernel) = Kernel::new(&cctx, shader_dummy, arg_infos) {
            panic!("Why is the shader valid?");
        }
    }

    #[test]
    fn smoke_kernel_creation_valid_shader() {
        let cctx = ComputeContext::new("App", 0).expect("Compute context creation failed");
        let shader_bytes = std::fs::read("data/shader/add_one.spv").expect("could not load shader");
        let mut shader = Vec::<u32>::new();
        for bytes in shader_bytes.chunks_exact(4) {
            shader.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }
        let data = vec![1.0, 2.0, 3.0];
        let arg_infos = vec![KernelArgInfo {
            is_uniformly_readonly: true,
            size: (std::mem::size_of::<f32>() * data.len())
                .try_into()
                .expect("Cast from usize to u64 not possible"),
        }];
        Kernel::new(&cctx, &shader, arg_infos).expect("Kernel creation failed");
    }

    #[test]
    fn smoke_submit_upload() {
        let cctx = ComputeContext::new("App", 0).expect("Compute context creation failed");
        let shader_bytes = std::fs::read("data/shader/add_one.spv").expect("could not load shader");
        let mut shader = Vec::<u32>::new();
        for bytes in shader_bytes.chunks_exact(4) {
            shader.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }
        let float_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let uint_data: Vec<u32> = vec![float_data.len().try_into().expect("too large for u32")];
        let arg_infos = vec![
            KernelArgInfo {
                is_uniformly_readonly: true,
                size: std::mem::size_of::<u32> as u64,
            },
            KernelArgInfo {
                is_uniformly_readonly: false,
                size: (std::mem::size_of::<f32>() * float_data.len())
                    .try_into()
                    .expect("Cast from usize to u64 not possible"),
            },
        ];
        Kernel::new(&cctx, &shader, arg_infos.clone()).expect("Kernel creation failed");

        let len = DeviceVariable::builder()
            .host_to_dev_transf(true)
            .is_uniformly_read(true)
            .size(std::mem::size_of::<u32>() as u64)
            .build(&cctx)
            .expect("build len device variable failed");
        let float_array = DeviceVariable::builder()
            .host_to_dev_transf(true)
            .is_uniformly_read(false)
            .size((float_data.len() * std::mem::size_of::<f32>()) as u64)
            .build(&cctx)
            .expect("building device variable failed");

        let mut dispatcher = Dispatcher::new(&cctx, 3).expect("Dispatcher creation failed");
        let variables = [len, float_array];
        dispatcher
            .upload_async(&variables, &[&uint_data, &float_data], 0)
            .expect("upload submission failed");

        dispatcher
            .sync_stream(0)
            .expect("Waiting on stream to finish its work has been cancelled unexpectedly");
    }

    #[test]
    fn smoke_full_loop() {
        let cctx = ComputeContext::new("App", 0).expect("Compute context creation failed");
        let shader_bytes = std::fs::read("data/shader/add_one.spv").expect("could not load shader");
        let mut shader = Vec::<u32>::new();
        for bytes in shader_bytes.chunks_exact(4) {
            shader.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }
        let mut float_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut uint_data: Vec<u32> = vec![float_data.len().try_into().expect("too large for u32")];
        let arg_infos = vec![
            KernelArgInfo {
                is_uniformly_readonly: true,
                size: std::mem::size_of::<u32> as u64,
            },
            KernelArgInfo {
                is_uniformly_readonly: false,
                size: (std::mem::size_of::<f32>() * float_data.len())
                    .try_into()
                    .expect("Cast from usize to u64 not possible"),
            },
        ];
        let kernel =
            Kernel::new(&cctx, &shader, arg_infos.clone()).expect("Kernel creation failed");

        let len = DeviceVariable::builder()
            .host_to_dev_transf(true)
            .dev_to_host_transf(true)
            .is_uniformly_read(true)
            .size(std::mem::size_of::<u32>() as u64)
            .build(&cctx)
            .expect("build len device variable failed");
        let float_array = DeviceVariable::builder()
            .host_to_dev_transf(true)
            .dev_to_host_transf(true)
            .is_uniformly_read(false)
            .size((float_data.len() * std::mem::size_of::<f32>()) as u64)
            .build(&cctx)
            .expect("building device variable failed");

        let mut dispatcher = Dispatcher::new(&cctx, 3).expect("Dispatcher creation failed");
        let variables = [len, float_array];
        let host_variables: &[&dyn DeviceTransferable] = &[&uint_data, &float_data];

        dispatcher
            .upload_async(&variables, host_variables, 0)
            .expect("upload submission failed");

        dispatcher
            .launch_async(&kernel, &variables, 0, [1, 1, 1])
            .expect("launch submission failed");

        uint_data[0] = 0;
        let host_variables: &mut [&mut dyn DeviceTransferable] =
            &mut [&mut uint_data, &mut float_data];
        dispatcher
            .download_sync(&variables, host_variables, 0)
            .expect("upload submission failed");

        assert_eq!(uint_data[0], 3);
        assert_eq!(float_data, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn two_kernels() {
        let cctx = ComputeContext::new("App", 0).expect("Compute context creation failed");
        let shader_bytes = std::fs::read("data/shader/add_one.spv").expect("could not load shader");
        let mut shader = Vec::<u32>::new();
        for bytes in shader_bytes.chunks_exact(4) {
            shader.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }

        let mut float_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut uint_data: Vec<u32> = vec![float_data.len().try_into().expect("too large for u32")];
        let arg_infos = vec![
            KernelArgInfo {
                is_uniformly_readonly: true,
                size: std::mem::size_of::<u32>() as u64,
            },
            KernelArgInfo {
                is_uniformly_readonly: false,
                size: (std::mem::size_of::<f32>() * float_data.len())
                    .try_into()
                    .expect("Cast from usize to u64 not possible"),
            },
        ];
        let kernel_1 =
            Kernel::new(&cctx, &shader, arg_infos.clone()).expect("Kernel creation failed");
        let kernel_2 =
            Kernel::new(&cctx, &shader, arg_infos.clone()).expect("Kernel creation failed");

        let len = DeviceVariable::builder()
            .size(arg_infos[0].size)
            .is_uniformly_read(true)
            .build(&cctx)
            .expect("build len device variable failed");
        let float_array = DeviceVariable::builder()
            .size(arg_infos[1].size)
            .build(&cctx)
            .expect("building float_array device variable failed");

        let dev_vars = [len, float_array];
        let host_vars: [&dyn DeviceTransferable; 2] = [&uint_data, &float_data];
        let mut dispatcher = Dispatcher::new(&cctx, 3).expect("Dispatcher creation failed");

        dispatcher
            .upload_async(&dev_vars, &host_vars, 0)
            .expect("upload submission failed");

        dispatcher
            .launch_async(&kernel_1, &dev_vars, 0, [1, 1, 1])
            .expect("launch submission failed");
        dispatcher
            .launch_async(&kernel_2, &dev_vars, 0, [1, 1, 1])
            .expect("launch submission failed");

        uint_data[0] = 0;
        let mut host_variables: [&mut dyn DeviceTransferable; 2] = [&mut uint_data, &mut float_data];
        dispatcher
            .download_sync(&dev_vars, &mut host_variables, 0)
            .expect("upload submission failed");

        assert_eq!(uint_data[0], 3);
        assert_eq!(float_data, [3.0, 4.0, 5.0]);
    }
}
