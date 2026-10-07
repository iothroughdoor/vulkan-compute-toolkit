mod device_loading;
mod workload;

pub use device_loading::Dispatcher;
pub use workload::{DeviceTransferable, Kernel, KernelArgInfo, KernelResourceManager};

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
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] Error),
}

#[derive(Clone)]
pub struct ComputeContext {
    vk_instance: Instance,
    dev: Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    comp_queue_fam_idx: u32,
    comp_queue_fam_props: vk::QueueFamilyProperties,
}

impl ComputeContext {
    pub fn new(app_name: &str, app_version: u32) -> Result<ComputeContext, ComputeContextError> {
        let vk_instance = create_vk_instance(
            &CString::new(app_name).map_err(|_| ComputeContextError::CStringCreation)?,
            app_version,
        )?;
        let (physical_device, compute_queue_family_index, compute_queue_family_props) =
            select_physical_device(&vk_instance)?;
        let device =
            create_logical_device(&vk_instance, physical_device, compute_queue_family_index)?;
        let memory_properties =
            unsafe { vk_instance.get_physical_device_memory_properties(physical_device) };
        // for now, we default to queue zero; in the future, we should do a more thoughtful
        // selection; probably round-robin at least?
        //let compute_queue = unsafe { device.get_device_queue(compute_queue_family_index, 0) };
        Ok(ComputeContext {
            vk_instance,
            dev: device,
            mem_props: memory_properties,
            comp_queue_fam_idx: compute_queue_family_index,
            comp_queue_fam_props: compute_queue_family_props,
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

    fn create_pipeline(
        &self,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        shader_module: vk::ShaderModule,
    ) -> Result<vk::Pipeline, ComputeContextError> {
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
            self.dev.destroy_pipeline_layout(pipeline_layout, None);
        }

        Ok(pipelines[0])
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
) -> Result<Device, Error> {
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0]);

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

#[cfg(test)]
mod tests {
    use super::*;

    struct KernelArgFloatVec {
        data: Vec<f32>,
    }

    impl DeviceTransferable for KernelArgFloatVec {
        fn cpy_to(&self, ptr: *mut std::ffi::c_void, offset: usize) {
            let mem_typed = unsafe {
                std::slice::from_raw_parts_mut::<f32>(
                    ptr.add(offset).cast(),
                    self.data.len() as usize,
                )
            };
            mem_typed.copy_from_slice(&self.data);
        }

        fn cpy_from(&mut self, ptr: *const std::ffi::c_void, offset: usize) {
            let mem_typed = unsafe {
                std::slice::from_raw_parts::<f32>(ptr.add(offset).cast(), self.data.len() as usize)
            };
            self.data.copy_from_slice(mem_typed);
        }
    }

    #[test]
    fn smoke() {
        let cctx = ComputeContext::new("App", 0).expect("Compute context creation failed");
        let res_mgr =
            KernelResourceManager::new(&cctx, 10).expect("Resource Manager creation failed");
        let shader_dummy = &[0u32];
        let arg = KernelArgFloatVec {
            data: vec![1.0, 2.0, 3.0],
        };
        let arg_infos = vec![KernelArgInfo {
            is_uniformly_readonly: true,
            size: (std::mem::size_of::<f32>() * arg.data.len())
                .try_into()
                .expect("Cast from usize to u64 not possible"),
        }];
        if let Ok(_kernel) = Kernel::new(&cctx, &res_mgr, shader_dummy, arg_infos) {
            panic!("Why is the shader valid?");
        }
    }
}
