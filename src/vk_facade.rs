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
pub enum VkFacadeError {
    #[error("Vulkan context: {0}")]
    VkContext(#[from] VkContextError),
}

#[derive(Debug, Error)]
pub enum VkContextError {
    #[error("CString creation failed")]
    CStringCreation,
    #[error("Vulkan instance creation failed")]
    VkInstanceCreation,
    #[error("Graphics device creation failed")]
    DeviceCreation,
    #[error("Could not enumerate physical devices")]
    PhysicalDeviceEnumeration,
    #[error("No physical device with transfer and computation queue")]
    NoSuitablePhysicalDevice,
}

pub struct VkContext {
    vk_instance: Instance,
    device: Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    compute_queue_family_index: u32,
    compute_command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    command_pool_permanent: vk::CommandPool,
    command_pool_short_lived: vk::CommandPool,
}

impl VkContext {
    pub fn new(app_name: &str, app_version: u32) -> Result<VkContext, VkContextError> {
        let vk_instance = create_vk_instance(
            &CString::new(app_name).map_err(|_| VkContextError::CStringCreation)?,
            app_version,
        )?;
        let (physical_device, compute_queue_family_index) = select_physical_device(&vk_instance)?;
        let device = create_device(&vk_instance, physical_device, compute_queue_family_index)?;
        let memory_properties =
            unsafe { vk_instance.get_physical_device_memory_properties(physical_device) };
        // for now, we default to queue zero; in the future, we should do a more thoughtful
        // selection; probably round-robin at least?
        let compute_queue = unsafe { device.get_device_queue(compute_queue_family_index, 0) };
        Ok(VkContext { vk_instance, device, memory_properties, compute_queue_family_index })
    }

    pub fn create_descriptor_pool(self) -> Result<vk::DescriptorPool, Error> {
        let descriptor_pool_size_storage = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(10); // bound by max_sets, but in principle this can be lower
        let descriptor_pool_size_uniform = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(10); // bound by max_sets, but in principle this can be lower
        let descriptor_pool_sizes = [descriptor_pool_size_storage, descriptor_pool_size_uniform];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(10); // if multiple compute shaders are active operating on multiple objects,
                          // we need a descriptor set for every such shader / object
        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .map_err(|_| Error::GpuError(String::from("Creating the descriptor pool failed.")))
        }
    }
}

pub struct 

fn create_vk_instance(app_name: &CStr, app_version: u32) -> Result<Instance, VkContextError> {
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
            .map_err(|_| VkContextError::VkInstanceCreation)
    }
}

fn select_physical_device(instance: &Instance) -> Result<(vk::PhysicalDevice, u32), VkContextError> {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|_| VkContextError::PhysicalDeviceEnumeration)?
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
                            Some((*pd, idx as u32))
                        } else {
                            None
                        }
                    })
            })
            .ok_or(VkContextError::NoSuitablePhysicalDevice)
    }
}

fn create_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<Device, VkContextError> {
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0]);

    let queue_infos = &[queue_info];
    let device_create_info = vk::DeviceCreateInfo::default().queue_create_infos(queue_infos);
    unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .map_err(|_| VkContextError::DeviceCreation)
    }
}
