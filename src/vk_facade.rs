use ash::{Device, Entry, Instance, prelude::VkResult, vk};
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
    ComputeContext(#[from] ComputeContextError),
}

#[derive(Debug, Error)]
pub enum ComputeContextError {
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

#[derive(Clone)]
pub struct ComputeContext {
    vk_instance: Instance,
    device: Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    compute_queue_family_index: u32,
    compute_queue_family_props: vk::QueueFamilyProperties,
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
            device,
            memory_properties,
            compute_queue_family_index,
            compute_queue_family_props,
        })
    }
}

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("Pipeline creation failed")]
    PipelineCreation,
    #[error("Shader module creation failed")]
    ShaderModuleCreation,
    #[error("Descriptor set layout creation failed")]
    DescriptorSetLayoutCreation,
    #[error("Pipeline layout creation failed")]
    PipelineLayoutCreation,
    #[error("Kernel Resource Manager Error: {0}")]
    KernelResourceManager(#[from] KernelResourceManagerError),
    #[error("Invalid resource specification")]
    InvalidResourceSpec,
    #[error("Unsupported descriptor type")]
    UnsupportedDescriptorType,
}

pub struct Kernel<'a> {
    device: Device,
    res_mgr: &'a KernelResourceManager,

    pipeline: vk::Pipeline,
    descriptor_set: vk::DescriptorSet,
    buffers: Vec<vk::Buffer>,
    memory: Vec<vk::DeviceMemory>,
}

pub struct KernelArgInfo {
    is_writeable: bool,
    size: u64,
}

pub trait DeviceTransferable {
    fn cpy_from(&mut self, memory: *const core::ffi::c_void, offset: usize);
    fn cpy_to(&self, memory: *mut core::ffi::c_void, offset: usize);
}

impl<'a> Kernel<'a> {
    pub fn new(
        cctx: &ComputeContext,
        res_mgr: &'a KernelResourceManager,
        shader_spirv: &[u32],
        arg_infos: Vec<KernelArgInfo>,
    ) -> Result<Self, KernelError> {
        let binding_nrs = arg_infos
            .iter()
            .enumerate()
            .map(|(i, _)| i.try_into().map_err(|_| KernelError::InvalidResourceSpec))
            .collect::<Result<Vec<u32>, KernelError>>()?;
        let descriptor_types = arg_infos
            .iter()
            .map(|&info| {
                if info.is_writeable {
                    vk::DescriptorType::STORAGE_BUFFER
                } else {
                    vk::DescriptorType::UNIFORM_BUFFER
                }
            })
            .collect::<Vec<vk::DescriptorType>>();

        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(shader_spirv);
        let shader_module = unsafe {
            cctx.device
                .create_shader_module(&shader_module_create_info, None)
                .map_err(|_| KernelError::ShaderModuleCreation)
        }?;
        let descriptor_set_layout =
            create_descriptor_set_layout(&cctx.device, &binding_nrs, &descriptor_types)?;
        let pipeline = create_pipeline(&cctx.device, &[descriptor_set_layout], shader_module)?;

        let mut buffers_with_memory: (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) = (vec![], vec![]);
        let memory_prop_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        for (i, &dt) in descriptor_types.iter().enumerate() {
            let mut buffer_usage_flags = vk::BufferUsageFlags::TRANSFER_DST;
            match dt {
                vk::DescriptorType::UNIFORM_BUFFER => {
                    buffer_usage_flags |= vk::BufferUsageFlags::UNIFORM_BUFFER
                }
                vk::DescriptorType::STORAGE_BUFFER => {
                    buffer_usage_flags |= vk::BufferUsageFlags::STORAGE_BUFFER
                }
                /* :TODO: Consider the other descriptor types: are they relevant for compute
                kernels? IMAGE and TEXEL Buffers might be. What about samplers? */
                _ => return Err(KernelError::UnsupportedDescriptorType),
            }
            let (buffer, buffer_memory) = create_buffer_with_memory(
                cctx,
                buffer_usage_flags,
                memory_prop_flags,
                arg_infos[i].size,
            )
            .map_err(|e| KernelError::KernelResourceManager(e))?;
            buffers_with_memory.0.push(buffer);
            buffers_with_memory.1.push(buffer_memory);
        }

        let descriptor_set = res_mgr
            .allocate_descriptor_set(descriptor_set_layout)
            .map_err(|e| KernelError::KernelResourceManager(e))?;

        bind_descriptor_set_to_buffer(
            &cctx.device,
            descriptor_set,
            &binding_nrs,
            &descriptor_types,
            &buffers_with_memory.0,
        )?;

        unsafe {
            cctx.device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
        }

        Ok(Kernel {
            device: cctx.device.clone(),
            res_mgr: &res_mgr,
            pipeline,
            descriptor_set,
            buffers: buffers_with_memory.0,
            memory: buffers_with_memory.1,
        })
    }
}

impl<'a> Drop for Kernel<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.res_mgr.free_descriptor_set(self.descriptor_set);
            for buffer in std::mem::take(&mut self.buffers) {
                self.device.destroy_buffer(buffer, None);
            }
            for mem in std::mem::take(&mut self.memory) {
                self.device.free_memory(mem, None);
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum KernelResourceManagerError {
    #[error("Creating the buffer pool failed")]
    Creation,
    #[error("Creation of buffer failed")]
    BufferCreation,
    #[error(
        "Requirements of buffer on memory is incompatible with the memory types the device offers"
    )]
    BufferMemoryIncompatibility,
    #[error("Device memory allocation failed")]
    DeviceMemoryAllocation,
    #[error("Binding memory to buffer failed")]
    BindingMemoryToBuffer,
    #[error("Allocating descriptor set failed")]
    DescriptorSetAllocation,
}

pub struct KernelResourceManager {
    device: Device,
    descriptor_pool: vk::DescriptorPool,
}

impl KernelResourceManager {
    pub fn new(cctx: &ComputeContext, desc_count: u32) -> Result<Self, KernelResourceManagerError> {
        let descriptor_pool_size_storage = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(desc_count); // bound by max_sets, but in principle this can be lower
        let descriptor_pool_size_uniform = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(desc_count); // bound by max_sets, but in principle this can be lower
        let descriptor_pool_sizes = [descriptor_pool_size_storage, descriptor_pool_size_uniform];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(desc_count); // if multiple compute shaders are active operating on multiple objects,
        // we need a descriptor set for every such shader / object
        let descriptor_pool = unsafe {
            cctx.device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .map_err(|_| KernelResourceManagerError::Creation)?
        };
        Ok(KernelResourceManager {
            device: cctx.device.clone(),
            descriptor_pool,
        })
    }

    fn allocate_descriptor_set(
        &self,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, KernelResourceManagerError> {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .map_err(|_| KernelResourceManagerError::DescriptorSetAllocation)?
        };
        Ok(descriptor_sets[0])
    }

    fn free_descriptor_set(&self, descriptor_set: vk::DescriptorSet) {
        unsafe {
            self.device
                .free_descriptor_sets(self.descriptor_pool, &[descriptor_set]);
        }
    }
}

impl Drop for KernelResourceManager {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

#[derive(Debug, Error)]
pub enum DispatcherError {
    #[error("Creation of the command pool failed")]
    CommandPoolCreation,
    #[error("Allocating the command buffers failed")]
    CommandBufferAllocation,
    #[error("Staging buffer creation failed")]
    StagingBufferCreation,
    #[error("Staging buffer not supported by the device")]
    StagingBufferNotSupported,
    #[error("Allocation of staging memory failed")]
    StagingMemoryAllocation,
    #[error("Binding staging buffer to memory failed")]
    StagingBufferBinding,
    #[error("Mapping staging memory failed.")]
    MappingStagingMemory,
    #[error("Cast could not be carried out.")]
    Casting,
    #[error("Starting cmd buffer failed")]
    BeginningCommandBuffer,
    #[error("Ending cmd buffer failed")]
    EndingCommandBuffer,
    #[error("Getting compute queue of idx {0} from family idx {1} failed")]
    GettingComputeQueue(u32, u32),
    #[error("Submitting to the compute queue failed")]
    ComputeQueueSubmission,
    #[error("Count of streams does not match the count of compute queue")]
    StreamCntQueueCntMismatch,
    #[error("Not enough buffers for all streams")]
    NotEnoughBuffers,
    #[error("Fence creation failed")]
    FenceCreation,
    #[error("Waiting on fence failed")]
    WaitingOnFence,
}

#[derive(Default, Clone)]
struct ComputeStream {
    max_submissions_in_flight: u32, //< number of command buffers per stream that can be in
    // use simultaneously
    compute_queue: vk::Queue,
    exec_kernel_submission_slots: Vec<(vk::CommandBuffer, vk::Fence)>,
    transfer_h2d_submission_slots: Vec<(vk::CommandBuffer, vk::Fence)>,
    transfer_d2h_submission_slots: Vec<(vk::CommandBuffer, vk::Fence)>,
    next_submission_slot: usize,
}

pub struct Dispatcher<'a> {
    cctx: &'a ComputeContext,
    command_pool_permanent: vk::CommandPool,
    max_stream_count: u32,
    compute_streams: Vec<ComputeStream>,
}

impl<'a> Dispatcher<'a> {
    const MAX_SUBMISSIONS_IN_FLIGHT: u32 = 3;

    pub fn new(
        cctx: &'a ComputeContext,
        requested_stream_count: u32,
    ) -> Result<Self, DispatcherError> {
        let command_pool_permanent = create_command_pool(
            &cctx.device,
            cctx.compute_queue_family_index,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )
        .map_err(|_| DispatcherError::CommandPoolCreation)?;

        let max_stream_count = std::cmp::min(
            cctx.compute_queue_family_props.queue_count,
            requested_stream_count,
        );

        let mut compute_streams = vec![ComputeStream::default(); max_stream_count as usize];

        let compute_command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool_permanent)
            .command_buffer_count(3 * max_stream_count * Self::MAX_SUBMISSIONS_IN_FLIGHT)
            .level(vk::CommandBufferLevel::PRIMARY);
        let mut command_buffers = unsafe {
            cctx.device
                .allocate_command_buffers(&compute_command_buffer_allocate_info)
        }
        .map_err(|_| DispatcherError::CommandBufferAllocation)?;

        let mut compute_queues = (0..max_stream_count)
            .map(|i| unsafe {
                cctx.device
                    .get_device_queue(cctx.compute_queue_family_index, i)
            })
            .collect::<Vec<vk::Queue>>();

        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        for stream in compute_streams.iter_mut().rev() {
            stream.max_submissions_in_flight = Self::MAX_SUBMISSIONS_IN_FLIGHT;
            stream.compute_queue = compute_queues
                .pop()
                .ok_or(DispatcherError::StreamCntQueueCntMismatch)?;
            for _ in 0..Self::MAX_SUBMISSIONS_IN_FLIGHT {
                stream.exec_kernel_submission_slots.push((
                    command_buffers
                        .pop()
                        .ok_or(DispatcherError::NotEnoughBuffers)?,
                    unsafe { cctx.device.create_fence(&fence_create_info, None) }
                        .map_err(|_| DispatcherError::FenceCreation)?,
                ));
                stream.transfer_h2d_submission_slots.push((
                    command_buffers
                        .pop()
                        .ok_or(DispatcherError::NotEnoughBuffers)?,
                    unsafe { cctx.device.create_fence(&fence_create_info, None) }
                        .map_err(|_| DispatcherError::FenceCreation)?,
                ));
                stream.transfer_d2h_submission_slots.push((
                    command_buffers
                        .pop()
                        .ok_or(DispatcherError::NotEnoughBuffers)?,
                    unsafe { cctx.device.create_fence(&fence_create_info, None) }
                        .map_err(|_| DispatcherError::FenceCreation)?,
                ));
            }
        }

        Ok(Dispatcher {
            cctx,
            command_pool_permanent,
            max_stream_count,
            compute_streams,
        })
    }

    fn create_stage(
        &self,
        kernel: &Kernel,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), DispatcherError> {
        let total_arg_size = kernel
            .buffers
            .iter()
            .map(|&buffer| unsafe { self.cctx.device.get_buffer_memory_requirements(buffer).size })
            .sum::<u64>();

        let buffer_create_info = vk::BufferCreateInfo::default()
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .size(total_arg_size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe {
            self.cctx
                .device
                .create_buffer(&buffer_create_info, None)
                .map_err(|_| DispatcherError::StagingBufferCreation)?
        };

        let buffer_memory_requirements = unsafe {
            self.cctx
                .device
                .get_buffer_memory_requirements(staging_buffer)
        };
        let memory_properties = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        let elligible_memory_type_index = find_memory_type_index(
            buffer_memory_requirements.memory_type_bits,
            &memory_properties,
            &self.cctx.memory_properties,
        )
        .ok_or(DispatcherError::StagingBufferNotSupported)?;
        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(buffer_memory_requirements.size)
            .memory_type_index(elligible_memory_type_index);
        let staging_memory = unsafe {
            self.cctx
                .device
                .allocate_memory(&memory_allocate_info, None)
                .map_err(|_| DispatcherError::StagingMemoryAllocation)?
        };

        unsafe {
            self.cctx
                .device
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
                .map_err(|_| DispatcherError::StagingBufferBinding)?
        };

        Ok((staging_buffer, staging_memory))
    }

    fn destroy_stage(&self, stage: (vk::Buffer, vk::DeviceMemory)) {
        unsafe { self.cctx.device.unmap_memory(stage.1) };
        unsafe { self.cctx.device.destroy_buffer(stage.0, None) };
    }

    fn upload_arguments(
        &self,
        kernel: &Kernel,
        stage: (vk::Buffer, vk::DeviceMemory),
        arg_vals: &[Box<dyn DeviceTransferable>],
        submission_slot: (vk::CommandBuffer, vk::Fence),
        queue: vk::Queue,
    ) -> Result<(), DispatcherError> {
        let command_buffer = submission_slot.0;
        let fence = submission_slot.1;
        let total_mem_size = unsafe {
            self.cctx
                .device
                .get_buffer_memory_requirements(stage.0)
                .size
        };
        let mapped_mem = unsafe {
            self.cctx
                .device
                .map_memory(stage.1, 0, total_mem_size, vk::MemoryMapFlags::empty())
                .map_err(|_| DispatcherError::MappingStagingMemory)?
        };

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.cctx
                .device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
        }
        .map_err(|_| DispatcherError::BeginningCommandBuffer)?;
        unsafe {
            self.cctx
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        };
        let mut offset: usize = 0;
        let command_buffers = [command_buffer];
        for (val, &buffer) in arg_vals.iter().zip(kernel.buffers.iter()) {
            // copy
            val.cpy_to(mapped_mem, offset);
            let val_size: usize =
                unsafe { self.cctx.device.get_buffer_memory_requirements(buffer).size }
                    .try_into()
                    .map_err(|_| DispatcherError::Casting)?;

            // command record
            let copy_region = vk::BufferCopy::default()
                .src_offset(offset.try_into().map_err(|_| DispatcherError::Casting)?)
                .dst_offset(0)
                .size(val_size.try_into().map_err(|_| DispatcherError::Casting)?);
            unsafe {
                self.cctx
                    .device
                    .cmd_copy_buffer(command_buffer, stage.0, buffer, &[copy_region]);
            }
            offset += val_size;
        }
        unsafe { self.cctx.device.end_command_buffer(command_buffer) }
            .map_err(|_| DispatcherError::EndingCommandBuffer)?;
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe {
            self.cctx
                .device
                .queue_submit(queue, &[submit_info], fence)
        }
        .map_err(|_| DispatcherError::ComputeQueueSubmission)?;

        Ok(())
    }

    fn launch(
        &self,
        kernel: &Kernel,
        command_buffer: vk::CommandBuffer,
        compute_queue: vk::Queue,
    ) -> Result<(), DispatcherError> {
        Ok(())
    }

    fn get_next_submission_slots(
        &mut self,
        stream_nr: usize,
    ) -> Result<[(vk::CommandBuffer, vk::Fence); 3], DispatcherError> {
        let stream = &mut self.compute_streams[stream_nr];
        let exec_kernel_submission_slot = stream.exec_kernel_submission_slots[stream.next_submission_slot];
        let transfer_h2d_submission_slot = stream.transfer_h2d_submission_slots[stream.next_submission_slot];
        let transfer_d2h_submission_slot = stream.transfer_d2h_submission_slots[stream.next_submission_slot];
        unsafe {
            self.cctx.device.wait_for_fences(
                &[
                    exec_kernel_submission_slot.1,
                    transfer_h2d_submission_slot.1,
                    transfer_d2h_submission_slot.1,
                ],
                true,
                u64::MAX,
            )
        }
        .map_err(|_| DispatcherError::WaitingOnFence)?;
        stream.next_submission_slot =
            (stream.next_submission_slot + 1) % (Self::MAX_SUBMISSIONS_IN_FLIGHT as usize);

        Ok([
            exec_kernel_submission_slot,
            transfer_h2d_submission_slot,
            transfer_d2h_submission_slot,
        ])
    }

    pub fn dispatch(
        &mut self,
        kernel: &Kernel,
        arg_vals: Vec<Box<dyn DeviceTransferable>>,
        stream_nr: usize,
    ) -> Result<(), DispatcherError> {
        let stage = self.create_stage(kernel)?;

        let compute_queue = self.compute_streams[stream_nr].compute_queue;
        let [exec_subm_slot, h2d_subm_slot, d2h_subm_slot] = self.get_next_submission_slots(stream_nr)?;

        self.upload_arguments(kernel, stage, &arg_vals, h2d_subm_slot, compute_queue)?;
        self.launch(kernel)?;

        self.destroy_stage(stage);

        Ok(())
    }
}

impl<'a> Drop for Dispatcher<'a> {
    fn drop(&mut self) {
        for stream in self.compute_streams {
            for i in 0..stream.max_submissions_in_flight {
                let exec_kernel_submission = stream.exec_kernel_submission_slots[i as usize];
                let transfer_h2d_submission = stream.transfer_h2d_submission_slots[i as usize];
                let transfer_d2h_submission = stream.transfer_h2d_submission_slots[i as usize];
                unsafe {
                    self.cctx.device.free_command_buffers(
                        self.command_pool_permanent,
                        &[
                            exec_kernel_submission.0,
                            transfer_h2d_submission.0,
                            transfer_d2h_submission.0,
                        ],
                    );
                    self.cctx
                        .device
                        .destroy_fence(exec_kernel_submission.1, None);
                    self.cctx
                        .device
                        .destroy_fence(transfer_h2d_submission.1, None);
                    self.cctx
                        .device
                        .destroy_fence(transfer_d2h_submission.1, None);
                }
            }
        }
        unsafe {
            self.device
                .destroy_command_pool(self.command_pool_permanent, None);
        }
    }
}

fn create_command_pool(
    device: &Device,
    queue_family_index: u32,
    flags: vk::CommandPoolCreateFlags,
) -> VkResult<vk::CommandPool> {
    let command_pool_create_info = vk::CommandPoolCreateInfo::default()
        .flags(flags)
        .queue_family_index(queue_family_index);
    unsafe { device.create_command_pool(&command_pool_create_info, None) }
}

fn create_vk_instance(app_name: &CStr, app_version: u32) -> Result<Instance, ComputeContextError> {
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
            .map_err(|_| ComputeContextError::VkInstanceCreation)
    }
}

fn select_physical_device(
    instance: &Instance,
) -> Result<(vk::PhysicalDevice, u32, vk::QueueFamilyProperties), ComputeContextError> {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|_| ComputeContextError::PhysicalDeviceEnumeration)?
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
            .ok_or(ComputeContextError::NoSuitablePhysicalDevice)
    }
}

fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<Device, ComputeContextError> {
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0]);

    let queue_infos = &[queue_info];
    let device_create_info = vk::DeviceCreateInfo::default().queue_create_infos(queue_infos);
    unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .map_err(|_| ComputeContextError::DeviceCreation)
    }
}

fn create_buffer_with_memory(
    cctx: &ComputeContext,
    buffer_usages: vk::BufferUsageFlags,
    requested_memory_properties: vk::MemoryPropertyFlags,
    size: u64,
) -> Result<(vk::Buffer, vk::DeviceMemory), KernelResourceManagerError> {
    let buffer_create_info = vk::BufferCreateInfo::default()
        .usage(buffer_usages)
        .size(size)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe {
        cctx.device
            .create_buffer(&buffer_create_info, None)
            .map_err(|_| KernelResourceManagerError::BufferCreation)?
    };

    let buffer_memory_requirements = unsafe { cctx.device.get_buffer_memory_requirements(buffer) };
    let elligible_memory_type_index = find_memory_type_index(
        buffer_memory_requirements.memory_type_bits,
        &requested_memory_properties,
        &cctx.memory_properties,
    )
    .ok_or(KernelResourceManagerError::BufferMemoryIncompatibility)?;

    let memory_allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(buffer_memory_requirements.size)
        .memory_type_index(elligible_memory_type_index);
    let device_memory = unsafe {
        cctx.device
            .allocate_memory(&memory_allocate_info, None)
            .map_err(|_| KernelResourceManagerError::DeviceMemoryAllocation)?
    };
    unsafe {
        cctx.device
            .bind_buffer_memory(buffer, device_memory, 0)
            .map_err(|_| KernelResourceManagerError::BindingMemoryToBuffer)?
    };
    Ok((buffer, device_memory))
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

fn bind_descriptor_set_to_buffer(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    binding_nrs: &[u32],
    descriptor_types: &[vk::DescriptorType],
    buffers: &[vk::Buffer],
) -> Result<(), KernelError> {
    let mut writes: Vec<vk::WriteDescriptorSet> = vec![];
    let mut buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]> = vec![];
    for i in 0..binding_nrs.len() {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffers[i])
            .offset(0)
            .range(vk::WHOLE_SIZE);
        buffer_infos.push([buffer_info]);
    }
    for i in 0..binding_nrs.len() {
        let write_descriptor_set = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(binding_nrs[i])
            .dst_array_element(0)
            .descriptor_type(descriptor_types[i])
            .descriptor_count(1)
            .buffer_info(&buffer_infos[i]);
        writes.push(write_descriptor_set);
    }
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
    Ok(())
}

fn create_pipeline(
    device: &Device,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    shader_module: vk::ShaderModule,
) -> Result<vk::Pipeline, KernelError> {
    let pipeline_layout_create_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(descriptor_set_layouts);
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .map_err(|_| KernelError::PipelineLayoutCreation)?
    };

    let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");
    let pipeline_create_infos = [vk::ComputePipelineCreateInfo::default()
        .layout(pipeline_layout)
        .stage(shader_stage_create_info)];
    let pipelines = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
            .map_err(|_| KernelError::PipelineCreation)?
    };

    unsafe {
        device.destroy_shader_module(shader_module, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
    }

    Ok(pipelines[0])
}

fn create_descriptor_set_layout(
    device: &Device,
    bindings: &[u32],
    descriptor_types: &[vk::DescriptorType],
) -> Result<vk::DescriptorSetLayout, KernelError> {
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
        device
            .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
            .map_err(|_| KernelError::DescriptorSetLayoutCreation)
    }
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
            is_writeable: true,
            size: (std::mem::size_of::<f32>() * arg.data.len())
                .try_into()
                .expect("Cast from usize to u64 not possible"),
        }];
        if let Ok(_kernel) = Kernel::new(&cctx, &res_mgr, shader_dummy, arg_infos) {
            panic!("Why is the shader valid?");
        }
    }
}
