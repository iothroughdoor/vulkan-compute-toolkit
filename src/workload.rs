use crate::{ComputeContext, ComputeContextError, find_memory_type_index, shader::ComputeShader};
use ash::vk;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DeviceVariableError {
    #[error("Creation of buffer failed: {0}")]
    BufferCreation(vk::Result),
    #[error(
        "Requirements of buffer on memory is incompatible with the memory types the device offers"
    )]
    BufferMemoryIncompatibility,
    #[error("Device memory allocation failed")]
    DeviceMemoryAllocation,
    #[error("Binding memory to buffer failed")]
    BindingMemoryToBuffer,
}

pub struct DeviceVariable<'a> {
    pub(crate) cctx: &'a ComputeContext,
    pub(crate) buffer: vk::Buffer,
    pub(crate) dev_memory: vk::DeviceMemory,
}

impl<'a> DeviceVariable<'a> {
    pub fn builder() -> DeviceVariableBuilder {
        DeviceVariableBuilder::default()
    }
}

impl<'a> Drop for DeviceVariable<'a> {
    fn drop(&mut self) {
        unsafe {
            self.cctx.dev.destroy_buffer(self.buffer, None);
            self.cctx.dev.free_memory(self.dev_memory, None);
        }
    }
}

pub struct DeviceVariableBuilder {
    size_: u64,
    host_to_dev_transf_: bool,
    dev_to_host_transf_: bool,
    is_uniformly_readonly_: bool,
    use_host_caching_: bool,
}

impl Default for DeviceVariableBuilder {
    fn default() -> Self {
        DeviceVariableBuilder{
            size_: 0,
            host_to_dev_transf_: true,
            dev_to_host_transf_: true,
            is_uniformly_readonly_: false,
            use_host_caching_: true,
        }
    }
}

impl DeviceVariableBuilder {
    pub fn size(mut self, x: u64) -> DeviceVariableBuilder {
        self.size_ = x;
        self
    }

    pub fn host_to_dev_transf(mut self, x: bool) -> DeviceVariableBuilder {
        self.host_to_dev_transf_ = x;
        self
    }

    pub fn dev_to_host_transf(mut self, x: bool) -> DeviceVariableBuilder {
        self.dev_to_host_transf_ = x;
        self
    }

    pub fn is_uniformly_read(mut self, x: bool) -> DeviceVariableBuilder {
        self.is_uniformly_readonly_ = x;
        self
    }

    pub fn use_host_caching(mut self, x: bool) -> DeviceVariableBuilder {
        self.use_host_caching_ = x;
        self
    }

    pub fn build<'a>(
        self,
        cctx: &'a ComputeContext,
    ) -> Result<DeviceVariable::<'a>, DeviceVariableError> {
        let mut buffer_usage = vk::BufferUsageFlags::empty();
        if self.host_to_dev_transf_ {
            buffer_usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if self.dev_to_host_transf_ {
            buffer_usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if self.is_uniformly_readonly_ {
            // for small, constant values, that are read by multiple threads in a local workgroup
            // this is what nvidia calls constant memory
            buffer_usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        } else {
            buffer_usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }

        let mut memory_properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        if self.use_host_caching_ {
            // only necessary for host - device transfers
            // stuff is cached on host side,
            // speeds up transfer
            memory_properties |= vk::MemoryPropertyFlags::HOST_CACHED;
        }

        let buffer_create_info = vk::BufferCreateInfo::default()
            .usage(buffer_usage)
            .size(self.size_)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            cctx.dev
                .create_buffer(&buffer_create_info, None)
                .map_err(|e| DeviceVariableError::BufferCreation(e))?
        };

        let buffer_memory_requirements = unsafe { cctx.dev.get_buffer_memory_requirements(buffer) };
        let elligible_memory_type_index = find_memory_type_index(
            buffer_memory_requirements.memory_type_bits,
            &memory_properties,
            &cctx.mem_props,
        )
        .ok_or(DeviceVariableError::BufferMemoryIncompatibility)?;

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(buffer_memory_requirements.size)
            .memory_type_index(elligible_memory_type_index);
        let dev_memory = unsafe {
            cctx.dev
                .allocate_memory(&memory_allocate_info, None)
                .map_err(|_| DeviceVariableError::DeviceMemoryAllocation)?
        };
        unsafe {
            cctx.dev
                .bind_buffer_memory(buffer, dev_memory, 0)
                .map_err(|_| DeviceVariableError::BindingMemoryToBuffer)?
        };

        Ok(DeviceVariable::<'a> {
            cctx,
            buffer,
            dev_memory,
        })
    }
}

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("{0}")]
    ComputeContext(#[from] ComputeContextError),
    #[error("Pipeline creation failed")]
    PipelineCreation,
    #[error("Descriptor set layout creation failed")]
    DescriptorSetLayoutCreation,
    #[error("Pipeline layout creation failed")]
    PipelineLayoutCreation,
    #[error("Invalid resource specification")]
    InvalidResourceSpec,
    #[error("Unsupported descriptor type")]
    UnsupportedDescriptorType,
}

pub struct Kernel<'a> {
    cctx: &'a ComputeContext,

    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) descriptor_set: vk::DescriptorSet,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) binding_nrs: Vec<u32>,
    pub(crate) descriptor_types: Vec<vk::DescriptorType>,
}

#[derive(Clone)]
pub struct KernelArgInfo {
    pub is_uniformly_readonly: bool,
    pub size: u64,
}

impl<'a> Kernel<'a> {
    pub fn new(
        cctx: &'a ComputeContext,
        //shader_spirv: &[u32],
        shader: &ComputeShader,
        arg_infos: Vec<KernelArgInfo>,
    ) -> Result<Self, KernelError> {
        let binding_nrs = arg_infos
            .iter()
            .enumerate()
            .map(|(i, _)| i.try_into().map_err(|_| KernelError::InvalidResourceSpec))
            .collect::<Result<Vec<u32>, KernelError>>()?;
        let descriptor_types = arg_infos
            .iter()
            .map(|info| {
                if info.is_uniformly_readonly {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                }
            })
            .collect::<Vec<vk::DescriptorType>>();

        let descriptor_set_layout =
            cctx.create_descriptor_set_layout(&binding_nrs, &descriptor_types)?;
        let (pipeline, pipeline_layout) =
            cctx.create_pipeline(&[descriptor_set_layout], shader)?;

        let descriptor_set = cctx.allocate_descriptor_set(descriptor_set_layout)?;

        Ok(Kernel {
            cctx: &cctx,
            binding_nrs,
            descriptor_types,
            pipeline,
            pipeline_layout,
            descriptor_set,
            descriptor_set_layout,
        })
    }
}

impl<'a> Drop for Kernel<'a> {
    fn drop(&mut self) {
        unsafe {
            self.cctx.dev.destroy_pipeline(self.pipeline, None);
            self.cctx.free_descriptor_set(self.descriptor_set).unwrap();
            self.cctx
                .dev
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.cctx
                .dev
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

pub trait DeviceTransferable {
    fn cpy_from(&mut self, memory: *const core::ffi::c_void, offset: usize);
    fn cpy_to(&self, memory: *mut core::ffi::c_void, offset: usize);
    fn size(&self) -> usize;
}


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


