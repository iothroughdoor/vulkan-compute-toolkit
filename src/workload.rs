use crate::{ComputeContext, ComputeContextError, find_memory_type_index};
use ash::{Device, vk};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DeviceVariableError {
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

#[derive(Default)]
pub struct DeviceVariableBuilder {
    size_: u64,
    host_to_dev_transf_: bool,
    dev_to_host_transf_: bool,
    is_uniformly_readonly_: bool,
    use_host_caching_: bool,
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
    ) -> Result<DeviceVariable, DeviceVariableError> {
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
                .map_err(|_| DeviceVariableError::BufferCreation)?
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

pub struct Kernel<'a, 'b> {
    cctx: &'b ComputeContext,
    res_mgr: &'a KernelResourceManager,

    pub(crate) pipeline: vk::Pipeline,
    pub(crate) descriptor_set: vk::DescriptorSet,
    pub(crate) binding_nrs: Vec<u32>,
    pub(crate) descriptor_types: Vec<vk::DescriptorType>,
}

#[derive(Clone)]
pub struct KernelArgInfo {
    pub is_uniformly_readonly: bool,
    pub size: u64,
}

pub trait DeviceTransferable {
    fn cpy_from(&mut self, memory: *const core::ffi::c_void, offset: usize);
    fn cpy_to(&self, memory: *mut core::ffi::c_void, offset: usize);
    fn size(&self) -> usize;
}

impl<'a, 'b> Kernel<'a, 'b> {
    pub fn new(
        cctx: &'b ComputeContext,
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
            .map(|info| {
                if info.is_uniformly_readonly {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                }
            })
            .collect::<Vec<vk::DescriptorType>>();

        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(shader_spirv);
        let shader_module = unsafe {
            cctx.dev
                .create_shader_module(&shader_module_create_info, None)
                .map_err(|_| KernelError::ShaderModuleCreation)
        }?;
        let descriptor_set_layout =
            cctx.create_descriptor_set_layout(&binding_nrs, &descriptor_types)?;
        let pipeline = cctx.create_pipeline(&[descriptor_set_layout], shader_module)?;

        let descriptor_set = res_mgr
            .allocate_descriptor_set(descriptor_set_layout)
            .map_err(|e| KernelError::KernelResourceManager(e))?;

        unsafe {
            cctx.dev
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
        }

        Ok(Kernel {
            cctx: &cctx,
            res_mgr: &res_mgr,
            binding_nrs, 
            descriptor_types,
            pipeline,
            descriptor_set,
        })
    }
}

impl<'a, 'b> Drop for Kernel<'a, 'b> {
    fn drop(&mut self) {
        unsafe {
            self.cctx.dev.destroy_pipeline(self.pipeline, None);
            self.res_mgr.free_descriptor_set(self.descriptor_set);
        }
    }
}

#[derive(Debug, Error)]
pub enum KernelResourceManagerError {
    #[error("Creating the buffer pool failed")]
    Creation,
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
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(desc_count); // if multiple compute shaders are active operating on multiple objects,
        // we need a descriptor set for every such shader / object
        let descriptor_pool = unsafe {
            cctx.dev
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .map_err(|_| KernelResourceManagerError::Creation)?
        };
        Ok(KernelResourceManager {
            device: cctx.dev.clone(),
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
