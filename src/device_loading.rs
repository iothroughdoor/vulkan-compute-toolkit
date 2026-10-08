use crate::{
    ComputeContext, find_memory_type_index,
    workload::{DeviceTransferable, DeviceVariable, Kernel},
};
use ash::{Device, vk};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Staging buffer creation failed")]
    StagingBufferCreation,
    #[error("Staging buffer not supported by the device")]
    StagingBufferNotSupported,
    #[error("Allocation of staging memory failed")]
    StagingMemoryAllocation,
    #[error("Binding staging buffer to memory failed")]
    StagingBufferBinding,
}

#[derive(Debug, Error)]
pub enum DispatcherError {
    #[error("{0}")]
    GeneralError(#[from] Error),
    #[error("Creation of the command pool failed")]
    CommandPoolCreation,
    #[error("Allocating the command buffers failed")]
    CommandBufferAllocation,
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
    #[error("Resetting of fence failed")]
    ResettingFence,
    #[error("Querying fence status failed")]
    FenceStatus,
    #[error("Semaphore creation failed")]
    SemaphoreCreation,
    #[error("Mapping staging memory failed.")]
    MappingStagingMemory,
    #[error("Waiting on queue to become idle failed: {0}")]
    WaitingOnQueue(vk::Result),
    #[error("Resetting command buffer failed: {0}")]
    ResettingCommandBuffer(vk::Result),
}

// resources to manage submissions in flight
#[derive(Clone)]
struct SubmissionSlot {
    cmd_buf: vk::CommandBuffer,
    host_finish_sig: vk::Fence,
    device_finish_sig: vk::Semaphore,
    stage: Option<(vk::Buffer, vk::DeviceMemory)>,
}

#[derive(Default, Clone)]
struct ComputeStream {
    max_submissions_in_flight: u32, //< number of submitted command buffers per stream that have not
    // been finished yet, we call those "submissions in flight"
    comp_queue: vk::Queue,
    submission_slots: Vec<SubmissionSlot>,
    next_submission_slot_index: usize,
}

pub struct Dispatcher<'a> {
    cctx: &'a ComputeContext,
    cmd_pool_permanent: vk::CommandPool,
    max_stream_count: u32,
    comp_streams: Vec<ComputeStream>,
}

impl<'a> Dispatcher<'a> {
    // has to be at least 2!
    const MAX_SUBMISSIONS_IN_FLIGHT: u32 = 3;

    pub fn new(
        cctx: &'a ComputeContext,
        requested_stream_count: u32,
    ) -> Result<Self, DispatcherError> {
        let cmd_pool_permanent = cctx
            .create_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .map_err(|_| DispatcherError::CommandPoolCreation)?;

        let max_stream_count = std::cmp::min(
            cctx.comp_queue_fam_props.queue_count,
            requested_stream_count,
        );

        /* Create Compute Streams */
        let mut comp_streams = vec![ComputeStream::default(); max_stream_count as usize];

        let comp_cmd_buf_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool_permanent)
            .command_buffer_count(3 * max_stream_count * Self::MAX_SUBMISSIONS_IN_FLIGHT)
            .level(vk::CommandBufferLevel::PRIMARY);
        let mut cmd_bufs = unsafe { cctx.dev.allocate_command_buffers(&comp_cmd_buf_alloc_info) }
            .map_err(|_| DispatcherError::CommandBufferAllocation)?;

        let mut comp_queues = (0..max_stream_count)
            .map(|i| unsafe { cctx.dev.get_device_queue(cctx.comp_queue_fam_idx, i) })
            .collect::<Vec<vk::Queue>>();

        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let sem_create_info = vk::SemaphoreCreateInfo::default();
        for stream in comp_streams.iter_mut().rev() {
            stream.max_submissions_in_flight = Self::MAX_SUBMISSIONS_IN_FLIGHT;
            stream.comp_queue = comp_queues
                .pop()
                .ok_or(DispatcherError::StreamCntQueueCntMismatch)?;
            for _ in 0..Self::MAX_SUBMISSIONS_IN_FLIGHT {
                stream.submission_slots.push(SubmissionSlot {
                    cmd_buf: cmd_bufs.pop().ok_or(DispatcherError::NotEnoughBuffers)?,
                    host_finish_sig: unsafe { cctx.dev.create_fence(&fence_create_info, None) }
                        .map_err(|_| DispatcherError::FenceCreation)?,
                    device_finish_sig: unsafe { cctx.dev.create_semaphore(&sem_create_info, None) }
                        .map_err(|_| DispatcherError::SemaphoreCreation)?,
                    stage: None,
                });
            }
        }

        Ok(Dispatcher {
            cctx,
            cmd_pool_permanent,
            max_stream_count,
            comp_streams,
        })
    }

    pub fn submit_upload(
        &mut self,
        dev_vars: &[DeviceVariable],
        arg_vals: &[Box<dyn DeviceTransferable>],
        stream_nr: usize,
    ) -> Result<(), DispatcherError> {
        let queue = self.comp_streams[stream_nr].comp_queue;
        let prev_submission_slot = self.get_current_submission_slot(stream_nr);
        let wait_semaphore = match unsafe {
            self.cctx
                .dev
                .get_fence_status(prev_submission_slot.host_finish_sig)
        } {
            Ok(false) => Some(prev_submission_slot.device_finish_sig),
            Ok(true) => None,
            _ => return Err(DispatcherError::FenceStatus),
        };

        let stage = create_stage(self.cctx, dev_vars)?;
        let subm_slot = self.acquire_next_submission_slot(stream_nr)?;
        let cmd_buf = subm_slot.cmd_buf;
        let fence = subm_slot.host_finish_sig;
        let signal_semaphore = subm_slot.device_finish_sig;
        subm_slot.stage = Some(stage);
        let stage = match self.get_current_submission_slot(stream_nr).stage {
            Some(stage) => stage,
            None => panic!("Somehow the stage vanished magically"),
        };

        let total_mem_size = unsafe { self.cctx.dev.get_buffer_memory_requirements(stage.0).size };
        let mapped_mem = unsafe {
            self.cctx
                .dev
                .map_memory(stage.1, 0, total_mem_size, vk::MemoryMapFlags::empty())
                .map_err(|_| DispatcherError::MappingStagingMemory)?
        };

        /* Record */
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.cctx
                .dev
                .begin_command_buffer(cmd_buf, &command_buffer_begin_info)
        }
        .map_err(|_| DispatcherError::BeginningCommandBuffer)?;
        let mut offset: usize = 0;
        let command_buffers = [cmd_buf];
        for (val, dev_var) in arg_vals.iter().zip(dev_vars.iter()) {
            val.cpy_to(mapped_mem, offset);
            let val_size = val.size();
            let copy_region = vk::BufferCopy::default()
                .src_offset(offset.try_into().map_err(|_| DispatcherError::Casting)?)
                .dst_offset(0)
                .size(val_size.try_into().map_err(|_| DispatcherError::Casting)?);
            unsafe {
                self.cctx
                    .dev
                    .cmd_copy_buffer(cmd_buf, stage.0, dev_var.buffer, &[copy_region]);
            }
            offset += val_size;
        }
        unsafe { self.cctx.dev.end_command_buffer(cmd_buf) }
            .map_err(|_| DispatcherError::EndingCommandBuffer)?;

        /* Submit */
        let wait_semaphores = wait_semaphore.map_or(vec![], |s| vec![s]);
        let signal_semaphores = [signal_semaphore];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores);
        unsafe { self.cctx.dev.queue_submit(queue, &[submit_info], fence) }
            .map_err(|_| DispatcherError::ComputeQueueSubmission)?;

        Ok(())
    }

    pub fn sync_stream(&self, stream_nr: usize) -> Result<(), DispatcherError> {
        let comp_queue = self.comp_streams[stream_nr].comp_queue;
        unsafe { self.cctx.dev.queue_wait_idle(comp_queue) }.map_err(|e| DispatcherError::WaitingOnQueue(e))?;
        Ok(())
    }

    /*
    pub fn submit_launch(
        &self,
        kernel: &Kernel,
        args: &[DeviceVariable],
        subm_slot: (vk::CommandBuffer, vk::Fence),
        compute_queue: vk::Queue,
    ) -> Result<(), DispatcherError> {
        let command_buffer = subm_slot.0;
        let fence = subm_slot.1;

        let buffers = args
            .iter()
            .map(|dev_var| dev_var.buffer)
            .collect::<Vec<vk::Buffer>>();

        bind_descriptor_set_to_buffer(
            &self.cctx.dev,
            kernel.descriptor_set,
            &kernel.binding_nrs,
            &kernel.descriptor_types,
            &buffers,
        );

        let compute_command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
        let submit_info = vk::SubmitInfo::default().command_buffers(&[subm_slot.0]);
        unsafe {
            self.cctx
                .dev
                .begin_command_buffer(compute_command_buffer, &compute_command_buffer_begin_info)
                .map_err(|_| Error::GpuError("Begin compute command buffer failed.".into()))?;

            vk_ctx.device.cmd_bind_pipeline(
                vk_ctx.compute_command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                kernel
                    .compute_pipeline
                    .expect("No pipeline found for kernel"),
            );
            vk_ctx.device.cmd_bind_descriptor_sets(
                vk_ctx.compute_command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                kernel
                    .compute_pipeline_layout
                    .expect("No pipeline layout found for kernel"),
                0,
                std::slice::from_ref(&kernel.descriptor_set),
                &[],
            );
            vk_ctx.device.cmd_dispatch(
                vk_ctx.compute_command_buffer,
                group_x_count,
                group_y_count,
                1,
            );

            vk_ctx
                .device
                .end_command_buffer(vk_ctx.compute_command_buffer)
                .map_err(|_| Error::GpuError("End compute command buffer failed.".into()))?;
            vk_ctx
                .device
                .queue_submit(
                    vk_ctx.compute_queue,
                    std::slice::from_ref(&submit_info),
                    vk::Fence::null(),
                )
                .map_err(|_| {
                    Error::GpuError(
                        "Memcpy from host to device failed: Cmd buffer submission failed.".into(),
                    )
                })?;
            vk_ctx.device
            .queue_wait_idle(vk_ctx.compute_queue)
            .map_err(|_| { Error::GpuError("Memcpy from host to device failed: waiting on queue to become idle aborted unexpectedly.".into()) })?;
        }

        Ok(())
    }*/

    fn destroy_stage(&self, stage: (vk::Buffer, vk::DeviceMemory)) {
        unsafe { self.cctx.dev.unmap_memory(stage.1) };
        unsafe { self.cctx.dev.destroy_buffer(stage.0, None) };
    }

    fn get_current_submission_slot(&self, stream_nr: usize) -> &SubmissionSlot {
        let stream = &self.comp_streams[stream_nr];
        let cur_subm_slot_idx = (((stream.next_submission_slot_index as i64) - 1
            + Self::MAX_SUBMISSIONS_IN_FLIGHT as i64)
            % (Self::MAX_SUBMISSIONS_IN_FLIGHT as i64)) as usize;
        &stream.submission_slots[cur_subm_slot_idx]
    }

    fn acquire_next_submission_slot(
        &mut self,
        stream_nr: usize,
    ) -> Result<&mut SubmissionSlot, DispatcherError> {
        let stream = &self.comp_streams[stream_nr];
        let subm_slot = &stream.submission_slots[stream.next_submission_slot_index];
        unsafe {
            self.cctx
                .dev
                .wait_for_fences(&[subm_slot.host_finish_sig], true, u64::MAX)
        }
        .map_err(|_| DispatcherError::WaitingOnFence)?;
        if let Some(stage) = subm_slot.stage {
            self.destroy_stage(stage);
        }
        unsafe {
            self.cctx
                .dev
                .reset_fences(&[subm_slot.host_finish_sig])
        }
        .map_err(|_| DispatcherError::ResettingFence)?;
        unsafe {
            self.cctx
                .dev
                .reset_command_buffer(subm_slot.cmd_buf, vk::CommandBufferResetFlags::empty())
        }.map_err(|e| DispatcherError::ResettingCommandBuffer(e))?;

        let stream = &mut self.comp_streams[stream_nr];
        let subm_slot = &mut stream.submission_slots[stream.next_submission_slot_index];
        stream.next_submission_slot_index =
            (stream.next_submission_slot_index + 1) % (Self::MAX_SUBMISSIONS_IN_FLIGHT as usize);

        Ok(subm_slot)
    }
}

impl<'a> Drop for Dispatcher<'a> {
    fn drop(&mut self) {
        for stream in &self.comp_streams {
            for i in 0..stream.max_submissions_in_flight {
                let subm_slot = &stream.submission_slots[i as usize];
                unsafe {
                    self.cctx
                        .dev
                        .free_command_buffers(self.cmd_pool_permanent, &[subm_slot.cmd_buf]);
                    self.cctx.dev.destroy_fence(subm_slot.host_finish_sig, None);
                }
            }
        }
        unsafe {
            self.cctx
                .dev
                .destroy_command_pool(self.cmd_pool_permanent, None);
        }
    }
}

fn bind_descriptor_set_to_buffer(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    binding_nrs: &[u32],
    descriptor_types: &[vk::DescriptorType],
    buffers: &[vk::Buffer],
) {
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
}

fn create_stage(
    cctx: &ComputeContext,
    args: &[DeviceVariable],
) -> Result<(vk::Buffer, vk::DeviceMemory), Error> {
    let total_arg_size = args
        .iter()
        .map(|dev_var| unsafe { cctx.dev.get_buffer_memory_requirements(dev_var.buffer).size })
        .sum();

    let buffer_create_info = vk::BufferCreateInfo::default()
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .size(total_arg_size)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let staging_buffer = unsafe {
        cctx.dev
            .create_buffer(&buffer_create_info, None)
            .map_err(|_| Error::StagingBufferCreation)?
    };

    let buffer_memory_requirements =
        unsafe { cctx.dev.get_buffer_memory_requirements(staging_buffer) };
    let mem_props = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::HOST_CACHED;
    let mem_type_idx = find_memory_type_index(
        buffer_memory_requirements.memory_type_bits,
        &mem_props,
        &cctx.mem_props,
    )
    .ok_or(Error::StagingBufferNotSupported)?;
    let memory_allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(buffer_memory_requirements.size)
        .memory_type_index(mem_type_idx);
    let staging_memory = unsafe {
        cctx.dev
            .allocate_memory(&memory_allocate_info, None)
            .map_err(|_| Error::StagingMemoryAllocation)?
    };

    unsafe {
        cctx.dev
            .bind_buffer_memory(staging_buffer, staging_memory, 0)
            .map_err(|_| Error::StagingBufferBinding)?
    };

    Ok((staging_buffer, staging_memory))
}
