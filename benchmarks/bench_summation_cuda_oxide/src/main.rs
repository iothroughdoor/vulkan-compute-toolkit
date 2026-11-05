use cuda_core::{CudaContext, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn sum(len: &[u32], mut summand_1: DisjointSlice<f32>, summand_2: &[f32]) {
        let index = thread::index_1d();
        let offset = index.get();
        if offset < len[0] as usize {
            if let Some(value) = summand_1.get_mut(index) {
                *value += summand_2[offset];
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let n: u32 = args.next().ok_or("Missing array length")?.parse()?;
    let local_size: u32 = args.next().ok_or("Missing local size")?.parse()?;
    if n == 0 || local_size == 0 || local_size > 1024 || n % local_size != 0 {
        return Err(
            "Array length must be positive and divisible by a local size in 1..=1024".into(),
        );
    }

    let summand_1: Vec<f32> = (0..n).map(|value| value as f32).collect();
    let summand_2: Vec<f32> = (0..n).map(|value| value as f32).collect();
    let mut sum = vec![0f32; n as usize];
    let len = [n];

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let mut dev_len = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let mut dev_summand_1 = DeviceBuffer::<f32>::zeroed(&stream, n as usize)?;
    let mut dev_summand_2 = DeviceBuffer::<f32>::zeroed(&stream, n as usize)?;
    let launch_config = LaunchConfig {
        grid_dim: (n / local_size, 1, 1),
        block_dim: (local_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let start = Instant::now();
    let submission = unsafe {
        (|| -> Result<(), DriverError> {
            dev_len.copy_from_host_async_unchecked(&stream, &len)?;
            dev_summand_1.copy_from_host_async_unchecked(&stream, &summand_1)?;
            dev_summand_2.copy_from_host_async_unchecked(&stream, &summand_2)?;
            module.sum(
                &stream,
                launch_config,
                &dev_len,
                &mut dev_summand_1,
                &dev_summand_2,
            )?;
            Ok(())
        })()
    };
    if let Err(error) = submission {
        stream.synchronize()?;
        return Err(error.into());
    }
    dev_summand_1.copy_to_host(&stream, &mut sum)?;
    let elapsed_time = start.elapsed();
    println!("CUDA Oxide: {} us", elapsed_time.as_micros());

    for (actual, input) in sum.into_iter().zip(summand_1.into_iter()) {
        assert_eq!(actual, 2f32 * input);
    }
    Ok(())
}
