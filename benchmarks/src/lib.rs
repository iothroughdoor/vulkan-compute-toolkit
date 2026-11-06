use std::time::{Duration, Instant};
use vulkan_computing_toolkit as vctk;

pub const N: u32 = 1 << 20;
pub const LOCAL_SIZE: u32 = 128;

pub fn summation(iterations: usize, measure_perf: bool) -> Duration {
    assert!(iterations > 0, "iterations must be positive");
    let summand_1: Vec<f32> = (0..N).map(|value| value as f32).collect();
    let summand_2: Vec<f32> = (0..N).map(|value| value as f32).collect();
    let mut sum = vec![0f32; N as usize];
    let len = vec![N];

    let cctx = vctk::ComputeContext::new("Summation Benchmark", 0)
        .expect("Creating the compute context failed");
    let ker_args = vec![
        vctk::KernelArgInfo {
            is_uniformly_readonly: true,
            size: std::mem::size_of::<u32>() as u64,
        },
        vctk::KernelArgInfo {
            is_uniformly_readonly: false,
            size: std::mem::size_of::<f32>() as u64 * N as u64,
        },
        vctk::KernelArgInfo {
            is_uniformly_readonly: false,
            size: std::mem::size_of::<f32>() as u64 * N as u64,
        },
    ];
    let spirv_code = include_bytes_plus::include_bytes!("data/shader/sum.spv" as u32);
    let shader = vctk::ComputeShader::builder()
        .spirv(spirv_code.into())
        .local_size_x(LOCAL_SIZE)
        .build();
    let ker = vctk::Kernel::new(&cctx, &shader, ker_args.clone()).expect("Kernel creation failed");
    let dev_len = vctk::DeviceVariable::builder()
        .is_uniformly_read(true)
        .size(ker_args[0].size)
        .build(&cctx)
        .expect("Constructing dev_len failed");
    let dev_summand_1 = vctk::DeviceVariable::builder()
        .size(ker_args[1].size)
        .build(&cctx)
        .expect("Constructing dev_summand_1 failed");
    let dev_summand_2 = vctk::DeviceVariable::builder()
        .size(ker_args[2].size)
        .build(&cctx)
        .expect("Constructing dev_summand_2 failed");
    let sizes = ker_args.iter().map(|arg| arg.size).sum();
    let mut dispatcher =
        vctk::Dispatcher::new(&cctx, 3, &[sizes, 1, 1]).expect("Dispatcher creation failed");
    let host_variables: [&dyn vctk::DeviceTransferable; 3] = [&len, &summand_1, &summand_2];
    let dev_variables = [dev_len, dev_summand_1, dev_summand_2];

    let start = Instant::now();
    let token = measure_perf.then(|| gungraun::perf_enable!());
    for _ in 0..iterations {
        run_summation(&mut dispatcher, &ker, &dev_variables, &host_variables, &mut sum);
    }
    if let Some(token) = token {
        gungraun::perf_disable!(token);
    }
    let elapsed = start.elapsed();
    for (actual, input) in sum.into_iter().zip(summand_1) {
        assert_eq!(actual, 2_f32 * input);
    }
    elapsed
}

#[inline(never)]
fn run_summation(
    dispatcher: &mut vctk::Dispatcher<'_>,
    ker: &vctk::Kernel<'_>,
    dev_variables: &[vctk::DeviceVariable<'_>],
    host_variables: &[&dyn vctk::DeviceTransferable],
    sum: &mut Vec<f32>,
) {
    dispatcher
        .upload_async(dev_variables, host_variables, 0)
        .expect("Device upload failed");
    dispatcher
        .launch_async(ker, dev_variables, 0, [N / LOCAL_SIZE, 1, 1])
        .expect("Launching kernel failed");
    let mut output: [&mut dyn vctk::DeviceTransferable; 1] = [sum];
    dispatcher
        .download_sync(&dev_variables[1..2], &mut output, 0)
        .expect("Downloading failed");
}