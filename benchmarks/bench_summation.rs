use include_bytes_plus::include_bytes;
use vulkan_computing_toolkit as vctk;
use std::process::Command;
use std::time::Instant;

fn main() {
    const N: u32 = 4096;
    const LOCAL_SIZE: u32 = 128;

    let summand_1: Vec<f32> = (0..N).map(|v| v as f32).collect();
    let summand_2: Vec<f32> = (0..N).map(|v| v as f32).collect();
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
            size: ((std::mem::size_of::<f32>() as u32) * N) as u64,
        },
        vctk::KernelArgInfo {
            is_uniformly_readonly: false,
            size: ((std::mem::size_of::<f32>() as u32) * N) as u64,
        },
    ];

    let spirv_code = include_bytes!("data/shader/sum.spv" as u32);
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

    dispatcher
        .upload_async(&dev_variables, &host_variables, 0)
        .expect("Device upload failed");

    dispatcher
        .launch_async(&ker, &dev_variables, 0, [N / LOCAL_SIZE, 1, 1])
        .expect("Launching kernel failed");

    let mut host_variables: [&mut dyn vctk::DeviceTransferable; 1] = [&mut sum];
    let dev_variables_2 = &dev_variables[1..2];

    dispatcher
        .download_sync(dev_variables_2, &mut host_variables, 0)
        .expect("Downloading failed");

    let elapsed_time = start.elapsed();
    println!("Vulkan: {} us", elapsed_time.as_micros());

    for (s, t) in sum.into_iter().zip(summand_1.into_iter()) {
        assert_eq!(s, 2_f32*t);
    }

    let status = Command::new("cargo")
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/benchmarks/bench_summation_cuda_oxide"))
        .args(["+1.99.1", "oxide", "run", "--", &N.to_string(), &LOCAL_SIZE.to_string()])
        .status()
        .expect("Starting the CUDA Oxide benchmark failed; install cargo-oxide and the CUDA toolkit");
    assert!(status.success(), "CUDA Oxide benchmark failed");
}
