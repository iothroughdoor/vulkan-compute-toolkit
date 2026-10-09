use vulkan_computing_toolkit as vctk;

const N: u32 = 4096;
const LOCAL_SIZE: u32 = 1024;

#[test]
fn test_sum_array_shader() {
    let nats: Vec<i32> = (0..(N as i32)).collect();
    let len: Vec<u32> = vec![N];
    let mut sum: Vec<i32> = vec![0];

    let cctx =
        vctk::ComputeContext::new("Smoke Test", 0).expect("Creating the compute context failed");

    let ker_args = vec![
        vctk::KernelArgInfo {
            is_uniformly_readonly: true,
            size: std::mem::size_of::<u32>() as u64,
        },
        vctk::KernelArgInfo {
            is_uniformly_readonly: false,
            size: ((std::mem::size_of::<i32>() as u32) * N) as u64,
        },
        vctk::KernelArgInfo {
            is_uniformly_readonly: false,
            size: std::mem::size_of::<i32>() as u64,
        },
    ];

    let ker = vctk::Kernel::new(&cctx, vctk::SUM_ARRAY_SHADER_SPV, ker_args.clone())
        .expect("Kernel creation failed");

    let dev_len = vctk::DeviceVariable::builder()
        .is_uniformly_read(true)
        .size(ker_args[0].size)
        .build(&cctx)
        .expect("Constructing dev_len failed");
    let dev_nats = vctk::DeviceVariable::builder()
        .size(ker_args[1].size)
        .build(&cctx)
        .expect("Constructing dev_nats failed");
    let dev_sum = vctk::DeviceVariable::builder()
        .size(ker_args[2].size)
        .build(&cctx)
        .expect("Constructing dev_sum failed");

    let mut dispatcher = vctk::Dispatcher::new(&cctx, 3).expect("Dispatcher creation failed");

    let host_variables: [&dyn vctk::DeviceTransferable; 3] = [&len, &nats, &sum];
    let dev_variables = [dev_len, dev_nats, dev_sum];

    dispatcher
        .upload_async(&dev_variables, &host_variables, 0)
        .expect("Device upload failed");

    dispatcher
        .launch_async(&ker, &dev_variables, 0, [N / LOCAL_SIZE, 1, 1])
        .expect("Launching kernel failed");

    let mut host_variables: [&mut dyn vctk::DeviceTransferable; 1] = [&mut sum];
    let dev_variables_2 = &dev_variables[2..3];

    dispatcher
        .download_sync(dev_variables_2, &mut host_variables, 0)
        .expect("Downloading failed");

    assert_eq!(sum[0], nats.iter().sum::<i32>());
}
