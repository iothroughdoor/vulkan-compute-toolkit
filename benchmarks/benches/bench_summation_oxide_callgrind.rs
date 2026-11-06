use gungraun::{
    BinaryBenchmarkConfig, Callgrind, Command, EntryPoint, Tool, binary_benchmark,
    binary_benchmark_group, main,
};

#[binary_benchmark]
fn bench_summation_oxide_callgrind() -> Command {
    Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench_summation_cuda_oxide/target/release/bench_summation_cuda_oxide"
    ))
    .args([benchmarks::N.to_string(), benchmarks::LOCAL_SIZE.to_string()])
    .env("PERF_SCOPE_ITERATIONS", "1")
    .build()
}

binary_benchmark_group!(name = summation_oxide_callgrind; benchmarks = bench_summation_oxide_callgrind);
main!(
    config = BinaryBenchmarkConfig::default()
        .default_tool(Tool::Callgrind)
        .tool(Callgrind::default().entry_point(EntryPoint::Custom(
            "*bench_summation_cuda_oxide::run_summation*".into()
        ))),
    binary_benchmark_groups = summation_oxide_callgrind
);