use gungraun::{
    BinaryBenchmarkConfig, Command, Perf, Tool, binary_benchmark, binary_benchmark_group, main,
};

#[binary_benchmark]
fn bench_summation_oxide_perf() -> Command {
    Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench_summation_cuda_oxide/target/release/bench_summation_cuda_oxide"
    ))
    .args([benchmarks::N.to_string(), benchmarks::LOCAL_SIZE.to_string()])
    .env("BENCH_MEASURE_PERF", "1")
    .env(
        "PERF_SCOPE_ITERATIONS",
        std::env::var("PERF_SCOPE_ITERATIONS").unwrap_or_else(|_| "1000".into()),
    )
    .build()
}

binary_benchmark_group!(name = summation_oxide_perf; benchmarks = bench_summation_oxide_perf);
main!(
    config = BinaryBenchmarkConfig::default()
        .default_tool(Tool::Perf)
        .tool(Perf::default()
            .disable_entry_point(true)
            .args(["--delay=-1"])
            .event_set("instructions:u,cycles:u,task-clock,page-faults")
            .record(true)
            .record_args(["--delay=-1", "--call-graph", "dwarf,16384", "-F", "999"])),
    binary_benchmark_groups = summation_oxide_perf
);