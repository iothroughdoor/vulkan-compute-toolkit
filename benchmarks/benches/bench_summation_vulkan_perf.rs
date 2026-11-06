use gungraun::{
    LibraryBenchmarkConfig, Perf, Tool, library_benchmark, library_benchmark_group, main,
};

#[library_benchmark(
    config = LibraryBenchmarkConfig::default()
        .env_clear(false)
        .default_tool(Tool::Perf)
        .tool(Perf::default()
            .disable_entry_point(true)
            .event_set("instructions:u,cycles:u,task-clock,page-faults")
            .record(true)
            .record_args(["--call-graph", "dwarf,16384", "-F", "999"]))
)]
fn bench_summation_vulkan_perf() {
    let iterations = std::env::var("PERF_SCOPE_ITERATIONS")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .expect("PERF_SCOPE_ITERATIONS must be a positive integer");
    benchmarks::summation(iterations, true);
}

library_benchmark_group!(name = summation_vulkan_perf; benchmarks = bench_summation_vulkan_perf);
main!(library_benchmark_groups = summation_vulkan_perf);