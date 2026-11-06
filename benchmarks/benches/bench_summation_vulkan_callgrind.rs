use gungraun::{
    Callgrind, EntryPoint, LibraryBenchmarkConfig, Tool, library_benchmark, library_benchmark_group, main,
};
#[library_benchmark(
    config = LibraryBenchmarkConfig::default()
    .default_tool(Tool::Callgrind)
    .tool(
        Callgrind::default()
            .entry_point(EntryPoint::Custom("*benchmarks::run_summation*".into()))
    )
    //.tool(
    //    Perf::default()
    //    .disable_entry_point(true)
    //    .event_set("task-clock")
    //)
)]
fn bench_summation_vulkan_callgrind() {
    benchmarks::summation(1, false);
}

library_benchmark_group!(name = summation_vulkan_callgrind; benchmarks = bench_summation_vulkan_callgrind);
main!(library_benchmark_groups = summation_vulkan_callgrind);