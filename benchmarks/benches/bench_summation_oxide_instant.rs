fn main() {
    let status = std::process::Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench_summation_cuda_oxide/target/release/bench_summation_cuda_oxide"
    ))
    .args([benchmarks::N.to_string(), benchmarks::LOCAL_SIZE.to_string()])
    .env("BENCH_MEASURE_PERF", "0")
    .env("BENCH_MEASURE_INSTANT", "1")
    .env(
        "PERF_SCOPE_ITERATIONS",
        std::env::var("PERF_SCOPE_ITERATIONS").unwrap_or_else(|_| "1000".into()),
    )
    .status()
    .expect("Starting CUDA Oxide failed; build it with just build-oxide first");
    assert!(status.success(), "CUDA Oxide benchmark failed: {status}");
}