fn main() {
    let iterations: usize = std::env::var("PERF_SCOPE_ITERATIONS")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .expect("PERF_SCOPE_ITERATIONS must be a positive integer");
    let elapsed = benchmarks::summation(iterations, false);
    println!(
        "Vulkan: {iterations} iterations, {:.3} ms total, {:.3} us/iteration",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
    );
}