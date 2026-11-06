bench_vulkan_callgrind:
    RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes" cargo bench -p benchmarks --bench bench_summation_vulkan_callgrind

bench_vulkan_perf:
    RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes" cargo bench -p benchmarks --bench bench_summation_vulkan_perf

bench_vulkan_instant:
    RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes" cargo bench -p benchmarks --bench bench_summation_vulkan_instant

show_vulkan_perf_cg:
    perf report --children --call-graph graph,0.5,caller -i target/gungraun/benchmarks/bench_summation_vulkan_perf/summation_vulkan_perf/bench_summation_vulkan_perf/perf.bench_summation_vulkan_perf.data

build-oxide:
    cd benchmarks/bench_summation_cuda_oxide && CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH:-/usr/local/cuda-13.0} RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes" cargo +1.99.1 oxide build

bench_oxide_callgrind: build-oxide
    cargo bench -p benchmarks --bench bench_summation_oxide_callgrind

bench_oxide_perf: build-oxide
    RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes" cargo bench -p benchmarks --bench bench_summation_oxide_perf

bench_oxide_instant: build-oxide
    RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes" cargo bench -p benchmarks --bench bench_summation_oxide_instant

show_oxide_perf_cg:
    perf report --children --call-graph graph,0.5,caller -i target/gungraun/benchmarks/bench_summation_oxide_perf/summation_oxide_perf/bench_summation_oxide_perf/perf.bench_summation_oxide_perf.data