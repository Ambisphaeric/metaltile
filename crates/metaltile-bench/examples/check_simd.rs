use metaltile_bench::runner::GpuRunner;

fn main() {
    let runner = GpuRunner::new().expect("Failed to create GpuRunner");
    println!("supports_simd_matrix: {}", runner.supports_simd_matrix());
}
