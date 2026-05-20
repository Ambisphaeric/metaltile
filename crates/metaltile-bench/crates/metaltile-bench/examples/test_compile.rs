use metaltile_bench::runner::GpuRunner;

fn main() {
    let runner = GpuRunner::new();
    println!("supports_simd_matrix: {}", runner.supports_simd_matrix());
    
    let simd_msl = r#"
#include <metal_stdlib>
using namespace metal;
#if __METAL_VERSION__ >= 310
#include <metal_simdgroup_matrix>
#endif
kernel void test_simd(
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    simdgroup_half8x8 a;
    simdgroup_half8x8 b;
    simdgroup_float8x8 c;
    simdgroup_multiply_accumulate(c, a, b, c);
}
"#;
    match runner.compile(simd_msl, "test_simd") {
        Ok(_) => println!("simdgroup kernel compiled OK"),
        Err(e) => println!("simdgroup kernel compile error: {}", e),
    }
}
