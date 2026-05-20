use metaltile_bench::runner::GpuRunner;

fn main() {
    let runner = GpuRunner::new().expect("Failed to create GpuRunner");
    
    let msl = r#"
#include <metal_stdlib>
using namespace metal;
#if __METAL_VERSION__ >= 310
#include <metal_simdgroup_matrix>
#endif

kernel void test_layout(
    device float* out [[buffer(0)]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    simdgroup_float8x8 m = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    thread float2& elems = (thread float2&)m.thread_elements();
    elems.x = float(simd_lane) * 2.0f;
    elems.y = float(simd_lane) * 2.0f + 1.0f;
    
    simdgroup_store(m, out, 8, ulong2(0,0), false);
}
"#;
    
    let kernel = runner.compile(msl, "test_layout").expect("compile failed");
    let buf = runner.buffer_zeros(64 * 4); // 64 floats
    let raw = runner.measure(&kernel, &[&buf], [1,1,1], [32,1,1], 0, 1);
    let bytes = runner.read_bytes(&buf, 64 * 4);
    
    let floats: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    
    println!("simdgroup_store layout (8x8 float matrix, stride=8):");
    for row in 0..8 {
        let start = row * 8;
        println!("Row {}: {:?}", row, &floats[start..start+8]);
    }
}
