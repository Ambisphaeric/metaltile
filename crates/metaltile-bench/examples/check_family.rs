use metaltile_bench::runner::GpuRunner;
use objc2_metal::{MTLDevice, MTLGPUFamily};

fn main() {
    let runner = GpuRunner::new().expect("Failed to create GpuRunner");
    let dev = &runner.inner.device;
    let families = [
        (MTLGPUFamily::Apple1, "Apple1"),
        (MTLGPUFamily::Apple2, "Apple2"),
        (MTLGPUFamily::Apple3, "Apple3"),
        (MTLGPUFamily::Apple4, "Apple4"),
        (MTLGPUFamily::Apple5, "Apple5"),
        (MTLGPUFamily::Apple6, "Apple6"),
        (MTLGPUFamily::Apple7, "Apple7"),
        (MTLGPUFamily::Apple8, "Apple8"),
        (MTLGPUFamily::Apple9, "Apple9"),
    ];
    for (fam, name) in families {
        println!("{}: {}", name, dev.supportsFamily(fam));
    }
}
