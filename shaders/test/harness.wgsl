// test/harness.wgsl: the bindings every test kernel shares (crates/shaders/tests/library.rs).
// Test code only: no runtime pipeline includes it.
//
// Input and output are arrays of 4-word elements; kernels bitcast to f32 where they need floats, so
// one bind-group layout serves every kernel.

struct Params {
    // Elements to process.
    count: u32,
    // A kernel-specific selector (which function, which mapper).
    mode: u32,
    // Kernel-specific scalars.
    a: f32,
    b: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<u32>>;

// Input element `i` as floats.
fn input_f32(i: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(input[i]);
}

// Stores floats as output element `i`.
fn store_f32(i: u32, v: vec4<f32>) {
    output[i] = bitcast<vec4<u32>>(v);
}
