struct Params { source: vec2<u32>, destination: vec2<u32>,
    red: vec4<f32>, green: vec4<f32>, blue: vec4<f32>,
    transform: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
fn encode(x: f32) -> f32 {
    if x <= 0.0031308 { return 12.92 * x; }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}
fn sample(x: u32, y: u32) -> vec3<f32> {
    let i = (y * p.source.x + x) * 3u;
    let rgb = vec3<f32>(input[i], input[i+1u], input[i+2u]);
    if p.transform == 0u { return rgb; }
    return vec3<f32>(encode(dot(p.red.xyz, rgb)), encode(dot(p.green.xyz, rgb)), encode(dot(p.blue.xyz, rgb)));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * 256u;
    if i >= p.destination.x * p.destination.y { return; }
    let xy = vec2<u32>(i % p.destination.x, i / p.destination.x);
    var rgb: vec3<f32>;
    if all(p.source == p.destination) {
        rgb = sample(xy.x, xy.y);
    } else {
        let pos = (vec2<f32>(xy) + 0.5) * (vec2<f32>(p.source) / vec2<f32>(p.destination)) - 0.5;
        let lo = vec2<u32>(clamp(floor(pos), vec2<f32>(0.0), vec2<f32>(p.source - 1u)));
        let hi = min(lo + 1u, p.source - 1u);
        let t = clamp(pos - vec2<f32>(lo), vec2<f32>(0.0), vec2<f32>(1.0));
        let top = sample(lo.x, lo.y) + (sample(hi.x, lo.y) - sample(lo.x, lo.y)) * t.x;
        let bot = sample(lo.x, hi.y) + (sample(hi.x, hi.y) - sample(lo.x, hi.y)) * t.x;
        rgb = top + (bot - top) * t.y;
    }
    let bytes = vec3<u32>(floor(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0 + 0.5));
    output[i] = bytes.x | (bytes.y << 8u) | (bytes.z << 16u) | 0xff000000u;
}
