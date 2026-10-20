// Camera exposure and input tone, before the spectral front pass.
//
// Mirrors `spektrafilm_math::tone::ToneCurve::apply`: a self-guided filter of
// log luminance gives each pixel a regional base, and Shadows, Highlights and
// Whites use that base. Stages, one dispatch each:
//   0  e = stops(luminance(gain * rgb))                       -> scratch[0, n)
//   1  horizontal box mean of (e, e^2)                        -> pairs
//   2  vertical box mean, guided coefficients (a, b)          -> scratch[n, 3n)
//   3  horizontal box mean of (a, b)                          -> pairs
//   4  vertical box mean, base = a*e + b, apply to the image
//   5  per-pixel curve (base = e), when no regional control is active
struct Params {
    n_pixels: u32,
    width: u32,
    height: u32,
    radius: u32,
    stage: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    controls: vec4<f32>, // Shadows, Highlights, Whites, Blacks
    luma_gain: vec4<f32>,
    // Highlight strength, detail epsilon, log floor (stops).
    constants: vec4<f32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> image: array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(3) var<storage, read_write> pairs: array<vec2<f32>>;

const GRAY: f32 = 0.184;

fn bell(x: f32) -> f32 {
    let t = clamp(x, 0.0, 1.0);
    return 16.0 * t * t * (1.0 - t) * (1.0 - t);
}
fn transition(x: f32) -> f32 {
    let t = clamp(x, 0.0, 1.0);
    return clamp(t * t * t * (10.0 + t * (-15.0 + 6.0 * t)), 0.0, 1.0);
}
fn finite(v: f32) -> bool {
    return v == v && abs(v) <= 3.4e38;
}

fn camera_rgb(i: u32) -> vec3<f32> {
    let base = i * 3u;
    return vec3<f32>(image[base], image[base + 1u], image[base + 2u]) * params.luma_gain.w;
}
fn stops(y: f32) -> f32 {
    let floor = GRAY * exp2(params.constants.z);
    return log2(max(abs(y), floor) / GRAY);
}
fn delta(e: f32) -> f32 {
    let c = params.controls;
    let ds = c.x * bell((e + 5.0) / 5.0);
    let e1 = e + ds;
    let dh = params.constants.x * c.y * bell(e1 / 3.0);
    return ds + dh + 1.2 * c.z * transition((e1 + dh - 0.5) / 3.5);
}

// Box mean of `pairs` (stage 1 reads e from scratch instead) along one axis.
fn horizontal(i: u32, from_e: bool, offset: u32) -> vec2<f32> {
    let w = params.width;
    let x = i % w;
    let row = i - x;
    let x0 = select(0u, x - params.radius, x >= params.radius);
    let x1 = min(x + params.radius, w - 1u);
    var sum = vec2<f32>(0.0);
    for (var k = x0; k <= x1; k++) {
        if from_e {
            let v = scratch[row + k];
            sum += vec2<f32>(v, v * v);
        } else {
            let j = offset + 2u * (row + k);
            sum += vec2<f32>(scratch[j], scratch[j + 1u]);
        }
    }
    return sum / f32(x1 - x0 + 1u);
}
fn vertical(i: u32) -> vec2<f32> {
    let w = params.width;
    let x = i % w;
    let y = i / w;
    let y0 = select(0u, y - params.radius, y >= params.radius);
    let y1 = min(y + params.radius, params.height - 1u);
    var sum = vec2<f32>(0.0);
    for (var k = y0; k <= y1; k++) {
        sum += pairs[k * w + x];
    }
    return sum / f32(y1 - y0 + 1u);
}

fn finish(i: u32, base: f32) {
    var rgb = camera_rgb(i);
    let y = dot(params.luma_gain.xyz, rgb);
    if !finite(y) {
        return;
    }
    let gain = exp2(delta(base));
    rgb *= gain;
    let c = params.controls;
    let toe = GRAY / 8.0;
    let magnitude = abs(y * gain);
    if c.w != 0.0 && magnitude < toe {
        let offset = c.w * (toe / 3.0) * (1.0 - transition(magnitude / toe));
        if c.w > 0.0 {
            rgb += vec3<f32>(offset);
        } else if magnitude > 0.0 {
            rgb *= max(0.0, magnitude + offset) / magnitude;
        } else {
            rgb = vec3<f32>(0.0);
        }
    }
    let o = i * 3u;
    image[o] = rgb.x;
    image[o + 1u] = rgb.y;
    image[o + 2u] = rgb.z;
}

@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_pixels { return; }
    let n = params.n_pixels;
    switch params.stage {
        case 0u: {
            let y = dot(params.luma_gain.xyz, camera_rgb(i));
            // Invalid pixels act as middle gray in the base.
            scratch[i] = select(0.0, stops(y), finite(y));
        }
        case 1u: {
            pairs[i] = horizontal(i, true, 0u);
        }
        case 2u: {
            let m = vertical(i);
            let variance = max(m.y - m.x * m.x, 0.0);
            let a = variance / (variance + params.constants.y);
            scratch[n + 2u * i] = a;
            scratch[n + 2u * i + 1u] = m.x * (1.0 - a);
        }
        case 3u: {
            pairs[i] = horizontal(i, false, n);
        }
        case 4u: {
            let ab = vertical(i);
            finish(i, ab.x * scratch[i] + ab.y);
        }
        default: {
            finish(i, stops(dot(params.luma_gain.xyz, camera_rgb(i))));
        }
    }
}
