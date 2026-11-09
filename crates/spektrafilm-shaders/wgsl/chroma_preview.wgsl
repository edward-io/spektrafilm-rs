// Multi-pass counterpart of spektrafilm_math::chroma. Scratch planes stay
// resident and the packed linear RGB buffer is read back after reconstruction.
struct Params {
    size: vec2u,
    operation: u32,
    radius: i32,
    direction: vec2i,
    step: i32,
    _pad: u32,
    values: vec4f,
    weights: array<vec4f, 9>,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<vec4f>;
@group(0) @binding(2) var<storage, read> b: array<vec4f>;
@group(0) @binding(3) var<storage, read> c: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> rgb: array<f32>;

fn reflect_index(x: i32, n: i32) -> u32 {
    if x >= 0 && x < n { return u32(x); }
    let period = 2 * n;
    let i = ((x % period) + period) % period;
    return u32(min(i, period - 1 - i));
}

fn index(xy: vec2i) -> u32 {
    return reflect_index(xy.y, i32(p.size.y)) * p.size.x + reflect_index(xy.x, i32(p.size.x));
}

fn weight(k: u32) -> f32 {
    return p.weights[k / 4u][k % 4u];
}

fn shift(xy: vec2i, delta: vec2f) -> vec2f {
    let f = floor(delta);
    let t = delta - f;
    let pt = xy + vec2i(f);
    let top = mix(a[index(pt)].xy, a[index(pt + vec2i(1, 0))].xy, t.x);
    let bottom = mix(a[index(pt + vec2i(0, 1))].xy, a[index(pt + vec2i(1, 1))].xy, t.x);
    return mix(top, bottom, t.y);
}

@compute @workgroup_size(16, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= p.size) { return; }
    let xy = vec2i(id.xy);
    let i = id.y * p.size.x + id.x;
    let j = 3u * i;
    switch p.operation {
        case 0u: { // Luminance and opponent deltas.
            let pixel = vec3f(rgb[j], rgb[j + 1u], rgb[j + 2u]);
            let y = dot(pixel, p.values.xyz);
            dst[i] = select(vec4f(0.0), vec4f(y, pixel.r - y, pixel.b - y, 0.0), y > 0.0);
        }
        case 1u: { // One finite Gaussian pass, also used for tangent filtering.
            var sum = vec4f(0.0);
            for (var k = -p.radius; k <= p.radius; k++) {
                sum += weight(u32(k + p.radius)) * a[index(xy + p.direction * k)];
            }
            dst[i] = sum;
        }
        case 2u: { // Normalize chroma by local luminance.
            dst[i] = vec4f(a[i].yz / max(b[i].x, 0.002), 0.0, 0.0);
        }
        case 3u: {
            let residual = a[i].xy - b[i].xy;
            dst[i] = vec4f(sqrt(0.5 * dot(residual, residual)), 0.0, 0.0, 0.0);
        }
        case 4u: { // Six partial selection passes find the median of eleven taps.
            var samples: array<f32, 11>;
            for (var k = 0; k < 11; k++) {
                samples[k] = a[index(xy + p.direction * (k - 5))].x;
            }
            for (var k = 0; k < 6; k++) {
                for (var l = k + 1; l < 11; l++) {
                    let lo = min(samples[k], samples[l]);
                    samples[l] = max(samples[k], samples[l]);
                    samples[k] = lo;
                }
            }
            dst[i] = vec4f(samples[5], 0.0, 0.0, 0.0);
        }
        case 5u: { // Guide, with optional directional color protection.
            var local = b[i].xy;
            if p.values.z != 0.0 { local = mix(local, c[i].yz, c[i].x); }
            dst[i] = vec4f(log2(max(a[i].x, 0.002)) / p.values.x, local / p.values.y, 0.0);
        }
        case 6u: { // One joint bilateral pass.
            let center = b[i].xyz;
            var total = 0.0;
            var sum = vec2f(0.0);
            var tap = 0u;
            for (var k = -p.radius; k <= p.radius; k += p.step) {
                let neighbor = index(xy + p.direction * k);
                let delta = b[neighbor].xyz - center;
                let w = weight(tap) * exp(-0.5 * dot(delta, delta));
                total += w;
                sum += w * a[neighbor].xy;
                tap++;
            }
            dst[i] = vec4f(sum / total, 0.0, 0.0);
        }
        case 7u: { // Normal derivative and protected color for this direction.
            let before = shift(xy, p.values.xy);
            let after = shift(xy, -p.values.xy);
            dst[i] = vec4f(0.5 * (after - before), 0.82 * a[i].xy + 0.09 * (before + after));
        }
        case 8u: {
            let signal = length(a[i].xy - b[i].xy) / (0.02 + 1.2 * c[i].x);
            dst[i] = vec4f(smoothstep(1.3, 2.15, signal), 0.0, 0.0, 0.0);
        }
        case 9u: { // Square part of the directional dilation footprint.
            var score = 0.0;
            for (var y = -1; y <= 1; y++) {
                for (var x = -1; x <= 1; x++) {
                    score = max(score, a[index(xy + vec2i(x, y))].x);
                }
            }
            dst[i] = vec4f(score, 0.0, 0.0, 0.0);
        }
        case 10u: { // Extend evidence along the normal and keep the best direction.
            var best = dst[i].x;
            var edge_color = dst[i].yz;
            for (var k = -p.radius; k <= p.radius; k++) {
                let score = a[index(xy + p.direction * k)].x;
                if score > best {
                    best = score;
                    edge_color = b[i].zw;
                }
            }
            dst[i] = vec4f(best, edge_color, 0.0);
        }
        case 11u: {
            let wide = mix(a[i].xy, b[i].xy, 0.96);
            dst[i] = vec4f(mix(wide, c[i].yz, c[i].x), 0.0, 0.0);
        }
        case 12u: { // Restore RGB without changing luminance or clipping highlights.
            let strength = smoothstep(0.030, 0.075, 1.2 * c[i].x);
            let pixel = vec3f(rgb[j], rgb[j + 1u], rgb[j + 2u]);
            if strength <= 0.0 || any(pixel < vec3f(0.0)) { return; }
            let y = dot(pixel, p.values.xyz);
            let delta = mix(pixel.rb - vec2f(y), a[i].xy * max(b[i].x, 0.002), strength);
            let green = -(p.values.x * delta.x + p.values.z * delta.y) / p.values.y;
            let bound = min(1.0, y / max(1e-10, -min(green, min(delta.x, delta.y))));
            let result = vec3f(y) + vec3f(delta.x, green, delta.y) * bound;
            rgb[j] = result.r;
            rgb[j + 1u] = result.g;
            rgb[j + 2u] = result.b;
        }
        default: {}
    }
}
