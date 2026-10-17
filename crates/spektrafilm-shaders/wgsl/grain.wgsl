// Exact-law simple grain; independent deterministic GPU streams. The optional
// post blur uses the same reference FIR/IIR filters as physical layered grain.
struct Params {
    n_pixels: u32,
    base_seed: u32,
    n_sub_layers: u32,
    // 1 = B&W monochrome grain: all lanes share the channel-0 RNG stream,
    // producing one native monochrome emulsion. 0 = independent RGB streams.
    monochrome: u32,
    density_min: vec4<f32>,            // .xyz used
    density_max: vec4<f32>,            // .xyz used (already includes density_min)
    n_particles_per_pixel: vec4<f32>,  // .xyz used (already divided by n_sub_layers)
    grain_uniformity: vec4<f32>,       // .xyz used
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> density_cmy: array<f32>;

@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let idx = gid.x+gid.y*groups.x*1024u;
    if idx >= params.n_pixels { return; }
    let base = idx * 3u;

    let n_sl_f = f32(params.n_sub_layers);
    let n_sl = i32(params.n_sub_layers);

    for (var ch_i: i32 = 0; ch_i < 3; ch_i++) {
        let ch = u32(ch_i);
        let dmin = params.density_min[ch];
        let dmax = params.density_max[ch];
        let npp = params.n_particles_per_pixel[ch];
        let gu = params.grain_uniformity[ch];
        if dmax<=0.0 || npp<=0.0 { density_cmy[base+ch]=0.0; continue; }
        let od_particle = dmax / npp;

        // CPU adds density_min before grain calc, subtracts after.
        let d_in = density_cmy[base + ch] + dmin;
        let p_raw = d_in / dmax;
        let p = clamp(p_raw, 1e-6, 1.0 - 1e-6);
        let saturation = 1.0 - p * gu * (1.0 - 1e-6);
        let lambda = npp / saturation;

        var sum: f32 = 0.0;
        for (var sl: i32 = 0; sl < n_sl; sl++) {
            // Matches the CPU stream split: seed = ch + sl*10, then mix
            // with pixel index. Use SplitMix to decorrelate small ints
            // before xor-combining. Monochrome (B&W) shares channel 0's
            // stream across all lanes.
            var seed_ch = u32(ch_i);
            if params.monochrome != 0u {
                seed_ch = 0u;
            }
            let layer_seed = seed_ch + u32(sl) * 10u + params.base_seed;
            var rng = hash(layer_seed+1u) ^ hash(idx+0x9e3779b9u);

            // Exact Poisson thinning preserves the full Poisson-binomial law.
            let developed = exact_poisson(lambda*p, &rng);
            sum = sum + developed * od_particle * saturation;
        }

        density_cmy[base + ch] = sum / n_sl_f - dmin;
    }
}
