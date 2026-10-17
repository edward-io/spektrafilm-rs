// One physical emulsion sublayer. All three dye channels are independent.
// PCG streams preserve the model's statistics; f64 export retains NumPy RNG.
struct Params {
    n_pixels: u32, n_rows: u32, layer: u32, flags: u32,
    density_min: vec4<f32>, density_max: vec4<f32>,
    particles: vec4<f32>, uniformity: vec4<f32>,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read> density: array<f32>;
@group(0) @binding(2) var<storage,read> curves: array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> sampled: array<f32>;
@group(0) @binding(4) var<storage,read> replay:array<f32>;
@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pixel=gid.x;if pixel>=p.n_pixels{return;}
    for(var ch=0u;ch<3u;ch++) {
        var q=density[pixel*3u+ch];if (p.flags&1u)!=0u {q=-q;}
        var lo=0u;var hi=p.n_rows;
        loop {if lo>=hi {break;}let mid=(lo+hi)/2u;if curves[mid*3u+ch].x<=q {lo=mid+1u;}else{hi=mid;}}
        var value=curves[ch][p.layer+1u];
        if lo>=p.n_rows {value=curves[(p.n_rows-1u)*3u+ch][p.layer+1u];}
        else if lo>0u {let a=curves[(lo-1u)*3u+ch];let b=curves[lo*3u+ch];let t=(q-a.x)/max(1e-20,b.x-a.x);value=mix(a[p.layer+1u],b[p.layer+1u],t);}
        let dmax=p.density_max[ch];let npp=p.particles[ch];
        if dmax<=0.0 || npp<=0.0 {sampled[pixel*3u+ch]=0.0;continue;}
        let prob=clamp((value+p.density_min[ch])/dmax,1e-6,0.999999);
        let sat=1.0-prob*p.uniformity[ch]*0.999999;
        var seed_ch=ch;if (p.flags&2u)!=0u {seed_ch=0u;}
        var rng=hash(pixel+0x9e3779b9u)^hash(seed_ch+p.layer*10u+1u);
        // Poisson thinning is an exact identity: Binomial(Poisson(lambda),p)
        // has Poisson(lambda*p) law. It avoids an approximate normal binomial
        // on the default exact-statistics path and needs fewer random draws.
        var count=0.;
        if (p.flags&8u)!=0u {count=replay[(p.layer*p.n_pixels+pixel)*3u+ch];}
        else if (p.flags&4u)!=0u {let seeds=fast_poisson(npp/sat,&rng);count=binomial(seeds,prob,&rng);}
        else {count=exact_poisson(npp/sat*prob,&rng);}
        sampled[pixel*3u+ch]=count*(dmax/npp)*sat;
    }
}
