// Reversal receiver saturation follows spatial diffusion.
struct Params { n_pixels:u32, positive:u32, _pad:vec2<u32>, knee:vec4<f32>, reference:vec4<f32> }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> correction:array<f32>;
@group(0) @binding(2) var<storage,read_write> log_raw:array<f32>;
@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    if gid.x>=p.n_pixels {return;}
    for(var c=0u;c<3u;c++) {
        let i=3u*gid.x+c;var v=correction[i];let k=p.knee[c];
        if p.positive!=0u && k<3.402823e38 {v=select(0.,v*(k+p.reference[c])/(k+v),k+v!=0.);}
        log_raw[i]-=v;
    }
}
