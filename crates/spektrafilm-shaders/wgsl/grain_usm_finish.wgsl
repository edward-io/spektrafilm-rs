struct Params {n:u32,enabled:u32,_pad:vec2<u32>,floor:vec4<f32>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> scale:array<f32>;
@group(0) @binding(2) var<storage,read> sharpened:array<f32>;
@group(0) @binding(3) var<storage,read_write> density:array<f32>;
@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    if gid.x>=p.n {return;}
    for(var c=0u;c<3u;c++){let i=gid.x*3u+c;var v=density[i];if p.enabled!=0u {v=sharpened[i]*scale[c];}density[i]=v-p.floor[c];}
}
