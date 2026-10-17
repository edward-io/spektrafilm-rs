struct Params {n:u32,groups:u32,_pad:vec2<u32>,value:vec4<f32>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> partial:array<f32>;
@group(0) @binding(2) var<storage,read_write> scale:array<f32>;
@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    if gid.x>=3u {return;}let c=gid.x;var before=0.;var after=0.;
    for(var g=0u;g<p.groups;g++){before+=partial[g*6u+c];after+=partial[g*6u+c+3u];}
    scale[c]=select(1.,before/max(after,1e-30),after>0.);
}
