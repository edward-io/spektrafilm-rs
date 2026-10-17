struct Params {n_pixels:u32,micro:u32,_p0:u32,_p1:u32,density_min:vec4<f32>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> sum:array<f32>;
@group(0) @binding(2) var<storage,read> clumping:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {let i=gid.x;if i>=p.n_pixels{return;}for(var c=0u;c<3u;c++){let k=i*3u+c;var value=sum[k];if p.micro!=0u {value*=clumping[k];}output[k]=value;}}
