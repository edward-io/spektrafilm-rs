struct Params { header:vec4<u32>, source_scale:vec4<f32>, destination_scale:vec4<f32> }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let pixel=g.x+g.y*groups.x*256u;
 if pixel>=p.header.x {return;}
 for(var c=0u;c<3u;c++) {let i=pixel*3u+c;output[i]=p.destination_scale[c]*output[i]+p.source_scale[c]*input[i];}
}
