// Power ratio and workgroup sums, without a CPU readback.
struct Params {n:u32,groups:u32,_pad:vec2<u32>,value:vec4<f32>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read> blurred:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<storage,read_write> partial:array<f32>;
var<workgroup> sums:array<f32,6144>;
@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(workgroup_id) group:vec3<u32>) {
    for(var c=0u;c<3u;c++) {
        var d=0.;var v=0.;
        if gid.x<p.n {let i=gid.x*3u+c;d=max(input[i],0.);if d>0. {v=pow(max(d,1e-6),1.+p.value.x)/pow(max(blurred[i],1e-6),p.value.x);}output[i]=v;}
        sums[local*6u+c]=d;sums[local*6u+c+3u]=v;
    }
    workgroupBarrier();
    var stride=512u;
    loop {
        if local<stride {for(var c=0u;c<6u;c++){sums[local*6u+c]+=sums[(local+stride)*6u+c];}}
        workgroupBarrier();if stride==1u {break;}stride/=2u;
    }
    if local==0u {for(var c=0u;c<6u;c++){partial[group.x*6u+c]=sums[c];}}
}
