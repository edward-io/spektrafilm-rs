struct Params {n:u32,size:u32,pad:vec2<u32>,minimum:vec4<f32>,inv:vec4<f32>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read> nodes:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
fn at(q:vec3<u32>)->vec3<f32> {let i=((q.x*p.size+q.y)*p.size+q.z)*3u;return vec3<f32>(nodes[i],nodes[i+1u],nodes[i+2u]);}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
    let pixel=gid.x+gid.y*groups.x*256u;if pixel>=p.n {return;}let i=pixel*3u;
    let q=clamp((vec3<f32>(input[i],input[i+1u],input[i+2u])-p.minimum.xyz)*p.inv.xyz,vec3<f32>(0.),vec3<f32>(f32(p.size-1u)));
    let index=min(vec3<u32>(floor(q)),vec3<u32>(p.size-2u));let f=q-vec3<f32>(index);var order=vec3<u32>(0u,1u,2u);
    for(var a=0u;a<2u;a++){for(var b=a+1u;b<3u;b++){if f[order[b]]>f[order[a]] {let t=order[a];order[a]=order[b];order[b]=t;}}}
    let a=at(index);var offset=vec3<u32>(0u);offset[order.x]=1u;let b=at(index+offset);offset[order.y]=1u;let c=at(index+offset);let d=at(index+vec3<u32>(1u));
    let v=a+f[order.x]*(b-a)+f[order.y]*(c-b)+f[order.z]*(d-c);
    output[i]=v.x;output[i+1u]=v.y;output[i+2u]=v.z;
}
