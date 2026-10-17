struct Params {n_pixels:u32,monochrome:u32,_p0:u32,_p1:u32,variance:vec4<f32>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@group(0) @binding(2) var<storage,read> replay:array<f32>;
fn hash(x:u32)->u32 {var s=x*747796405u+2891336453u;let w=((s>>((s>>28u)+4u))^s)*277803737u;return (w>>22u)^w;}
@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let pixel=gid.x;if pixel>=p.n_pixels{return;}
 if p._p0!=0u {for(var c=0u;c<3u;c++){output[pixel*3u+c]=replay[p._p1+pixel*3u+c];}return;}
 for(var c=0u;c<3u;c++){var lane=c;if p.monochrome!=0u {lane=0u;}
 let seed=hash(pixel+777u+lane*0x9e3779b9u);let u=max(1e-7,f32(hash(seed))*(1.0/4294967296.0));let v=f32(hash(seed+1u))*(1.0/4294967296.0);
 let z=sqrt(-2.0*log(u))*cos(6.28318530718*v);output[pixel*3u+c]=exp(-0.5*p.variance.x+sqrt(p.variance.x)*z);}
}
