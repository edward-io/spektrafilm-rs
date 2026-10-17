struct Params { shape:vec4<u32> }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> input:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read> kernel:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read_write> output:array<vec2<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let i=g.x+g.y*groups.x*256u;let n=p.shape.x*p.shape.y;if i>=n {return;}
 let x=reverseBits(i%p.shape.x)>>(32u-p.shape.z);let y=reverseBits(i/p.shape.x)>>(32u-p.shape.w);let j=y*p.shape.x+x;
 let a=input[j];let b=kernel[j];output[i]=vec2<f32>(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x)/(f32(n)*max(kernel[0].x,1e-30));
}
