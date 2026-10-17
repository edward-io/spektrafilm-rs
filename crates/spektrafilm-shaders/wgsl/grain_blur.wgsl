// Per-channel Gaussian kernel, shared by dye clouds, clumping and final blur.
struct Params {width:u32,height:u32,axis:u32,stride:u32,radii:vec4<u32>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> source:array<f32>;
@group(0) @binding(2) var<storage,read> kernels:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let pixel=gid.x;if pixel>=p.width*p.height{return;}let x=pixel%p.width;let y=pixel/p.width;
 for(var c=0u;c<3u;c++){let radius=i32(p.radii[c]);var sum=0.0;
  for(var k=-radius;k<=radius;k++){var sx=i32(x);var sy=i32(y);if p.axis==0u {sx=clamp(sx+k,0,i32(p.width)-1);}else{sy=clamp(sy+k,0,i32(p.height)-1);}
   sum+=source[(u32(sy)*p.width+u32(sx))*3u+c]*kernels[c*p.stride+u32(k+radius)];}
  output[pixel*3u+c]=sum;
 }
}
