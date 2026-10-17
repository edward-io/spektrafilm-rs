// Per-channel FIR, with the reference's radius rounding and reflected edges.
struct Params { shape:vec4<u32>, modes:vec4<u32>, radius:vec4<u32>, coefficient_hi:array<vec4<f32>,3>, coefficient_lo:array<vec4<f32>,3> }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read> kernel:array<f32>;
@group(0) @binding(4) var<storage,read_write> forward_lo:array<f32>;
fn reflect(index:i32,n:u32)->i32 {
 if index>=0i && index<i32(n) {return index;}
 if index<0i && index>=-i32(n) {return -index-1i;}
 if index>=i32(n) && index<2i*i32(n) {return 2i*i32(n)-1i-index;}
 let period=2i*i32(n);var value=index%period;if value<0i {value+=period;}
 if value<i32(n) {return value;}return period-1i-value;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let pixel=g.x+g.y*groups.x*256u;
 let n=p.shape.x*p.shape.y;if pixel>=n {return;}
 let x=pixel%p.shape.x;let y=pixel/p.shape.x;
 if all(p.modes.xyz==vec3<u32>(0u)) && p.radius.x==p.radius.y && p.radius.x==p.radius.z {
  let r=i32(p.radius.x);var sum=vec3<f32>(0.0);
  for(var k= -r;k<=r;k++) {
   var sx=i32(x);var sy=i32(y);
   if p.shape.z==0u {sx=reflect(sx+k,p.shape.x);}else{sy=reflect(sy+k,p.shape.y);}
   let i=(u32(sy)*p.shape.x+u32(sx))*3u;let j=u32(k+r);
   sum+=vec3<f32>(input[i],input[i+1u],input[i+2u])*vec3<f32>(kernel[j],kernel[p.shape.w+j],kernel[2u*p.shape.w+j]);
  }
  output[pixel*3u]=sum.x;output[pixel*3u+1u]=sum.y;output[pixel*3u+2u]=sum.z;return;
 }
 for(var c=0u;c<3u;c++) {
  if p.modes[c]!=0u {continue;}
  let r=i32(p.radius[c]);var sum=0.0;
  for(var k= -r;k<=r;k++) {
   var sx=i32(x);var sy=i32(y);
   if p.shape.z==0u {sx=reflect(sx+k,p.shape.x);}else{sy=reflect(sy+k,p.shape.y);}
   sum+=input[(u32(sy)*p.shape.x+u32(sx))*3u+c]*kernel[c*p.shape.w+u32(k+r)];
  }
  output[pixel*3u+c]=sum;
 }
}
