// Centered reflected image tiles, or centered PSF; bit-reverse both axes for DIT.
struct Params { shape:vec4<u32>, frame:vec4<u32>, tile:vec4<u32> }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> source:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<vec2<f32>>;
fn mirror(x:i32,n:i32)->u32 {if n<=1 {return 0u;}let period=2*(n-1);let q=((x%period)+period)%period;return u32(select(q,period-q,q>=n));}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let i=g.x+g.y*groups.x*256u;if i>=p.shape.x*p.shape.y {return;}
 let x=reverseBits(i%p.shape.x)>>(32u-p.shape.z);let y=reverseBits(i/p.shape.x)>>(32u-p.shape.w);
 let r=i32(p.frame.z);var value=0.0;
 if p.tile.w==0u {
  let sx=mirror(i32(p.tile.x)+i32(x)-r,i32(p.frame.x));let sy=mirror(i32(p.tile.y)+i32(y)-r,i32(p.frame.y));
  value=source[(sy*p.frame.x+sx)*3u+p.frame.w];
 }else{
  let dx=select(i32(x),i32(x)-i32(p.shape.x),x>p.shape.x/2u);let dy=select(i32(y),i32(y)-i32(p.shape.y),y>p.shape.y/2u);
  if abs(dx)<=r && abs(dy)<=r {
   let radius=sqrt(f32(dx*dx+dy*dy));
   for(var k=0u;k<p.tile.z;k++) {let lambda=max(source[k*4u],1e-6);let weight=source[k*4u+1u+p.frame.w];value+=weight*exp(-radius/lambda)/(6.283185307179586*lambda*lambda);}
  }
 }
 output[i]=vec2<f32>(value,0.0);
}
