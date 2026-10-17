// Young–van Vliet recurrence, one independent line/channel per invocation.
// Double-single states avoid coefficient cancellation at large sigma. Only
// the stored image is rounded to f32; no CPU image passes or GPU readbacks.
struct Params { shape:vec4<u32>, modes:vec4<u32>, radius:vec4<u32>, coefficient_hi:array<vec4<f32>,3>, coefficient_lo:array<vec4<f32>,3> }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read> kernel:array<f32>;
@group(0) @binding(4) var<storage,read_write> forward_lo:array<f32>;
// Recover an exactly representable nearby subtraction from the IEEE mantissas.
// Metal's fast-math reassociation otherwise cancels the error terms in TwoSum.
fn nearby_difference(s:f32,a:f32)->f32 {
 if s==a {return 0.0;}
 let us=bitcast<u32>(s);let ua=bitcast<u32>(a);
 let es=i32((us>>23u)&255u);let ea=i32((ua>>23u)&255u);let e=min(es,ea);
 let ms=i32((us&0x7fffffu)|0x800000u)<<u32(es-e);
 let ma=i32((ua&0x7fffffu)|0x800000u)<<u32(ea-e);
 let value=ldexp(f32(ms-ma),e-150);
 return select(value,-value,(ua&0x80000000u)!=0u);
}
fn two_sum(a:f32,b:f32)->vec2<f32> {
 let s=a+b;let big=select(b,a,abs(a)>=abs(b));let small=select(a,b,abs(a)>=abs(b));
 if big==0.0 || (sign(big)!=sign(small) && abs(small)>=0.5*abs(big)) {return vec2<f32>(s,0.0);}
 return vec2<f32>(s,small-nearby_difference(s,big));
}
fn add(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {
 let sum=two_sum(a.x,b.x);return two_sum(sum.x,(sum.y+a.y)+b.y);
}
fn mul(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {
 let v=a.x*b.x;let e=fma(a.x,b.x,-v)+a.x*b.y+a.y*b.x;return two_sum(v,e);
}
fn index(line:u32,c:u32,j:u32)->u32 {
 if p.shape.z==0u {return (line*p.shape.x+j)*3u+c;}
 return (j*p.shape.x+line)*3u+c;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let c=g.x%3u;let line=g.x/3u;let lines=select(p.shape.x,p.shape.y,p.shape.z==0u);
 if line>=lines || p.modes[c]==0u {return;}
 let n=select(p.shape.y,p.shape.x,p.shape.z==0u);
 let hi=p.coefficient_hi[c];let lo=p.coefficient_lo[c];
 let a=vec2<f32>(hi.x,lo.x);let b1=vec2<f32>(hi.y,lo.y);let b2=vec2<f32>(hi.z,lo.z);let b3=vec2<f32>(hi.w,lo.w);
 var w1=vec2<f32>(input[index(line,c,0u)],0.0);var w2=w1;var w3=w1;
 for(var j=0u;j<n;j++) {
  let i=index(line,c,j);let v=add(add(add(mul(a,vec2<f32>(input[i],0.0)),mul(b1,w1)),mul(b2,w2)),mul(b3,w3));
  output[i]=v.x;forward_lo[i]=v.y;w3=w2;w2=w1;w1=v;
 }
 let end=index(line,c,n-1u);w1=vec2<f32>(output[end],forward_lo[end]);w2=w1;w3=w1;
 for(var j=i32(n)-1;j>=0;j--) {
  let i=index(line,c,u32(j));let v=add(add(add(mul(a,vec2<f32>(output[i],forward_lo[i])),mul(b1,w1)),mul(b2,w2)),mul(b3,w3));
  output[i]=v.x+v.y;w3=w2;w2=w1;w1=v;
 }
}
