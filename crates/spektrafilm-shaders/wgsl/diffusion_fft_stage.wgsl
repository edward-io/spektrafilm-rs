// Radix-2 butterflies on already bit-reversed rows/columns. Twiddles are
// generated in f64 once, avoiding large-angle f32 sin/cos phase error.
struct Params { shape:vec4<u32>, direction:vec4<u32> }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> input:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read_write> output:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read> twiddles:array<vec2<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let i=g.x+g.y*groups.x*256u;let n=p.shape.x*p.shape.y;if i>=n/2u {return;}
 let vertical=p.direction.x!=0u;let length=select(p.shape.x,p.shape.y,vertical);let per_line=length/2u;let line=select(i/per_line,i%p.shape.x,vertical);let j=select(i%per_line,i/p.shape.x,vertical);let half_span=p.shape.z/2u;
 let a_line=(j/half_span)*p.shape.z+j%half_span;let b_line=a_line+half_span;
 let ia=select(line*p.shape.x+a_line,a_line*p.shape.x+line,vertical);let ib=select(line*p.shape.x+b_line,b_line*p.shape.x+line,vertical);
 var t=twiddles[(j%half_span)*(p.shape.w/p.shape.z)];if p.direction.y!=0u {t.y= -t.y;}
 let a=input[ia];let b=input[ib];let v=vec2<f32>(b.x*t.x-b.y*t.y,b.x*t.y+b.y*t.x);
 output[ia]=a+v;output[ib]=a-v;
}
