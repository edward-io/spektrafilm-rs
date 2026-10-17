struct Params { shape:vec4<u32>, frame:vec4<u32>, tile:vec4<u32>, mix:vec4<f32> }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> convolution:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read> original:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
 let i=g.x+g.y*groups.x*256u;if i>=p.tile.z*p.tile.w {return;}
 let x=i%p.tile.z;let y=i/p.tile.z;let dst=((p.tile.y+y)*p.frame.x+p.tile.x+x)*3u+p.frame.w;
 let blurred=convolution[(y+p.frame.z)*p.shape.x+x+p.frame.z].x;
 output[dst]=(1.0-p.mix.x)*original[dst]+p.mix.x*blurred;
}
