// Tensor Hermite interpolation with linearly mixed orthogonal slopes and
// the reference per-cell corner bounds. Coordinates outside the LUT clamp.
struct Params { n_pixels:u32, size:u32, pad0:u32, pad1:u32,
    data_min:vec4<f32>, inv:vec4<f32>, }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read> nodes:array<vec4<f32>>; // value, dx, dy, dz
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
fn node(i:u32,j:u32,k:u32,c:u32)->vec4<f32> {return nodes[((i*p.size+j)*p.size+k)*3u+c];}
fn hm(a:f32,b:f32,da:f32,db:f32,t:f32)->f32 {
    let t2=t*t;let t3=t2*t;
    return (2.0*t3-3.0*t2+1.0)*a+(t3-2.0*t2+t)*da+(-2.0*t3+3.0*t2)*b+(t3-t2)*db;
}
fn lm(a:f32,b:f32,t:f32)->f32 {return a+t*(b-a);}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) groups:vec3<u32>) {
    let pixel=gid.x+gid.y*groups.x*256u;if pixel>=p.n_pixels {return;}
    let base=pixel*3u;
    let v=vec3<f32>(input[base],input[base+1u],input[base+2u]);
    let coord=clamp((v-p.data_min.xyz)*p.inv.xyz,vec3<f32>(0.0),vec3<f32>(f32(p.size-1u)));
    let index=min(vec3<u32>(floor(coord)),vec3<u32>(p.size-2u));let t=coord-vec3<f32>(index);
    for(var c=0u;c<3u;c++) {
        let a=node(index.x,index.y,index.z,c);let b=node(index.x+1u,index.y,index.z,c);
        let d=node(index.x,index.y+1u,index.z,c);let e=node(index.x+1u,index.y+1u,index.z,c);
        let f=node(index.x,index.y,index.z+1u,c);let g=node(index.x+1u,index.y,index.z+1u,c);
        let h=node(index.x,index.y+1u,index.z+1u,c);let j=node(index.x+1u,index.y+1u,index.z+1u,c);
        let vz0=hm(hm(a.x,b.x,a.y,b.y,t.x),hm(d.x,e.x,d.y,e.y,t.x),lm(a.z,b.z,t.x),lm(d.z,e.z,t.x),t.y);
        let vz1=hm(hm(f.x,g.x,f.y,g.y,t.x),hm(h.x,j.x,h.y,j.y,t.x),lm(f.z,g.z,t.x),lm(h.z,j.z,t.x),t.y);
        let sz0=lm(lm(a.w,b.w,t.x),lm(d.w,e.w,t.x),t.y);
        let sz1=lm(lm(f.w,g.w,t.x),lm(h.w,j.w,t.x),t.y);
        let lo=min(min(min(a.x,b.x),min(d.x,e.x)),min(min(f.x,g.x),min(h.x,j.x)));
        let hi=max(max(max(a.x,b.x),max(d.x,e.x)),max(max(f.x,g.x),max(h.x,j.x)));
        output[base+c]=clamp(hm(vz0,vz1,sz0,sz1,t.z),lo,hi);
    }
}
