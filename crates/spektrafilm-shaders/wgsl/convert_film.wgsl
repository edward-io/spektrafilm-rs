// Damped Gauss-Newton inversion of the spectral scan, bounded by dye capacity.
struct Params {
    n:u32,bands:u32,lut:u32,pad:u32,
    maximum:vec4<f32>,midpoint:vec4<f32>,log_rgb:vec4<f32>,
    seed0:vec4<f32>,seed1:vec4<f32>,seed2:vec4<f32>,
    cal0:vec4<f32>,cal1:vec4<f32>,cal2:vec4<f32>,
}
struct Band {dye:vec4<f32>,rgb:vec4<f32>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read> bands:array<Band>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
fn solve(m:mat3x3<f32>,v:vec3<f32>)->vec3<f32> {
    let c0=cross(m[1],m[2]);let c1=cross(m[2],m[0]);let c2=cross(m[0],m[1]);
    let det=dot(m[0],c0);if abs(det)<1e-30 {return vec3<f32>(0.);}
    return vec3<f32>(dot(c0,v),dot(c1,v),dot(c2,v))/det;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
    let pixel=gid.x+gid.y*groups.x*256u;if pixel>=p.n {return;}
    let i=pixel*3u;let rgb=vec3<f32>(input[i],input[i+1u],input[i+2u]);
    let y=rgb.x*p.cal0.xyz+rgb.y*p.cal1.xyz+rgb.z*p.cal2.xyz;
    var x=y;
    if p.lut==0u {
        let seed=mat3x3<f32>(p.seed0.xyz,p.seed1.xyz,p.seed2.xyz);
        x=clamp(p.midpoint.xyz+seed*(p.log_rgb.xyz-log(max(y,vec3<f32>(1e-12)))/log(10.)),vec3<f32>(0.),p.maximum.xyz);
        for(var iteration=0u;iteration<8u;iteration++) {
            var f=vec3<f32>(0.);var j=mat3x3<f32>(vec3<f32>(0.),vec3<f32>(0.),vec3<f32>(0.));
            for(var w=0u;w<p.bands;w++) {
                let d=bands[w].dye.xyz;let v=bands[w].rgb.xyz*exp(-log(10.)*dot(d,x));f+=v;
                j[0]-=log(10.)*v*d.x;j[1]-=log(10.)*v*d.y;j[2]-=log(10.)*v*d.z;
            }
            let residual=f-y;if all(abs(residual)<vec3<f32>(1e-6)) {break;}
            let jt=transpose(j);var normal=jt*j;
            normal[0].x+=1e-7;normal[1].y+=1e-7;normal[2].z+=1e-7;
            x=clamp(x-solve(normal,jt*residual),vec3<f32>(0.),p.maximum.xyz);
        }
    }
    output[i]=x.x;output[i+1u]=x.y;output[i+2u]=x.z;
}
