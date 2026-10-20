// Native canvas interpolation and paper development. Kernel definitions follow
// Vispy's spatial filters (BSD-2-Clause; see data/licenses/vispy-spatial-filters.txt).
struct Params {
    uv:vec4<f32>, // visible subrectangle of the source texture
    options:vec4<f32>, // opacity, kernel, development time (-1 = final), crossfade
    maxima:vec4<f32>, // channel density maxima, neutral max
    chroma_max:vec4<f32>,
    coefficients0:vec4<f32>, // neutral, yellow, magenta, cyan
    coefficients1:vec4<f32>, // receiver, silver, anti, warm
    flags:vec4<f32>, // stain, linear framebuffer, unused, unused
    transform0:vec4<f32>, // optional affine map from uv: x row, y row
    transform1:vec4<f32>, // offset, enabled (pixels outside the image are transparent)
}
@group(0) @binding(0) var image:texture_2d<f32>;
@group(0) @binding(1) var previous:texture_2d<f32>;
@group(0) @binding(2) var<uniform> p:Params;
struct Vertex { @builtin(position) position:vec4<f32>, @location(0) uv:vec2<f32> }
@vertex fn vs_main(@builtin(vertex_index) i:u32)->Vertex {
    let corner=array<vec2<f32>,3>(vec2<f32>(0.,0.),vec2<f32>(2.,0.),vec2<f32>(0.,2.))[i];
    return Vertex(vec4<f32>(corner*vec2<f32>(2.,-2.)+vec2<f32>(-1.,1.),0.,1.),mix(p.uv.xy,p.uv.zw,corner));
}
fn sinc(x:f32)->f32 {if abs(x)<1e-6 {return 1.;}let y=x*3.141592653589793;return sin(y)/y;}
fn weight(value:f32,mode:u32)->f32 {
    let x=abs(value);
    if mode==1u {return max(0.,1.-x);}
    if mode==2u { // Vispy Cubic.weight polynomial; normalized below.
        if x<2. {return pow(2.-x,3.)/6.;}return 0.;
    }
    if mode==3u {
        if x<1. {return ((x-1.8)*x-0.2)*x+1.;}
        if x<2. {let t=x-1.;return ((-t/3.+0.8)*t-7./15.)*t;}return 0.;
    }
    if mode==4u {
        if x<1. {return ((13./11.*x-453./209.)*x-3./209.)*x+1.;}
        if x<2. {let t=x-1.;return ((-6./11.*t+270./209.)*t-156./209.)*t;}
        if x<3. {let t=x-2.;return ((t/11.-45./209.)*t+26./209.)*t;}return 0.;
    }
    if x>=4. {return 0.;}
    if mode==5u {return sinc(x)*sinc(x/4.);}
    let t=x*3.141592653589793/4.;return sinc(x)*(0.42+0.5*cos(t)+0.08*cos(2.*t));
}
fn sample_image(uv:vec2<f32>,old:bool)->vec3<f32> {
    let size=vec2<i32>(textureDimensions(image));
    let position=uv*vec2<f32>(size)-0.5;
    let mode=u32(p.options.y);
    if mode==0u {
        let at=clamp(vec2<i32>(floor(position+0.5)),vec2<i32>(0),size-1);
        if old {return textureLoad(previous,at,0).rgb;}return textureLoad(image,at,0).rgb;
    }
    var radius=1;if mode==2u || mode==3u {radius=2;}if mode==4u {radius=3;}if mode>=5u {radius=4;}
    let base=vec2<i32>(floor(position));
    var result=vec3<f32>(0.);var total=0.;
    for(var y=1-radius;y<=radius;y++) {
        let wy=weight(position.y-f32(base.y+y),mode);
        var row=vec3<f32>(0.);var norm=0.;
        for(var x=1-radius;x<=radius;x++) {
            let wx=weight(position.x-f32(base.x+x),mode);
            let at=clamp(base+vec2<i32>(x,y),vec2<i32>(0),size-1);
            var value=textureLoad(image,at,0).rgb;
            if old {value=textureLoad(previous,at,0).rgb;}
            row+=wx*value;norm+=wx;
        }
        result+=wy*row;total+=wy*norm;
    }
    return result/max(total,1e-8);
}
fn developing(rgb:vec3<f32>)->vec3<f32> {
    let density=clamp(-2.2*log(clamp(rgb,vec3<f32>(0.0001),vec3<f32>(1.))),vec3<f32>(0.),vec3<f32>(6.));
    let neutral=min(density.x,min(density.y,density.z));let chroma=density-neutral;
    let density_latent=density/(p.maxima.xyz+0.0001);let chroma_latent=chroma/(p.chroma_max.xyz+0.0001);
    let neutral_latent=neutral/(p.maxima.w+0.0001);
    let shadow=clamp(dot(density,vec3<f32>(0.34,0.39,0.27)),0.,1.75)/1.75;let highlight=1.-shadow;
    let warmth=clamp((chroma.y+1.25*chroma.z-0.85*chroma.x)/(chroma.x+chroma.y+chroma.z+0.0001),0.,1.);
    let chroma_strength=max(chroma_latent.x,max(chroma_latent.y,chroma_latent.z));
    let receiver=(0.3+0.7*highlight)*vec3<f32>(0.04,0.08,0.15);
    let silver_map=clamp(0.24+0.46*shadow+0.30*neutral_latent,0.,1.);
    let silver=clamp(silver_map*vec3<f32>(0.18,0.19,0.20)+chroma_latent.yzx*vec3<f32>(0.03,0.02,0.04),vec3<f32>(0.),vec3<f32>(0.28));
    let anti=(highlight*vec3<f32>(0.10,0.06,0.02)+(1.-density_latent.zxy)*vec3<f32>(0.13,0.07,0.18)+chroma_latent.yzx*vec3<f32>(0.04,0.02,0.06))*(0.40+0.60*highlight);
    let warm=(0.24+0.42*highlight+0.34*max(warmth,chroma_strength))*vec3<f32>(0.02,0.08,0.19);
    let stain=highlight*vec3<f32>(0.03,0.06,0.11);
    let d=neutral*p.coefficients0.x+chroma*vec3<f32>(p.coefficients0.w,p.coefficients0.z,p.coefficients0.y)+receiver*p.coefficients1.x+silver*p.coefficients1.y+anti*p.coefficients1.z+warm*p.coefficients1.w+stain*p.flags.x;
    return exp(-clamp(d,vec3<f32>(0.),vec3<f32>(8.))/2.2);
}
@fragment fn fs_main(in:Vertex)->@location(0) vec4<f32> {
    var uv=in.uv;
    if p.transform1.z>0.5 {
        uv=vec2<f32>(dot(p.transform0.xy,in.uv),dot(p.transform0.zw,in.uv))+p.transform1.xy;
        if any(uv<vec2<f32>(0.)) || any(uv>vec2<f32>(1.)) {return vec4<f32>(0.);}
    }
    var rgb=sample_image(uv,false);
    if p.options.z>=0. {rgb=developing(rgb);}
    if p.options.w<1. {rgb=mix(sample_image(uv,true),rgb,p.options.w);}
    rgb=clamp(rgb,vec3<f32>(0.),vec3<f32>(1.));
    // egui composites in gamma space; match it on either framebuffer format.
    rgb*=p.options.x;
    if p.flags.y>0. {rgb=select(pow((rgb+0.055)/1.055,vec3<f32>(2.4)),rgb/12.92,rgb<vec3<f32>(0.04045));}
    return vec4<f32>(rgb,p.options.x);
}
