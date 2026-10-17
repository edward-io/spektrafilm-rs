// Named transfer functions from spektrafilm-math/transfer.rs, unclipped.
// Exposure scale acts before decoding, as in Python's preprocess stage.
struct Params { n_pixels: u32, mode: u32, encode: u32, _pad: u32,
    scale: f32, _pad1: f32, _pad2: f32, _pad3: f32,
    controls: vec4<f32>, } // x=gamma, y=predecode scale
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> rgb: array<f32>;
fn lg(x: f32) -> f32 { return log2(x) * 0.3010299956639812; }
fn p10(x: f32) -> f32 { return exp2(x * 3.321928094887362); }
fn signed_pow(x: f32, p: f32) -> f32 { return sign(x) * pow(abs(x), p); }
fn bt709_encode(x: f32) -> f32 {
    if x < 0.018 { return 4.5*x; } return 1.099*pow(x,0.45)-0.099;
}
fn ties_even(x: f32) -> f32 {
    let f=floor(x); let d=x-f;
    if d < 0.5 { return f; } if d > 0.5 { return f+1.0; }
    return f + f32(u32(f) & 1u);
}
fn encode(x: f32) -> f32 {
    switch params.mode {
        case 0u: { return x; }
        case 1u: { if x<=0.0031308 { return 12.92*x; } return 1.055*pow(x,1.0/2.4)-0.055; }
        case 2u: { if x<1.0/512.0 { return 16.0*x; } return pow(x,1.0/1.8); }
        case 3u: { return bt709_encode(x); }
        case 4u: { return signed_pow(x,1.0/params.controls.x); }
        case 5u: {
            if x<=0.0 { return (-16.0+9.72)/17.52; }
            if x<exp2(-15.0) { return (log2(exp2(-16.0)+x*0.5)+9.72)/17.52; }
            return (log2(x)+9.72)/17.52;
        }
        case 6u: { if x<=0.0078125 { return 10.5402377416545*x+0.0729055341958355; } return (log2(x)+9.72)/17.52; }
        case 7u: { if x<0.01 { return 5.6*x+0.125; } return 0.241514*lg(x+0.00873)+0.598206; }
        case 8u: { if x>=0.01125 { return (420.0+(lg(x+0.01)-lg(0.19))*261.5)/1023.0; } return (x*(171.2102946929-95.0)/0.01125+95.0)/1023.0; }
        case 9u: { if x>0.010591 { return 0.247190*lg(5.555556*x+0.052272)+0.385537; } return 5.367655*x+0.092809; }
        case 10u: {
            let a=(262144.0-16.0)/117.45; let b=928.0/1023.0; let c=95.0/1023.0;
            let s=7.0*0.6931471805599453*exp2(7.0-14.0*c/b)/(a*b); let t=(exp2(6.0-14.0*c/b)-64.0)/a;
            if x<t { return (x-t)/s; } return (log2(a*x+64.0)-6.0)/14.0*b+c;
        }
        case 11u: { if x<=0.00262409 { return x*10.44426855; } return (log2(x+0.0075)+7.0)*0.07329248; }
        case 12u: { if x<0.005 { return 8.283605932402494*x+0.09246575342465753; } return 0.08692876065491224*log(x+0.005494072432257808)+0.5300133392291939; }
        case 13u: { if x<0.00089 { return 8.735631*x+0.092864; } return 0.344676*lg(0.555556*x+0.009468)+0.790453; }
        case 14u: { if x<0.000889 { return 8.799461*x+0.092864; } return 0.245281*lg(5.555556*x+0.064829)+0.384316; }
        case 15u: {
            let v=x/0.9;
            if v < -0.014 { return -0.36726845*lg(-v*14.98325+1.0)+0.12783901; }
            if v<=0.014 { return 1.9754798*v+0.12512219; } return 0.36726845*lg(v*14.98325+1.0)+0.12240537;
        }
        case 16u: { let v=x+0.01; if v<0.0 { return v*15.1927; } return 0.224282*lg(v*155.975327+1.0); }
        case 17u: { if x>=0.01 { return 0.08550479*log2(x+0.00964052)+0.69336945; } if x >= -0.05641088 { return 47.28711236*(x+0.05641088)*(x+0.05641088); } return 0.0; }
        case 18u: { if x<=0.0078 { return 6.025*x+0.0929; } return lg(x*0.9892+0.0108)*0.256663+0.584555; }
        case 19u: { if x<0.328 { return 0.635386119257087*signed_pow(x+0.0075,1.0/3.0); } return 0.1466275659824047*log(x)+0.6050830889540567; }
        case 20u: { let y=pow(max(x/10000.0,0.0),2610.0/16384.0); return pow((3424.0/4096.0+2413.0/128.0*y)/(1.0+2392.0/128.0*y),2523.0/32.0); }
        case 21u: { let e=signed_pow(x/1000.0,1.0/1.2); if e<=1.0/12.0 { return sqrt(max(3.0*e,0.0)); } return 0.17883277*log(12.0*e-0.28466892)+0.55991073; }
        case 22u: { if x>exp2(-9.72) { return clamp(ties_even((log2(x)+2.5)*50.0+425.0),64.0,940.0)/1023.0; } return 64.0/1023.0; }
        case 23u: { return signed_pow(x/52.37,1.0/2.6); }
        case 24u: { let black=p10((95.0-685.0)/300.0); return (685.0+300.0*lg(x*(1.0-black)+black))/1023.0; }
        case 25u: { let span=log(316.2/0.001); let toe=2.718281828459045*0.001; if x<0.0 { return 0.0; } if x<=toe { return x/(toe*span); } return log(x/0.001)/span; }
        case 26u: { if x<0.0 { return params.controls.z*x+0.075; } return log(x+params.controls.w)*params.controls.y+params.controls.x; }
        case 27u: { return log(1.0+112.0*x)/log(113.0); }
        case 28u: { return bt709_encode(max(x,0.0))/(1.099*pow(2.0,0.45)-0.099); }
        case 29u: { let v=x*155.0/219.0/0.9; var y=v*5.0+0.030001222851889303; if v>=0.0 { y=0.432699*lg(v+0.037584)+0.616596+0.03; } return (y*876.0+64.0)/1023.0; }
        case 30u: { if x<0.0228 { return 4.0*x; } return 1.1115*pow(x,0.45)-0.1115; }
        case 31u: { if x>216.0/24389.0 { return (116.0*signed_pow(x,1.0/3.0)-16.0)/100.0; } return (24389.0/27.0)*x/100.0; }
        default: { return x; }
    }
}
fn decode(x: f32) -> f32 {
    switch params.mode {
        case 0u: { return x; }
        case 1u: { if x<=0.04045 { return x/12.92; } return pow((x+0.055)/1.055,2.4); }
        case 2u: { if x<0.03125 { return x/16.0; } return pow(x,1.8); }
        case 3u: { if x<0.081 { return x/4.5; } return pow((x+0.099)/1.099,1.0/0.45); }
        case 4u: { return signed_pow(x,params.controls.x); }
        case 5u: { if x<(-15.0+9.72)/17.52 { return (exp2(x*17.52-9.72)-exp2(-16.0))*2.0; } return min(exp2(x*17.52-9.72),65504.0); }
        case 6u: { if x<=0.155251141552511 { return (x-0.0729055341958355)/10.5402377416545; } return exp2(x*17.52-9.72); }
        case 7u: { if x<0.181 { return (x-0.125)/5.6; } return p10((x-0.598206)/0.241514)-0.00873; }
        case 8u: { if x*1023.0>=171.2102946929 { return p10((x*1023.0-420.0)/261.5)*0.19-0.01; } return (x*1023.0-95.0)*0.01125/(171.2102946929-95.0); }
        case 9u: { if x>5.367655*0.010591+0.092809 { return (p10((x-0.385537)/0.247190)-0.052272)/5.555556; } return (x-0.092809)/5.367655; }
        case 10u: { let a=(262144.0-16.0)/117.45; let b=928.0/1023.0; let c=95.0/1023.0; let s=7.0*0.6931471805599453*exp2(7.0-14.0*c/b)/(a*b); let t=(exp2(6.0-14.0*c/b)-64.0)/a; if x<0.0 { return x*s+t; } return (exp2(14.0*(x-c)/b+6.0)-64.0)/a; }
        case 11u: { if x<=0.02740668 { return x/10.44426855; } return exp2(x/0.07329248-7.0)-0.0075; }
        case 12u: { if x<0.005*8.283605932402494+0.09246575342465753 { return (x-0.09246575342465753)/8.283605932402494; } return exp((x-0.5300133392291939)/0.08692876065491224)-0.005494072432257808; }
        case 13u: { if x<0.100537775223865 { return (x-0.092864)/8.735631; } return (p10((x-0.790453)/0.344676)-0.009468)/0.555556; }
        case 14u: { if x<0.100686685370811 { return (x-0.092864)/8.799461; } return (p10((x-0.384316)/0.245281)-0.064829)/5.555556; }
        case 15u: { if x<0.097465473 { return -0.9*(p10((0.12783901-x)/0.36726845)-1.0)/14.98325; } if x<=0.15277891 { return 0.9*(x-0.12512219)/1.9754798; } return 0.9*(p10((x-0.12240537)/0.36726845)-1.0)/14.98325; }
        case 16u: { if x<0.0 { return x/15.1927-0.01; } return (p10(x/0.224282)-1.0)/155.975327-0.01; }
        case 17u: { if x>=47.28711236*(0.01+0.05641088)*(0.01+0.05641088) { return exp2((x-0.69336945)/0.08550479)-0.00964052; } if x>=0.0 { return sqrt(x/47.28711236)-0.05641088; } return -0.05641088; }
        case 18u: { if x<=0.14 { return (x-0.0929)/6.025; } return (p10(3.89616*x-2.27752)-0.0108)/0.9892; }
        case 19u: { if x<0.4418377321603128 { let v=x/0.635386119257087; return v*v*v-0.0075; } return exp((x-0.6050830889540567)/0.1466275659824047); }
        case 20u: { let p=pow(max(x,0.0),32.0/2523.0); return 10000.0*pow(max(p-3424.0/4096.0,0.0)/(2413.0/128.0-2392.0/128.0*p),16384.0/2610.0); }
        case 21u: { var e=x*x/3.0; if x>0.5 { e=(exp((x-0.55991073)/0.17883277)+0.28466892)/12.0; } return 1000.0*pow(e,1.2); }
        case 22u: { return exp2((x*1023.0-425.0)/50.0-2.5); }
        case 23u: { return 52.37*signed_pow(x,2.6); }
        case 24u: { let black=p10((95.0-685.0)/300.0); return (p10((1023.0*x-685.0)/300.0)-black)/(1.0-black); }
        case 25u: { let span=log(316.2/0.001); if x<=1.0/span { return span*x*2.718281828459045*0.001; } return exp(x*span+log(0.001)); }
        case 26u: { if x<0.075 { return (x-0.075)/params.controls.z; } return exp((x-params.controls.x)/params.controls.y)-params.controls.w; }
        case 27u: { return (pow(113.0,x)-1.0)/112.0; }
        case 28u: { let v=x*(1.099*pow(2.0,0.45)-0.099); if x<bt709_encode(0.018)/(1.099*pow(2.0,0.45)-0.099) { return v/4.5; } return pow((v+0.099)/1.099,1.0/0.45); }
        case 29u: { let v=(x*1023.0-64.0)/876.0; var y=(v-0.030001222851889303)/5.0; if x>=( (0.432699*lg(0.037584)+0.616596+0.03)*876.0+64.0)/1023.0 { y=p10((v-0.616596-0.03)/0.432699)-0.037584; } return 219.0*y*0.9/155.0; }
        case 30u: { if x<1.1115*pow(0.0228,0.45)-0.1115 { return x/4.0; } return pow((x+0.1115)/1.1115,1.0/0.45); }
        case 31u: { if x>0.08 { let v=(x*100.0+16.0)/116.0; return v*v*v; } return x*100.0/(24389.0/27.0); }
        default: { return x; }
    }
}
@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i=gid.x+gid.y*groups.x*1024u; if i>=params.n_pixels { return; }
    for(var c=0u;c<3u;c++) {
        let j=i*3u+c;
        if params.encode!=0u { rgb[j]=encode(rgb[j]); }
        else { rgb[j]=decode(rgb[j]*params_scale()); }
    }
}
fn params_scale() -> f32 { return params.scale; }
