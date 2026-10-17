// Shared exact Poisson and explicit fast-statistics particle samplers.
fn hash(x:u32)->u32 {var s=x*747796405u+2891336453u;let w=((s>>((s>>28u)+4u))^s)*277803737u;return (w>>22u)^w;}
fn uniform(r:ptr<function,u32>)->f32 {*r=hash(*r);return clamp(f32(*r)*(1.0/4294967296.0),1e-7,0.9999999);}
fn normal(r:ptr<function,u32>)->f32 {return sqrt(-2.0*log(uniform(r)))*cos(6.28318530718*uniform(r));}
fn fast_poisson(lam:f32,r:ptr<function,u32>)->f32 {
    if lam<30.0 {let limit=exp(-lam);var product=1.0;var k=0u;loop {product*=uniform(r);if product<=limit {break;}k++;if k>256u {break;}}return f32(k);}
    return max(0.0,round(lam+sqrt(lam)*normal(r)));
}
// Stable Poisson log PMF. Direct lgamma subtraction loses the tail at the
// very large particle counts used by small previews. The deviance series
// avoids cancellation near the mean; Stirling's error is accurate for k>=10.
fn poisson_logpmf(k:f32,lam:f32)->f32 {
    if k<10. {
        var factorial=0.;for(var i=2u;i<=u32(k);i++){factorial+=log(f32(i));}
        return -lam+k*log(lam)-factorial;
    }
    let delta=(k-lam)/lam;
    var deviance=k*log(k/lam)+lam-k;
    if abs(delta)<0.1 {
        let series=0.5+delta*(-1./6.+delta*(1./12.+delta*(-1./20.+delta*(1./30.+delta*(-1./42.+delta*(1./56.+delta*(-1./72.+delta*(1./90.-delta/110.))))))));
        deviance=lam*delta*delta*series;
    }
    let inv=1./k;let inv2=inv*inv;
    let error=inv*(1./12.-inv2*(1./360.-inv2*(1./1260.-inv2/1680.)));
    return -deviance-error-0.5*log(6.283185307179586*k);
}
fn exact_poisson(lam:f32,r:ptr<function,u32>)->f32 {
    if lam<=0. {return 0.;}
    if lam<10. {return fast_poisson(lam,r);}
    // Hormann PTRS transformed rejection, as used by NumPy's exact sampler.
    let b=0.931+2.53*sqrt(lam);let a=-0.059+0.02483*b;
    let inv_alpha=1.1239+1.1328/(b-3.4);let vr=0.9277-3.6224/(b-2.);
    loop {
        let u=uniform(r)-0.5;let v=uniform(r);let us=0.5-abs(u);
        let k=floor((2.*a/us+b)*u+lam+0.43);
        if us>=0.07 && v<=vr {return max(0.,k);}
        if k<0. || (us<0.013 && v>us) {continue;}
        if log(v)+log(inv_alpha)-log(a/(us*us)+b)<=poisson_logpmf(k,lam) {return k;}
    }
    return 0.;
}
fn binomial(n:f32,prob:f32,r:ptr<function,u32>)->f32 {
    let mean=n*prob;let variance=mean*(1.-prob);
    // Match use_fast_stats: normal approximation only above its variance
    // threshold; inverse CDF handles the sparse / near-saturated tails.
    if variance>10. {return clamp(round(mean+sqrt(variance)*normal(r)),0.,n);}
    let p_minor=min(prob,1.-prob);let q=1.-p_minor;
    var logq=log(q);
    if p_minor<0.01 {let p2=p_minor*p_minor;logq=-p_minor-0.5*p2-p2*p_minor/3.-p2*p2/4.;}
    var mass=exp(n*logq);var u=uniform(r);var k=0.;
    loop {
        if u<=mass || k>=n {break;}
        u-=mass;k+=1.;mass*=((n-k+1.)*p_minor)/(k*q);
    }
    if prob>0.5 {return n-k;}return k;
}
