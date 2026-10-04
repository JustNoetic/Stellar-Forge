// Integrate scattering and oblate projected solid angle together, in host
// equatorial radii. The ring and spheroid share their pole.
const float RS_PI = 3.14159265358979323846;
const float RS_NODE[16] = float[16](
    -0.98940093499,-0.94457502307,-0.86563120239,-0.75540440836,
    -0.61787624440,-0.45801677766,-0.28160355078,-0.09501250984,
     0.09501250984,0.28160355078,0.45801677766,0.61787624440,
     0.75540440836,0.86563120239,0.94457502307,0.98940093499);
const float RS_WEIGHT[16] = float[16](
    0.02715245941,0.06225352394,0.09515851168,0.12462897126,
    0.14959598882,0.16915651940,0.18260341504,0.18945061046,
    0.18945061046,0.18260341504,0.16915651940,0.14959598882,
    0.12462897126,0.09515851168,0.06225352394,0.02715245941);

float ringshine_hg(float g, float mu) {
    g=clamp(g,-0.99,0.99);
    return (1.0-g*g)/(4.0*RS_PI*pow(max(1e-6,1.0+g*g-2.0*g*mu),1.5));
}
float ringshine_cs(float g, float mu) {
    g=clamp(g,-0.99,0.99);
    return ringshine_hg(g,mu)*1.5*(1.0+mu*mu)/(2.0+g*g);
}
float ringshine_absorbed(float depth) {
    return depth<0.001 ? depth*(1.0-depth*(0.5-depth/6.0)) : 1.0-exp(-depth);
}
float ringshine_radiance(float mu_v,float sun,float theta,float alpha,
                        vec4 props,bool textured,bool lit) {
    float mu_0=abs(sun);
    if(mu_0<1e-7 || alpha<=0.0) return 0.0;
    mu_v=max(mu_v,1e-7);
    float tau=-log(max(1e-7,1.0-min(alpha,0.9999999)));
    float vd=tau/mu_v,ld=tau/mu_0,single;
    if(lit) {
        single=mu_0/(mu_v+mu_0)*ringshine_absorbed(vd+ld);
    } else {
        // Stable difference of exponentials, including coincident depths.
        float delta=abs(ld-vd);
        float ratio=delta<0.001 ? 1.0-0.5*delta+delta*delta/6.0
                               : (1.0-exp(-delta))/delta;
        single=vd*exp(-min(vd,ld))*ratio;
    }
    float pf,ms_weight;
    if(textured) {
        float chunks=clamp((alpha-0.1)/0.5,0.0,1.0);
        float forward=mix(0.95,0.5,chunks);
        pf=forward*ringshine_hg(props.r,theta)+(1.0-forward)*ringshine_hg(props.g,theta);
        ms_weight=chunks;
    } else {
        float balance=clamp(props.b,0.0,1.0),forward=mix(0.1,0.9,balance);
        pf=forward*ringshine_cs(props.r,theta)+(1.0-forward)*ringshine_cs(props.g,theta);
        ms_weight=clamp(1.0-balance*0.7,0.1,1.0);
    }
    if(!lit) return single*pf*props.a;
    float gamma=sqrt(0.08);
    float Hv=(1.0+2.0*mu_v)/(1.0+2.0*mu_v*gamma);
    float H0=(1.0+2.0*mu_0)/(1.0+2.0*mu_0*gamma);
    float multiple=0.92*mu_0/(mu_v+mu_0)*max(0.0,Hv*H0-1.0)
                    *ringshine_absorbed(vd+ld)/(4.0*RS_PI);
    return single*pf+multiple*ms_weight;
}
float ringshine_arc(float lo,float hi,float r,float slat,float clat,
                    float rho,float eta,float sun,float phi,float alpha,
                    vec4 props,bool textured,bool lit) {
    if(hi<=lo) return 0.0;
    float horizontal=r-rho*clat,height=rho*slat;
    float d0sq=horizontal*horizontal+height*height,rc=r*rho*clat;
    float scale=sqrt(d0sq/max(rc,1e-12));
    // Concentrate nodes at the closest ring element without a fixed LUT grid.
    float b0=atan(lo/scale),b1=atan(hi/scale);
    float half_span=0.5*(b1-b0),middle=0.5*(b1+b0);
    float normal_length=sqrt(clat*clat+slat*slat*eta*eta);
    float cos_sun=sqrt(max(0.0,1.0-sun*sun)),total=0.0;
    for(int i=0;i<16;++i) {
        float beta=middle+half_span*RS_NODE[i],tb=tan(beta),a=scale*tb;
        float hs=sin(0.5*a),ca=cos(a),sa=sin(a);
        float d2=max(1e-16,d0sq+4.0*rc*hs*hs),d=sqrt(d2);
        float horizon=(r*clat-1.0/rho)-2.0*r*clat*hs*hs;
        float mu_p=max(0.0,horizon/(normal_length*d)),mu_v=height/d;
        float theta=((rho*clat-r*ca)*cos_sun*cos(phi)-r*sa*cos_sun*sin(phi)
                     +(lit ? -1.0 : 1.0)*height*abs(sun))/d;
        float L=ringshine_radiance(mu_v,sun,clamp(theta,-1.0,1.0),alpha,props,textured,lit);
        total+=RS_WEIGHT[i]*L*mu_p*mu_v/d2*r*scale*(1.0+tb*tb);
    }
    return total*half_span;
}
float ringshine_band(float r,float signed_slat,float flattening,float sun,
                     float phi,float alpha,vec4 props,bool textured) {
    float slat=abs(signed_slat);
    if(slat<1e-7 || alpha<=0.0 || abs(sun)<1e-7) return 0.0;
    float clat=sqrt(max(0.0,1.0-slat*slat));
    float ff=1.0-flattening,eta=1.0/(ff*ff);
    float rho=ff/sqrt(ff*ff*clat*clat+slat*slat);
    if(r*rho*clat<=1.0) return 0.0;
    float limit=acos(clamp(1.0/(r*rho*clat),0.0,1.0));
    bool lit=signed_slat*sun>=0.0;
    float cos_sun=sqrt(max(0.0,1.0-sun*sun)),shadow=0.0;
    if(cos_sun>1e-7) {
        float arg=sqrt(max(0.0,(1.0-1.0/(r*r))*(cos_sun*cos_sun+sun*sun*eta)))/cos_sun;
        if(arg<1.0) shadow=acos(clamp(arg,0.0,1.0));
    }
    // Exact lit arcs of the oblate host: no spherical shadow-CDF weights.
    float cursor=-limit,total=0.0;
    for(int turn=-1;turn<=1;++turn) {
        float start=phi+float(turn)*2.0*RS_PI-shadow,end=start+2.0*shadow;
        if(end<=cursor || start>=limit || shadow==0.0) continue;
        total+=ringshine_arc(cursor,min(start,limit),r,slat,clat,rho,eta,sun,phi,alpha,props,textured,lit);
        cursor=max(cursor,end);
        if(cursor>=limit) return total;
    }
    return total+ringshine_arc(cursor,limit,r,slat,clat,rho,eta,sun,phi,alpha,props,textured,lit);
}
