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

// Caller samples the original segment materials at each radial node.
vec3 ringshine_material_radiance(float mu_v,float sun,float mu,bool lit);
vec3 ringshine_arc(float lo,float hi,float r,float slat,float clat,
                    float rho,float eta,float sun,float phi,bool lit) {
    if(hi<=lo) return vec3(0.0);
    float horizontal=r-rho*clat,height=rho*slat;
    float d0sq=horizontal*horizontal+height*height,rc=r*rho*clat;
    float scale=sqrt(d0sq/max(rc,1e-12));
    // Concentrate nodes at the closest ring element without a fixed LUT grid.
    float b0=atan(lo/scale),b1=atan(hi/scale);
    float half_span=0.5*(b1-b0),middle=0.5*(b1+b0);
    float normal_length=sqrt(clat*clat+slat*slat*eta*eta);
    float cos_sun=sqrt(max(0.0,1.0-sun*sun));
    vec3 total=vec3(0.0);
    for(int i=0;i<16;++i) {
        float beta=middle+half_span*RS_NODE[i],tb=tan(beta),a=scale*tb;
        float hs=sin(0.5*a),ca=cos(a),sa=sin(a);
        float d2=max(1e-16,d0sq+4.0*rc*hs*hs),d=sqrt(d2);
        float horizon=(r*clat-1.0/rho)-2.0*r*clat*hs*hs;
        float mu_p=max(0.0,horizon/(normal_length*d)),mu_v=height/d;
        float theta=((rho*clat-r*ca)*cos_sun*cos(phi)-r*sa*cos_sun*sin(phi)
                     +(lit ? -1.0 : 1.0)*height*abs(sun))/d;
        vec3 L=ringshine_material_radiance(mu_v,sun,clamp(theta,-1.0,1.0),lit);
        total+=RS_WEIGHT[i]*L*mu_p*mu_v/d2*r*scale*(1.0+tb*tb);
    }
    return total*half_span;
}
vec3 ringshine_band(float r,float signed_slat,float flattening,float sun,
                     float phi) {
    float slat=abs(signed_slat);
    if(slat<1e-7 || sun==0.0) return vec3(0.0);
    float clat=sqrt(max(0.0,1.0-slat*slat));
    float ff=1.0-flattening,eta=1.0/(ff*ff);
    float rho=ff/sqrt(ff*ff*clat*clat+slat*slat);
    if(r*rho*clat<=1.0) return vec3(0.0);
    float limit=acos(clamp(1.0/(r*rho*clat),0.0,1.0));
    bool lit=signed_slat*sun>=0.0;
    float cos_sun=sqrt(max(0.0,1.0-sun*sun)),shadow=0.0;
    if(cos_sun>1e-7) {
        float arg=sqrt(max(0.0,(1.0-1.0/(r*r))*(cos_sun*cos_sun+sun*sun*eta)))/cos_sun;
        if(arg<1.0) shadow=acos(clamp(arg,0.0,1.0));
    }
    // Exact lit arcs of the oblate host: no spherical shadow-CDF weights.
    float cursor=-limit;
    vec3 total=vec3(0.0);
    for(int turn=-1;turn<=1;++turn) {
        float start=phi+float(turn)*2.0*RS_PI-shadow,end=start+2.0*shadow;
        if(end<=cursor || start>=limit || shadow==0.0) continue;
        total+=ringshine_arc(cursor,min(start,limit),r,slat,clat,rho,eta,sun,phi,lit);
        cursor=max(cursor,end);
        if(cursor>=limit) return total;
    }
    return total+ringshine_arc(cursor,limit,r,slat,clat,rho,eta,sun,phi,lit);
}
