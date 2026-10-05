"""CPU counterpart of common/ring_optics.glsl; alpha is always vertical.

Phase mixtures retain each material's parameters and share total slab extinction.
The multiple-scattering and opposition terms remain authored approximations.
"""
import math
import numpy as np
from numba import njit

RING_MAX_G = .99
RING_MAX_ALPHA = 1.0 - 2.0**-23  # float32 GLSL 0.9999999
DISK_R2 = (.0694318442,.3300094782,.6699905218,.9305681558)
DISK_W = (.1739274226,.3260725774,.3260725774,.1739274226)

@njit(cache=True)
def ring_tau(alpha):
    return -math.log1p(-min(RING_MAX_ALPHA,max(0.0,alpha)))

@njit(cache=True)
def ring_hg(g, mu):
    g = min(RING_MAX_G,max(-RING_MAX_G,g))
    mu = min(1.0,max(-1.0,mu))
    ag = abs(g)
    signed_mu = mu if g >= 0.0 else -mu
    d = (1.0-ag)**2 + 2.0*ag*(1.0-signed_mu)
    return (1.0-g*g)/(4.0*math.pi*d*math.sqrt(d))

@njit(cache=True)
def ring_phase(mu, alpha, gf, gb, balance, textured):
    mu = min(1.0,max(-1.0,mu))
    pf, pb = ring_hg(gf,mu),ring_hg(gb,mu)
    if textured:
        multiple = min(1.0,max(0.0,(alpha-.1)/.5))
        forward = .95-.45*multiple
    else:
        balance = min(1.0,max(0.0,balance))
        forward = .1+.8*balance
        multiple = min(1.0,max(.1,1.0-.7*balance))
        gf,gb = min(RING_MAX_G,max(-RING_MAX_G,gf)),min(RING_MAX_G,max(-RING_MAX_G,gb))
        pf *= 1.5*(1.0+mu*mu)/(2.0+gf*gf)
        pb *= 1.5*(1.0+mu*mu)/(2.0+gb*gb)
    return forward*pf+(1.0-forward)*pb,multiple

@njit(cache=True)
def ring_transfer(tau, mu_v, mu_0, lit):
    if tau <= 0.0 or mu_v <= 0.0 or mu_0 <= 0.0:
        return 0.0,0.0
    mu_v,mu_0 = max(mu_v,1e-7),max(mu_0,1e-7)
    vd,ld = tau/mu_v,tau/mu_0
    if not lit:
        delta = abs(ld-vd)
        ratio = -math.expm1(-delta)/delta if delta > 0.0 else 1.0
        return vd*math.exp(-min(vd,ld))*ratio,0.0
    single = mu_0/(mu_v+mu_0)*(-math.expm1(-vd-ld))
    gamma = math.sqrt(.08)
    hv = (1+2*mu_v)/(1+2*mu_v*gamma)
    h0 = (1+2*mu_0)/(1+2*mu_0*gamma)
    return single,.92*single*max(0.0,hv*h0-1.0)/(4*math.pi)

@njit(cache=True)
def ring_slab_response(mu_v, sun_elevation, alpha, on_lit_side):
    return ring_transfer(ring_tau(alpha),mu_v,abs(sun_elevation),on_lit_side)[0]

@njit(cache=True)
def ring_component_radiance(tau, alpha, mu_v, sun, mu, gf, gb, balance,
                            textured, unlit, lit):
    single,multiple = ring_transfer(tau,mu_v,abs(sun),lit)
    pf,ms_weight = ring_phase(mu,alpha,gf,gb,balance,textured)
    if lit:
        angle = math.acos(min(1.0,max(-1.0,-mu)))
        density = min(1.0,max(0.0,tau/1.5))
        single *= (1+.8*density/(1+angle/.07))*(1+.3*density*math.exp(-angle/.006))
    else:
        single *= max(0.0,unlit)
    return single*pf+multiple*ms_weight

@njit(cache=True)
def ring_phase_radiance(mu_v, sun_elevation, alpha, cos_theta, asymmetry,
                        backscatter, scatter, textured, unlit_factor, on_lit_side):
    return ring_component_radiance(ring_tau(alpha),alpha,mu_v,sun_elevation,cos_theta,
                                  asymmetry,backscatter,scatter,textured,unlit_factor,on_lit_side)

@njit(cache=True)
def ring_mixed_radiance(samples, signed_view, sun, mu, sin_radius):
    """Area quadrature of mixed slabs for the distant-annulus approximation.

    Each [radius,layer] row: shared tau, own alpha, gf, gb, balance, family,
    opposite multiplier, RGB, unused, radial-area weight times tau fraction.
    """
    result=np.zeros(3,dtype=np.float64)
    cosine=math.sqrt(max(0.0,1.0-sun*sun))
    vx=(-mu-sun*signed_view)/cosine if cosine>1e-7 else math.sqrt(max(0.,1.-signed_view*signed_view))
    vz=math.sqrt(max(0.0,1.0-signed_view*signed_view-vx*vx))
    resolve=sin_radius>0.0 and (abs(sun)<2*sin_radius or sin_radius>.001)
    count=64 if resolve else 1
    for q in range(count):
        sample_sun,sample_mu,source_weight=sun,mu,1.0
        if resolve:
            row=q//16
            az=(q%16+.5)*(2*math.pi/16)
            r=min(sin_radius,.999)*math.sqrt(DISK_R2[row])
            cd=math.sqrt(1-r*r)
            sx=cosine*cd-sun*r*math.sin(az)
            sample_sun=sun*cd+cosine*r*math.sin(az)
            sz=r*math.cos(az)
            sample_mu=-(sx*vx+sample_sun*signed_view+sz*vz)
            source_weight=DISK_W[row]/(16*cd)
        lit=signed_view*sample_sun>=0.0
        for n in range(samples.shape[0]):
            for m in range(samples.shape[1]):
                material=samples[n,m]
                if material[11]<=0.0:
                    continue
                response=ring_component_radiance(material[0],material[1],abs(signed_view),
                    sample_sun,sample_mu,material[2],material[3],material[4],material[5]>.5,
                    material[6],lit)*material[11]*source_weight
                for channel in range(3):
                    result[channel]+=material[7+channel]*response
    return result
