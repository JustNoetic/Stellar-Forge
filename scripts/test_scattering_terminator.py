"""GPU terminator sweeps against 1024-cell direct-scattering integration.

Includes Earth from 200 km at exposure 12500. Uses the cached optical-depth
field in both paths to isolate angular radiance interpolation; independent
sun-path and medium validation lives in test_scattering_segment.py.
"""
from pathlib import Path
import sys

import moderngl
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.rendering.shader_loader import load_shader
from engine.rendering.scattering_lut import ScatteringLUTCache
from test_scattering_segment import PARAMETERS


def main():
    ctx=moderngl.create_standalone_context(require=460)
    multiple=ctx.texture((64,64),4,np.zeros((64,64,4),'f4').tobytes(),dtype='f4')
    multiple.filter=(moderngl.LINEAR,moderngl.LINEAR)
    vertex='#version 460\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0,1);}'
    decl='\n'.join('uniform vec3 '+key+';' if isinstance(value,tuple) else 'uniform float '+key+';'
                   for key,value in PARAMETERS.items() if key not in ('u_scattering_bottom_km','u_scattering_sun_radius'))
    fragment='#version 460\n#define PI 3.14159265358979323846\n'+decl+'''
    uniform sampler2D u_multi_scatter_lut;
    uniform bool u_reference;
    uniform float u_altitude, u_azimuth;
    layout(location=0) out vec4 out_L;
    ''' + load_shader('common/sun_terminator.glsl')+load_shader('common/scattering_segment.glsl')+'''
    void main(){
        float r=u_planet_radius_km+u_altitude;
        float mu=mix(-0.35,0.03,gl_FragCoord.y/96.0);
        float mus=mix(-0.45,0.15,gl_FragCoord.x/512.0);
        vec3 a=vec3(0,r,0), v=vec3(sqrt(1-mu*mu),mu,0);
        vec3 sun=vec3(sqrt(1-mus*mus)*cos(u_azimuth),mus,sqrt(1-mus*mus)*sin(u_azimuth));
        if(r>u_atmo_radius_km){
            float disc=u_atmo_radius_km*u_atmo_radius_km-r*r+r*r*mu*mu;
            if(mu>=0 || disc<0){out_L=vec4(0,0,0,1);return;}
            a+=(-r*mu-sqrt(disc))*v;
            r=length(a);mu=dot(a,v)/r;
        }
        bool ground=mu<scattering_horizon(r);
        float distance=scattering_boundary_distance(r,mu,ground);
        vec3 b=a+distance*v,T=endpoint_transmittance(a,b);
        vec3 L;
        if(!u_reference){L=endpoint_radiance(a,b,sun,T);}
        else{
            L=vec3(0); vec3 running=vec3(1);
            float closest=clamp(-r*mu,0.0,distance);
            float split=closest/max(distance,1e-6);
            float nu=dot(v,sun),g=u_mie_g;
            float pr=3.0/(16.0*PI)*(1.0+nu*nu);
            float pm=3.0/(8.0*PI)*(1.0-g*g)/(2.0+g*g)*(1.0+nu*nu)/pow(1.0+g*g-2.0*g*nu,1.5);
            const int N=1024;
            for(int i=0;i<N;i++){
                float t0=float(i)/N,t1=float(i+1)/N;
                float s0=t0<split?closest*(1-pow(1-t0/max(split,1e-6),2.0)):closest+(distance-closest)*pow((t0-split)/max(1-split,1e-6),2.0);
                float s1=t1<split?closest*(1-pow(1-t1/max(split,1e-6),2.0)):closest+(distance-closest)*pow((t1-split)/max(1-split,1e-6),2.0);
                float ds=s1-s0;
                vec3 p=a+0.5*(s0+s1)*v,R,M,ext;
                endpoint_medium(p,R,M,ext);
                float rq=length(p),sinp=u_planet_radius_km/max(rq,u_planet_radius_km+0.01);
                vec2 term=sun_terminator(dot(p,sun)/rq,sinp,sqrt(max(0.0,1-sinp*sinp)),u_scattering_sun_radius,sqrt(1-u_scattering_sun_radius*u_scattering_sun_radius));
                float ms=max(term.y,scattering_horizon(rq)+1e-5);
                vec3 radial=p/rq,tangent=sun-dot(sun,radial)*radial;
                tangent=length(tangent)>1e-6?normalize(tangent):normalize(cross(radial,vec3(0,0,1)));
                vec3 effective=radial*ms+tangent*sqrt(max(0.0,1-ms*ms));
                vec3 incoming=exp(-endpoint_tau(p,effective,false))*term.x;
                vec3 ct=exp(-ext*ds);
                vec3 factor=mix((1-ct)/max(ext,vec3(1e-20)),vec3(ds),lessThan(ext*ds,vec3(1e-4)));
                L+=running*factor*(R*pr+M*pm)*incoming;
                running*=ct;
            }
        }
        out_L=vec4(L,float(endpoint_solar_umbra(a,sun,u_scattering_sun_radius)
            && endpoint_solar_umbra(b,sun,u_scattering_sun_radius)));
    }
    '''
    program=ctx.program(vertex_shader=vertex,fragment_shader=fragment)
    for name,value in PARAMETERS.items():
        if name in program: program[name].value=value
    ScatteringLUTCache.configure(program); program['u_multi_scatter_lut'].value=3
    target=ctx.texture((512,96),4,dtype='f4'); fbo=ctx.framebuffer([target]); vao=ctx.vertex_array(program,[])
    params=dict(PARAMETERS)
    cache=ScatteringLUTCache(ctx)
    tables=cache.get(params,multiple); cache.bind(tables)
    program['u_scattering_bottom_km'].value=params['u_scattering_bottom_km']
    program['u_scattering_azimuth_count'].value=cache.size[3]
    fbo.use();ctx.viewport=(0,0,512,96)
    for altitude,azimuth in ((100.,0.),(100.,.5),(100.,1.5),(20.,.5),(200.,0.),(200.,.5),(200.,1.5)):
        program['u_altitude'].value=altitude;program['u_azimuth'].value=azimuth
        arrays=[]
        for ref in (False,True):
            program['u_reference'].value=ref;vao.render(moderngl.TRIANGLES,vertices=3)
            arrays.append(np.frombuffer(target.read(),'f4').reshape(96,512,4).copy())
        actual,reference=arrays
        umbra=actual[...,3]>.5
        actual,reference=actual[...,:3],reference[...,:3]
        assert np.all(actual[umbra]==0), 'Direct light leaked into the full-disc umbra'
        peak=reference.max(axis=1).max(axis=1)
        valid=peak>1e-8
        error=np.linalg.norm(actual[valid]-reference[valid],axis=2)/np.maximum(np.linalg.norm(reference[valid],axis=2),peak[valid,None]*.01)
        print('size',cache.size,'alt',altitude,'az',azimuth,'relative error median/p95/max',np.quantile(error,[.5,.95,1]))
        assert np.quantile(error,.95)<.07, (altitude,azimuth,'transport error')
        x=-.45+(np.arange(512)+.5)/512*.6
        # Sunward edge of the shadow: 1% of each reference row's peak blue radiance.
        shifts=[]
        for y in range(96):
            threshold=reference[y,:,2].max()*.01
            if threshold<1e-8: continue
            ai=np.flatnonzero(actual[y,:,2]>threshold); ri=np.flatnonzero(reference[y,:,2]>threshold)
            if len(ai) and len(ri):shifts.append(np.rad2deg(np.arcsin(x[ai[0]])-np.arcsin(x[ri[0]])))
        print('terminator shifts(deg) p05/median/p95',np.quantile(shifts,[.05,.5,.95]))
        assert np.quantile(np.abs(shifts),.95)<.15, (altitude,azimuth,'terminator shift')
        # Exposure magnifies low-amplitude interpolation spill that a relative
        # error against the bright peak misses. Check the user's 200 km case.
        if altitude==200:
            def display(L):
                x=np.maximum(0,L)*12500
                return np.clip(x*(2.51*x+.03)/(x*(2.43*x+.59)+.14),0,1)**(1/2.2)
            delta=np.max(np.abs(display(actual)-display(reference)),axis=2)
            error99=np.quantile(delta,.99)
            print('Exposure 12500, p99 display-channel error:',error99)
            assert error99<.10, (azimuth,'high-exposure terminator leakage',error99)
    cache.release(); multiple.release(); ctx.release()
    print('Analytical terminator, full umbra, and high-exposure regressions passed')


if __name__=='__main__':
    main()
