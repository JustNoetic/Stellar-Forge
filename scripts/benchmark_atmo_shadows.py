"""Fullscreen atmosphere-shadow GPU benchmark on an Earth-sized atmosphere.

Run: venv/Scripts/python.exe scripts/benchmark_atmo_shadows.py --width 1920 --height 1080
The production fragment shader renders close to opaque host-ring shadows and
external-ring shadows. GL timer queries exclude LUT baking and CPU readback.
These are atmosphere-pass timings, not whole-application FPS predictions.
"""
import argparse
import os
import statistics
import sys

import moderngl
import numpy as np

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.test_scattering_segment import PARAMETERS, PSI, create_production_probe
from engine.rendering.scattering_lut import ScatteringLUTCache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width',type=int,default=1024)
    parser.add_argument('--height',type=int,default=1024)
    parser.add_argument('--cells',type=int,default=24)
    parser.add_argument('--samples',type=int,default=7)
    args=parser.parse_args()
    if min(args.width,args.height,args.samples)<=0 or not 4<=args.cells<=64:
        parser.error('Positive dimensions/samples and 4-64 cells are required')
    ctx=moderngl.create_standalone_context(require=460)
    multiple=ctx.texture((64,64),4,np.tile(np.r_[PSI,1].astype('f4'),(4096,1)).tobytes(),dtype='f4')
    cache=ScatteringLUTCache(ctx)
    tables=cache.get(PARAMETERS,multiple)
    probe=create_production_probe(ctx,tables,multiple,size=(args.width,args.height))
    p,au=probe['program'],probe['au']
    instance=np.zeros(28,'f4'); instance.view('u4')[15]=1
    probe['instance'].write(instance.tobytes())
    p['u_num_ring_planes'].value=1
    p['u_ring_station_count'].value=args.cells
    p['u_ring_station_cells'].write(np.pad(np.linspace(0,1,9,dtype='f4'),(0,27)).tobytes())
    centers=np.zeros((16,3),'f4')
    normals=np.zeros((16,3),'f4'); normals[0]=[0,1,0]
    params=np.zeros((16,4),'f4'); params[0]=[0,10000/au,1,0]
    masks=np.zeros(16,'u4'); masks[0]=1
    for name,buf in [('u_ring_normal',normals),('u_ring_params',params),('u_ring_coplanar_mask',masks)]:
        p[name].write(buf.tobytes())
    print(ctx.info['GL_RENDERER'])
    print(f'{args.width}x{args.height}, {args.cells} visibility/ray cells, median GPU ms')
    names=['Station slicing','Stochastic subtraction','Blackrack subtraction','Endpoint subtraction']
    for external in (False,True):
        centers[0]=[0,6460/au,0] if external else [0,0,0]
        p['u_ring_center'].write(centers.tobytes())
        camera=np.array((0,6372,0) if external else (np.sqrt(6372**2-200**2),200,0),float)
        ground=camera*6371/6372
        sun=(0,1,0) if external else (.98,-.2,0)
        print('External rings' if external else 'Host rings')
        for method,name in enumerate(names):
            p['u_atmo_shadow_method'].value=method
            if method==0: p['u_ring_station_count'].value=8
            else: p['u_ring_station_count'].value=args.cells
            probe['draw'](camera,ground,sun=sun,readback=False)
            # Warm driver specialization and clocks before collecting samples.
            for _ in range(8): probe['vao'].render(moderngl.TRIANGLES,vertices=3)
            ctx.finish()
            times=[]
            for _ in range(args.samples):
                with ctx.query(time=True) as query:
                    probe['vao'].render(moderngl.TRIANGLES,vertices=3)
                times.append(query.elapsed/1e6)
            print(f'  {name}: {statistics.median(times):.3f}',flush=True)
            assert ctx.error=='GL_NO_ERROR'
    cache.release(); ctx.release()


if __name__=='__main__':
    main()
