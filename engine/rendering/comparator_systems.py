"""Collect any checked systems without switching or advancing simulations."""
import copy
import math
from pathlib import Path

import numpy as np

from engine.core.constants import AU_TO_KM
from engine.physics.physics_core import load_system_from_data
from engine.rendering.planetshine import _build_tex_idx_arr
from engine.rendering.render_utils import generate_ring_shadow_grad, hex_to_rgb


def selected_systems(app):
    names = app.camera.get('size_comparator_systems')
    if names is None:
        names = [app.active_system_name]
        if app.comparison_enabled and app.comparison_system_name not in names:
            names.append(app.comparison_system_name)
        app.camera['size_comparator_systems'] = names
    return names


def prepare_rings(bodies, ring_bodies, textures):
    """Use the same textured spans, procedural segments and HSBA as the scene."""
    result = []
    for bi, segments, pole, radius in ring_bodies:
        body = bodies[bi]
        image = textures.get(body['name'].lower())
        spans = [(seg, None, seg['inner'], seg['outer']) for seg in segments]
        if image is not None and segments:
            inner = float(body.get('ring_texture_inner', min(s['inner'] for s in segments)))
            outer = float(body.get('ring_texture_outer', max(s['outer'] for s in segments)))
            textured = [s for s in segments if s['inner'] >= inner-1e-4 and s['outer'] <= outer+1e-4]
            samples = np.frombuffer(image.tobytes(), dtype=np.uint8).astype('f4').reshape(-1,4)/255
            spans = [(textured[0] if textured else segments[0], samples, inner, outer)]
            spans += [(s,None,s['inner'],s['outer']) for s in segments
                      if s['outer'] <= inner+1e-4 or s['inner'] >= outer-1e-4]
        for seg, samples, inner, outer in spans:
            color = hex_to_rgb(seg.get('color','#ffffff'))
            profile = generate_ring_shadow_grad(
                sorted(seg.get('gradient',[]) if samples is None else [], key=lambda g:g['p']),
                tex_sampled=samples, raw_color=color,
                hue_shift=seg.get('hue_shift',0), saturation=seg.get('saturation',1),
                brightness=seg.get('brightness',1), alpha_boost=seg.get('alpha_boost',1))
            result.append(dict(body_idx=bi, inner_r=inner*radius, outer_r=outer*radius,
                               opacity=seg.get('opacity',1), shadow_grad=profile))
    return result


class ComparatorSystems:
    def __init__(self):
        self.cache = {}
        self.errors = {}

    def load(self, app, name):
        snapshot = app.sys_mgr.get_snapshot(name)
        path = Path(app.sys_mgr._system_json_path(name))
        if not path.exists() and name == app.sys_mgr.SOLAR_SYSTEM_NAME:
            path = Path(app.sys_mgr.default_json)
        stamp = (id(snapshot), path.stat().st_mtime_ns if path.exists() else None)
        if name in self.cache and self.cache[name][0] == stamp:
            return self.cache[name][1]
        raw = snapshot.bodies_data if snapshot is not None else app.sys_mgr.load_system_data(name)
        bundle = load_system_from_data(copy.deepcopy(raw))
        bodies = bundle['bodies_data']
        visual = np.asarray(bundle['visual_data'], dtype='f4').reshape(-1,14)[:,:9].copy()
        for bi, body in enumerate(bodies):
            if body.get('type') != 'Star':
                f = min(.95,max(0,float(body.get('oblateness',0))))
                visual[bi,3] = float(body['req_km'])/AU_TO_KM if body.get('req_km') else visual[bi,3]/(1-f)**(1/3)
        data = dict(bodies=bodies, visuals=visual, parents=bundle['parent_indices'],
                    masses=np.array([b.get('m',0) for b in bodies],dtype='f8'),
                    tex_indices=_build_tex_idx_arr(bodies, app.texture_slices),
                    spins=np.array([math.radians(float(b.get('W0',0))) for b in bodies]),
                    rings=prepare_rings(bodies,bundle['ring_bodies'],app.ring_textures),
                    atmospheres=bundle['atmo_bodies'])
        self.cache[name] = stamp,data
        return data

    def collect(self, app, live):
        names = selected_systems(app)
        available = set(app.sys_mgr.list_systems()) | set(live)
        names[:] = list(dict.fromkeys(n for n in names if n in available))
        self.cache = {n:v for n,v in self.cache.items() if n in names and n not in live}
        self.errors = {}
        bodies, sources, rings, atmospheres = [],[],[],[]
        arrays = {n:[] for n in ('visuals','parents','tex_indices','spins','masses')}
        for name in names:
            try:
                data = live[name][1] if name in live else self.load(app,name)
            except (OSError,ValueError,KeyError,TypeError) as error:
                self.errors[name] = str(error)
                continue
            role = live[name][0] if name in live else 'saved'
            offset = len(bodies)
            bodies.extend(data['bodies'])
            sources.extend((name,i,role) for i in range(len(data['bodies'])))
            for field in arrays:
                value = np.asarray(data[field])
                if field == 'parents':
                    value = np.where(value >= 0,value+offset,-1)
                arrays[field].append(value)
            rings.extend(dict(r,body_idx=r['body_idx']+offset) for r in data['rings'])
            atmospheres.extend(dict(a,body_idx=a['body_idx']+offset) for a in data['atmospheres'])
        result = {n:np.concatenate(a) if a else np.empty((0,9) if n=='visuals' else (0,))
                  for n,a in arrays.items()}
        result.update(bodies=bodies, sources=sources, rings=rings, atmospheres=atmospheres)
        app._comparator_sources = sources
        app._comparator_bodies = bodies
        return result
