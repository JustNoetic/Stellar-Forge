"""Layout/navigation regressions; optional native GPU image checks (--gpu)."""
import argparse
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.rendering.size_comparator import (
    SIZE_SCALE, SizeComparatorRenderer, arrange_bodies, comparator_view,
    pan_view, screen_to_world, zoom_view,
)


class LayoutTests(unittest.TestCase):
    def test_multiple_systems_offsets_deselection_and_empty_scene(self):
        import json
        import tempfile
        from engine.ephemeris.system_manager import SystemManager
        from engine.rendering.comparator_systems import ComparatorSystems
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manager=SystemManager(str(root/'systems'),str(root/'default.json'))
            raw=[{'name':'Star','type':'Star','m':1,'r':1},
                 {'name':'Planet','type':'Terrestrial','parentId':'Star','a':1,'m':3e-6,'r':0.01,
                  'rings':[{'inner':1.2,'outer':2,'opacity':0.5}]}]
            (root/'default.json').write_text(json.dumps(raw))
            for name in ('Second','Third'):
                manager.save_system_data(name,raw)
            app=types.SimpleNamespace(camera={'size_comparator_systems':['Solar System','Second','Third']},
                active_system_name='Solar System',comparison_enabled=False,sys_mgr=manager,
                texture_slices={'planet':1},ring_textures={})
            systems=ComparatorSystems()
            scene=systems.collect(app,{})
            self.assertEqual(len(scene['bodies']),6)
            np.testing.assert_equal(scene['parents'],[-1,0,-1,2,-1,4])
            self.assertEqual([r['body_idx'] for r in scene['rings']],[1,3,5])
            self.assertEqual(scene['sources'][4],('Third',0,'saved'))
            np.testing.assert_equal(scene['tex_indices'],[0,1]*3)
            np.testing.assert_allclose(scene['visuals'][0,3],scene['visuals'][4,3])
            app.camera['size_comparator_systems']=['Third']
            scene=systems.collect(app,{})
            np.testing.assert_equal(scene['parents'],[-1,0])
            self.assertEqual(scene['rings'][0]['body_idx'],1)
            self.assertEqual(set(systems.cache),{'Third'})
            app.camera['size_comparator_systems']=[]
            scene=systems.collect(app,{})
            self.assertEqual(scene['visuals'].shape,(0,9))
            self.assertEqual(scene['sources'],[])

    def test_colliding_labels_are_hidden_without_moving(self):
        app = types.SimpleNamespace(ui_visible=True,show_outliner=False,window_width=900,
            window_height=720,camera={'size_comparator_view':{'center':[0,0],'height':7.2}})
        bodies = [{'name':'Jupiter'},{'name':'Saturn'},{'name':'Uranus'},{'name':'Neptune'}]
        draws=[]
        draw=types.SimpleNamespace(add_text=lambda x,y,color,text:draws.append((x,y,text)))
        positions=np.array([[0,0],[.3,0],[.55,0],[.75,0]])
        radii=np.array([.2,.18,.1,.09])
        with patch('imgui.get_background_draw_list',return_value=draw), \
             patch('imgui.get_color_u32_rgba',return_value=0), \
             patch('imgui.calc_text_size',side_effect=lambda s:(len(s)*7,14)):
            SizeComparatorRenderer.labels(app,bodies,positions,radii,radii,[-1]*4,100)
        names=[d for d in draws if d[2] in ('Jupiter','Saturn','Uranus','Neptune')]
        self.assertEqual(len(names),1)
        self.assertEqual(names[0],(450-len('Jupiter')*7/2,305.0,'Jupiter'))

    def test_true_radii_and_orbital_order(self):
        bodies = [{'name': 'Sun', 'type': 'Star'}, {'name': 'Far', 'a': 2},
                  {'name': 'Near', 'a': 1}]
        pos, radii, _ = arrange_bodies(bodies, [100/SIZE_SCALE, 2/SIZE_SCALE, 1/SIZE_SCALE], [-1, 0, 0], [])
        np.testing.assert_allclose(radii, [100, 2, 1])
        self.assertLess(pos[0, 0], pos[2, 0])
        self.assertLess(pos[2, 0], pos[1, 0])
        np.testing.assert_equal(pos[:, 1], 0)

    def test_moon_columns_do_not_overlap(self):
        bodies = [{'type': 'Star'}, {'a': 1}, {'a': 2}, {'a': 1}, {'a': 0.1}]
        pos, r, _ = arrange_bodies(bodies, np.array([10, 2, 1, 1.5, 0.2])/SIZE_SCALE,
                                  [-1, 0, 1, 1, 3], [])
        self.assertLess(pos[3, 1], pos[1, 1] - r[1] - r[3])
        self.assertLess(pos[4, 1], pos[3, 1] - r[3] - r[4])
        self.assertLess(pos[2, 1], pos[4, 1] - r[4] - r[2])
        np.testing.assert_equal(pos[1:, 0], pos[1, 0])

    def test_star_rows_and_starless_roots(self):
        bodies = [{'type': 'Star'}, {'a': 1}, {'type': 'Star'}, {'a': 1}, {'a': 0}]
        pos, r, bounds = arrange_bodies(bodies, np.array([10, 1, 3, 2, 0.5])/SIZE_SCALE,
                                       [-1, 0, -1, 2, -1], [])
        self.assertLess(pos[2, 1] + r[2], pos[0, 1] - r[0])
        self.assertLess(pos[4, 1], pos[2, 1] - r[2])
        self.assertEqual(len(set(map(tuple, pos))), len(bodies))
        self.assertTrue(np.isfinite(bounds).all())

    def test_ring_spacing_and_bounds(self):
        rings = [{'body_idx': 1, 'outer_r': 10/SIZE_SCALE}]
        pos, r, bounds = arrange_bodies([{'type': 'Star'}, {'a': 1}, {'a': 2}],
                                       np.array([10, 2, 1])/SIZE_SCALE, [-1, 0, 0], rings)
        self.assertAlmostEqual(pos[2, 0] - pos[1, 0], 2 + (10-2)*0.25 + 1 + 0.5)
        self.assertGreaterEqual(bounds[2], pos[1, 0]+10)

    def test_empty_and_cyclic_hierarchies(self):
        pos, radii, _ = arrange_bodies([], [], [], [])
        self.assertEqual(pos.shape, (0, 2))
        self.assertEqual(len(radii), 0)
        pos, _, _ = arrange_bodies([{'a': 0}, {'a': 0}], [1/SIZE_SCALE]*2, [1, 0], [])
        self.assertTrue(np.isfinite(pos).all())
        self.assertFalse(np.array_equal(pos[0], pos[1]))

    def test_zoom_keeps_cursor_world_point(self):
        view = comparator_view({})
        point = screen_to_world(view, 870, 230, 1280, 720)
        zoom_view(view, 3, 870, 230, 1280, 720)
        self.assertLess(view['height'], 20)
        np.testing.assert_allclose(screen_to_world(view, 870, 230, 1280, 720), point)
        zoom_view(view, -3, 870, 230, 1280, 720)
        self.assertAlmostEqual(view['height'], 20)

    def test_pan_follows_drag_at_every_scale(self):
        view = comparator_view({})
        pan_view(view, 120, 60, 600)
        np.testing.assert_allclose(view['center'], [-4, 2])


def gpu_checks(output):
    import imgui
    import moderngl
    from PIL import Image
    from engine.rendering.render_utils import create_icosphere_mesh
    from engine.ephemeris.system_manager import temperature_to_rgb
    from engine.rendering.shader_loader import load_shader

    ctx = moderngl.create_standalone_context(require=460,
        **({'backend': 'egl'} if sys.platform.startswith('linux') else {}))
    print('GPU:', ctx.info['GL_RENDERER'])
    vertices, indices = create_icosphere_mesh(5)
    vbo, ibo = ctx.buffer(vertices.tobytes()), ctx.buffer(indices.tobytes())
    renderer = SizeComparatorRenderer(ctx, vbo, ibo)
    # Match the app's compatibility property on ModernGL versions without it.
    if not hasattr(type(ctx), 'depth_mask'):
        type(ctx).depth_mask = property(lambda c: c.fbo.depth_mask,
                                        lambda c, v: setattr(c.fbo, 'depth_mask', v))
    w = h = 512
    target = ctx.texture((w, h), 4, dtype='f4')
    depth = ctx.depth_renderbuffer((w, h))
    fbo = ctx.framebuffer([target], depth)
    atlas = ctx.texture((16, 1), 4, np.ones((1,16,4), dtype='f4').tobytes(), dtype='f4')
    atlas.profile_revision = 0
    app = types.SimpleNamespace(camera={'size_comparator_view': {'center':[1.25,0], 'height':2.4}},
        window_width=w, window_height=h, fb_width=w, fb_height=h,
        show_outliner=False, show_inspector=False, ui_visible=False,
        pick_request=None, gpu_body_textures={}, _comparator_primary_count=1)
    body = [{'name':'Planet','type':'Terrestrial','a':1}]
    visuals = np.array([[1,1,1,1/SIZE_SCALE,0,0,1,0,0]], dtype='f4')
    renderer.layout_key = tuple((b.get('name'),b.get('type'),-1) for b in body)
    fbo.use()
    renderer.render(app,body,visuals,[-1],[0],[0],[],atlas)
    image = np.frombuffer(fbo.read(components=4,dtype='f4'),dtype='f4').reshape(h,w,4)
    yy,xx = np.indices((h,w))
    disk = (xx+0.5-w/2)**2+(yy+0.5-h/2)**2 < (h/2.4-2)**2
    fraction = np.mean(np.max(image[disk,:3],axis=1)>0.0001)
    assert abs(fraction-0.75)<0.02, fraction
    assert image[330,160,0] > image[180,330,0], 'Light must come from upper left'
    print(f'Illuminated projected disk: {fraction:.3%}')

    body[0] = {'name':'Warm Star','type':'Star','star_props':{'temp':3500}}
    renderer.layout_key = tuple((b.get('name'),b.get('type'),-1) for b in body)
    renderer.render(app,body,visuals,[-1],[0],[0],[],atlas)
    star = np.frombuffer(fbo.read(components=4,dtype='f4'),dtype='f4').reshape(h,w,4)
    expected = np.array(temperature_to_rgb(3500))**2.2
    np.testing.assert_allclose(star[h//2,w//2,:3], expected, atol=0.008)
    assert star[h//2,w//2,0]>star[h//2,w//2,1]>star[h//2,w//2,2]
    assert star[h//2,w//2,0] > star[h//2,60,0], 'Limb darkening'

    composite = ctx.program(vertex_shader=load_shader('post/fullscreen_quad.vert'),
                            fragment_shader=load_shader('post/composite.frag'))
    quad = ctx.buffer(np.array([-1,-1,1,-1,-1,1,1,1],dtype='f4').tobytes())
    # Production composite uses in_position with generated UVs.
    vao = ctx.vertex_array(composite,[(quad,'2f','in_position')])
    composite['u_size_comparator'].value=True
    composite['u_main_texture'].value=0
    composite['u_bloom_texture'].value=0
    composite['u_conv_bloom_texture'].value=0
    result = ctx.simple_framebuffer((w,h))
    result.use()
    ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
    target.use(location=0)
    vao.render(moderngl.TRIANGLE_STRIP)
    srgb = np.frombuffer(result.read(components=3),dtype=np.uint8).reshape(h,w,3)
    np.testing.assert_allclose(srgb[h//2,w//2]/255, temperature_to_rgb(3500), atol=0.01)
    print('Blackbody color, limb darkening, and LDR composite verified')

    if output:
        output.mkdir(parents=True,exist_ok=True)
        Image.fromarray(srgb[::-1]).save(output/'star-color.png')
        Image.fromarray((np.clip(image[::-1,:,:3],0,1)**(1/2.2)*255).astype('uint8')).save(output/'planet-phase.png')
    # Exercise textures, clouds, ring atlas, picking and repeated zoom renders.
    app.ui_visible=True
    imgui.create_context()
    io=imgui.get_io(); io.display_size=(w,h); io.fonts.get_tex_data_as_rgba32()
    imgui.new_frame()
    body[0]={'name':'Ringed Planet','type':'Gas Giant'}
    renderer.layout_key=None
    app.camera['size_comparator_view']['fit']=True
    tex=ctx.texture((8,4),4,np.tile(np.array([100,160,230,255],dtype='uint8'),32).tobytes())
    app.gpu_body_textures={1:{'diffuse':tex,'clouds':tex}}
    rings=[{'body_idx':0,'inner_r':1.3/SIZE_SCALE,'outer_r':2/SIZE_SCALE,
            'opacity':0.4,'shadow_grad':np.ones((32,4),dtype='f4'),'row_idx':0}]
    fbo.use()
    renderer.render(app,body,visuals,[-1],[1],[0.4],rings,atlas)
    center=app.camera['size_comparator_view']['center']
    pos,_,_=arrange_bodies(body,visuals[:,3],[-1],rings)
    scale=h/app.camera['size_comparator_view']['height']
    app.pick_request=((pos[0,0]-center[0])*scale+w/2, -(pos[0,1]-center[1])*scale+h/2)
    renderer.render(app,body,visuals,[-1],[1],[0.4],rings,atlas)
    assert app.camera['inspected_idx']==0
    app.camera['size_comparator_focus']=0
    renderer.render(app,body,visuals,[-1],[1],[0.4],rings,atlas)
    assert ctx.error=='GL_NO_ERROR',ctx.error
    imgui.end_frame()
    print('Textured body, clouds, labels, rings, picking, and focus verified')
    # Minified alternating opaque/gap bands must preserve mean coverage and
    # linear radiance. The second array layer must remain entirely independent.
    striped=np.ones((4096,4),dtype='f4'); striped[::2,3]=0
    green=np.ones((4096,4),dtype='f4'); green[:,:3]=[0,1,0]
    filtered_rings=[dict(rings[0],shadow_grad=striped),dict(rings[0],shadow_grad=green)]
    fbo.use()
    renderer.render(app,body,visuals,[-1],[1],[0.4],filtered_rings,atlas)
    mip=ctx.program(vertex_shader=load_shader('post/fullscreen_quad.vert'),
        fragment_shader='''#version 460 core
        uniform sampler2DArray rings; uniform float layer; out vec4 color;
        void main(){color=textureLod(rings,vec3(0.5,0.5,layer),12.0);}''')
    mip_vao=ctx.vertex_array(mip,[(quad,'2f','in_position')])
    mip['rings'].value=0
    ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
    renderer.profile_texture.use(location=0)
    alpha=1-math.exp(-0.4/0.2)
    for layer,expected in ((0,[alpha/2]*4),(1,[0,alpha,0,alpha])):
        mip['layer'].value=layer
        mip_vao.render(moderngl.TRIANGLE_STRIP)
        pixel=np.frombuffer(fbo.read(components=4,dtype='f4'),dtype='f4').reshape(h,w,4)[h//2,w//2]
        np.testing.assert_allclose(pixel,expected,atol=1e-5)
    mip_vao.release();mip.release()
    print('Ring mipmaps preserve band coverage and color with no mixing between layers')
    # A true-scale Earth atmosphere must scatter outside the solid limb and
    # attenuate/tint the disk. Turning it off must restore the original image.
    app.ui_visible=False
    app.camera.update(atmo_enabled=False,atmo_quality=2)
    earth_radius=6378.0
    earth_world=earth_radius*1e-5
    body=[{'name':'Earth','type':'Terrestrial'}]
    visuals=np.array([[0.2,0.3,0.4,earth_world/SIZE_SCALE,0,0,1,0,0]],dtype='f4')
    app.camera['size_comparator_view']={'center':[0.25+earth_world,0],'height':earth_world*2.4}
    renderer.layout_key=(('Earth','Terrestrial',-1),)
    atmosphere=[{'body_idx':0,'planet_radius_km':earth_radius,'height':105.0,
        'atmo_radius_km':earth_radius+105,'surface_radius_au':earth_world/SIZE_SCALE,
        'surface_pressure':1.0,'temperature':288.15,'composition':{'N2':0.78,'O2':0.21,'Ar':0.01,'O3':1e-9},
        'beta_mie':2e-6,'h_mie':1.2,'mie_g':0.76,'intensity':1.0}]
    def earth_image():
        fbo.use()
        renderer.render(app,body,visuals,[-1],[0],[0],[],atlas,
                        atmospheres=atmosphere,masses=[3.003e-6])
        return np.frombuffer(fbo.read(components=4,dtype='f4'),dtype='f4').reshape(h,w,4).copy()
    off=earth_image()
    app.camera['atmo_enabled']=True
    on=earth_image()
    assert np.isfinite(on).all()
    radial=np.sqrt((xx+0.5-w/2)**2+(yy+0.5-h/2)**2)
    ring=(radial>h/2.4+0.5)&(radial<h/2.4+3)&(xx<w/2)
    assert np.max(on[ring,2]-off[ring,2])>0.01, 'Atmosphere must extend beyond the solid limb'
    assert np.mean(abs(on[disk,:3]-off[disk,:3]))>0.001, 'Scattering/extinction must affect the disk'
    # Rotation/oblateness, pan and zoom only alter ray geometry, not the LUTs.
    bake_count=renderer.atmospheres.bake_count
    visuals[0,8]=0.15
    app.camera['size_comparator_view']['center'][0]+=earth_world*0.1
    oblate=earth_image()
    assert np.isfinite(oblate).all()
    assert renderer.atmospheres.bake_count==bake_count
    # An opaque foreground plane must clip the atmospheric ray completely.
    baseline_center=app.camera['size_comparator_view']['center'][:]
    fbo.use(); fbo.clear(0.1,0.2,0.3,1,depth=0.0)
    pos,radius,_=arrange_bodies(body,visuals[:,3],[-1],[])
    projection=np.diag([2/app.camera['size_comparator_view']['height']]*2+[-1,1]).astype('f4')
    renderer.atmospheres.render(app,body,visuals,pos,radius,app.camera['size_comparator_view'],
        projection,np.array([-0.48989795,0.84852814,0.2]),1.0,atmosphere,[3.003e-6])
    occluded=np.frombuffer(fbo.read(components=4,dtype='f4'),dtype='f4').reshape(h,w,4)
    np.testing.assert_allclose(occluded[:,:,:3],np.broadcast_to([0.1,0.2,0.3],(h,w,3)),atol=1e-5)
    visuals[0,8]=0.0
    app.camera['size_comparator_view']['center']=[0.25+earth_world,0]
    app.camera['atmo_quality']=0
    np.testing.assert_allclose(earth_image(),off,atol=1e-7)
    app.camera['atmo_quality']=2
    atmosphere[0]['composition']={'CO2':1.0}
    changed=earth_image()
    assert renderer.atmospheres.bake_count>bake_count
    assert not np.allclose(changed,on), 'Composition edits must update scattering'
    app.camera['atmo_enabled']=False
    np.testing.assert_allclose(earth_image(),off,atol=1e-7)
    # A nearly transparent atmosphere must preserve rings exactly. This catches
    # gaps or double blending where the foreground/background passes meet.
    atmosphere[0]['surface_pressure']=1e-12
    atmosphere[0]['beta_mie']=0.0
    app.camera['size_comparator_view']['height']=earth_world*4.8
    clear_ring=[{'body_idx':0,'inner_r':earth_world*1.3/SIZE_SCALE,
        'outer_r':earth_world*2/SIZE_SCALE,'opacity':0.4,
        'shadow_grad':np.ones((32,4),dtype='f4'),'row_idx':0}]
    def ring_image():
        fbo.use()
        renderer.render(app,body,visuals,[-1],[0],[0],clear_ring,atlas,
                        atmospheres=atmosphere,masses=[3.003e-6])
        return np.frombuffer(fbo.read(components=4,dtype='f4'),dtype='f4').reshape(h,w,4).copy()
    ring_off=ring_image()
    app.camera['atmo_enabled']=True
    np.testing.assert_allclose(ring_image(),ring_off,atol=5e-6)
    assert ctx.error=='GL_NO_ERROR',ctx.error
    if output:
        Image.fromarray((np.clip(on[::-1,:,:3],0,1)**(1/2.2)*255).astype('uint8')).save(output/'earth-atmosphere.png')
    print('Atmosphere limb, disk scattering/extinction, oblate rays, cache invalidation, depth occlusion, ring splitting, and toggles verified')
    renderer.release()
    for obj in (vbo,ibo,tex,atlas,fbo,target,depth,vao,quad,composite,result): obj.release()
    ctx.release()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(LayoutTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): sys.exit(1)
    if args.gpu: gpu_checks(args.output)
