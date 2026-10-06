"""Native ImGui layout and interaction regressions (no simulation or window).

Run with the project's Python environment. Optional --render DIR writes real
ImGui/OpenGL previews to DIR; its empty viewport is deliberately not a simulation.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import imgui
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.core.input_handler import InputHandlerMixin
from engine.ui import render_ui
from engine.ui.workspace import apply_theme, workspace, SETTINGS_SECTIONS


class Fixture(SimpleNamespace):
    def save_settings(self):
        self.saves += 1

    def _get_active_refraction_and_lens_params(self, *args, **kwargs):
        return None, None


def fixture():
    bodies = [
        dict(name='Sun', type='Star', m=1., r=1., star_props=dict(temp=5778, lum=1., radius=1., **{'class': 'G2V', 'stage': 'Main sequence'})),
        dict(name='Earth', type='Terrestrial', m=3.003e-6, r=6371/696340, a=1., rotation_period=23.93),
        dict(name='Moon', type='Moon', m=3.694e-8, r=1737/696340, a=.00257, rotation_period=655.7),
        dict(name='Mars', type='Terrestrial', m=3.227e-7, r=3389/696340, a=1.52, rotation_period=24.62),
    ]
    visuals = np.zeros((4, 9), dtype='f4')
    visuals[:, :3] = [.3, .5, .8]
    visuals[:, 6] = 1.
    visuals[:, 4] = [b['r'] * 696340 / 149597870.7 for b in bodies]
    positions = np.array([[0, 0, 0], [1, 0, 0], [1.00257, 0, 0], [1.52, 0, 0]], dtype='f8')
    velocities = np.array([[0, 0, 0], [0, 0, -6.28], [0, 0, -6.5], [0, 0, -5.1]], dtype='f8')
    camera = dict(inspected_idx=1, inspected_is_cmp=False, inspect_bary=False,
                  tracking_idx=1, tracking_is_cmp=False, tracking_bary=False,
                  edit_mode=False, movement_mode=0, flight_speed=.1,
                  fov=45., target=np.zeros(3), cam_pos_rel=np.zeros(3),
                  cam_pos_rel_prev=np.zeros(3), atmo_quality=2)
    app = Fixture(camera=camera, ui_visible=True, show_outliner=True,
                  show_inspector=True, show_time_hud=True, fb_width=1280,
                  fb_height=720, active_system_name='Solar System', saves=0,
                  shared_state=dict(lock=threading.Lock(), crud_queue=[]),
                  time_ctrl=dict(paused=False, multiplier=1., time_direction=1,
                                 timeline_playing=False, timeline_speed=1.,
                                 timeline_scrub_float=0.), scrub_index=[0],
                  comparison_enabled=False, comparison_system_name='Second',
                  comparison_offset_au=10., star_catalog=None,
                  sys_mgr=SimpleNamespace(list_systems=lambda: ['Solar System', 'Second']),
                  sys_mgr_spice=SimpleNamespace(is_downloading=False, download_error=None),
                  bodies_data=bodies, visual_arr=visuals, texture_mean_colors={},
                  num_bodies_cmp=4, bodies_data_cmp=bodies, visual_data_cmp=visuals,
                  visual_arr_cmp=visuals, parent_snap_cmp=np.array([-1, 0, 1, 0]),
                  mass_snap_cmp=np.array([b['m'] for b in bodies]),
                  subsys_mass_buf_cmp=np.array([b['m'] for b in bodies]),
                  subsys_pos_buf_cmp=positions, subsys_vel_buf_cmp=velocities,
                  pos_snap_cmp=positions, vel_snap_cmp=velocities,
                  tree_indices_snap_cmp=np.arange(4), tree_depths_snap_cmp=np.array([0, 1, 2, 1]),
                  atmo_bodies_cmp=[], _comparator_sources=[], _comparator_bodies=[],
                  comparator_systems=SimpleNamespace(errors={}))
    app.texture_slices = {}
    state = dict(bodies=bodies, visuals=visuals, positions=positions, velocities=velocities,
                 parents=np.array([-1, 0, 1, 0]), mass=np.array([b['m'] for b in bodies]),
                 depths=np.array([0, 1, 2, 1]))
    return app, state


class NativeUI:
    def __init__(self, width=1280, height=720, render=False):
        self.context = imgui.create_context()
        imgui.set_current_context(self.context)
        self.io = imgui.get_io()
        self.io.ini_file_name = None
        self.io.display_size = (width, height)
        self.io.delta_time = 1/60
        self.app, self.state = fixture()
        self.items = {}
        self.windows = {}
        self.scroll_children = {}
        self.frame_overrides = {}
        self.selected_tab = None
        self.triggers = {'switch_system': lambda name: setattr(self.app, 'active_system_name', name)}
        for key in ('ephem_switch', 'ephem_exit', 'cmp_switch', 'load_system', 'ephem_export'):
            self.triggers[key] = lambda *args, **kwargs: None
        self.ctx = None
        self.renderer = None
        if render:
            import moderngl
            from engine.rendering.imgui_renderer import ModernGLImGuiRenderer
            self.ctx = moderngl.create_standalone_context(require=430)
            self.fbo = self.ctx.simple_framebuffer((width, height), components=4)
            self.renderer = ModernGLImGuiRenderer(self.ctx)
        else:
            self.io.fonts.get_tex_data_as_rgba32()

    def frame(self, **overrides):
        self.items = {}
        self.windows = {}
        imgui.new_frame()
        originals = {name: getattr(imgui, name) for name in
                     ('button', 'small_button', 'checkbox', 'selectable', 'begin_tab_item', 'menu_item', 'begin_menu',
                      'collapsing_header', 'tree_node', 'input_double', 'combo', 'drag_float', 'slider_float', 'slider_int')}

        def record(name):
            def call(label, *args, **kwargs):
                if name == 'begin_tab_item' and label == self.selected_tab:
                    kwargs['flags'] = kwargs.get('flags', 0) | imgui.TAB_ITEM_SET_SELECTED
                result = originals[name](label, *args, **kwargs)
                self.items[label] = (tuple(imgui.get_item_rect_min()), tuple(imgui.get_item_rect_max()))
                return result
            return call

        original_begin = imgui.begin
        original_child = imgui.begin_child

        def begin(label, *args, **kwargs):
            result = original_begin(label, *args, **kwargs)
            self.windows[label] = (tuple(imgui.get_window_position()), tuple(imgui.get_window_size()))
            return result

        def child(label, *args, **kwargs):
            result = original_child(label, *args, **kwargs)
            self.windows[label] = (tuple(imgui.get_window_position()), tuple(imgui.get_window_size()))
            if label in self.scroll_children:
                imgui.set_scroll_y(self.scroll_children[label])
            return result

        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in originals:
                stack.enter_context(patch.object(imgui, name, record(name)))
            stack.enter_context(patch.object(imgui, 'begin', begin))
            stack.enter_context(patch.object(imgui, 'begin_child', child))
            s = self.state
            params = dict(app=self.app, ctx=self.ctx, bodies_data=s['bodies'], visual_data=s['visuals'],
                          atmo_bodies=[], ring_bodies=[], star_idx=[0], num_bodies=4,
                          mass_snap=s['mass'], parent_snap=s['parents'], tree_indices_snap=np.arange(4),
                          tree_depths_snap=s['depths'], subsys_mass_buf=s['mass'],
                          subsys_pos_buf=s['positions'], subsys_vel_buf=s['velocities'],
                          pos_snap_render=s['positions'], vel_snap_render=s['velocities'],
                          visual_arr=s['visuals'], cam_world_pos_f8=np.array([1., 0., .0002]),
                          cur_y=2026, cur_m=10, cur_d=6, cur_h=12, cur_mn=0, cur_s=0, cur_tz='UTC',
                          display_t=26.75, dt_render=1/60, tl_active=False, tl_prog=0,
                          tl_times=[0., 1., 2.], is_scrubbing=False, spice_valid_mask=np.ones(4, dtype=bool),
                          prog_rings=None, ring_precomputed=[], ring_render_groups=[], ring_gradient_tex=None,
                          switch_triggers=self.triggers)
            params.update(self.frame_overrides)
            params.update(overrides)
            render_ui(**params)
        imgui.render()
        if self.renderer:
            self.fbo.use()
            self.fbo.clear(.008, .012, .025, 1.)
            self.renderer.render(imgui.get_draw_data())

    def click(self, label, **overrides):
        self.frame(**overrides)
        low, high = self.items[label]
        self.io.mouse_pos = ((low[0] + high[0])/2, (low[1] + high[1])/2)
        self.io.mouse_down[0] = True
        self.frame(**overrides)
        self.io.mouse_down[0] = False
        self.frame(**overrides)

    def save_image(self, path):
        from PIL import Image
        image = Image.frombytes('RGBA', self.fbo.size, self.fbo.read(components=4))
        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path)

    def select_tab(self, label):
        # Tab headers do not reliably expose their own last-item rectangle.
        # Use ImGui's native selection request instead of a stale item rectangle.
        self.selected_tab = label
        self.frame(); self.frame()
        self.selected_tab = None

    def drag(self, label, distance):
        self.frame()
        low, high = self.items[label]
        x, y = (low[0]+high[0])/2, (low[1]+high[1])/2
        self.io.mouse_pos = (x, y)
        self.io.mouse_down[0] = True
        self.frame()
        self.io.mouse_pos = (x+distance, y)
        self.frame()
        self.io.mouse_down[0] = False
        self.frame()

    def close(self):
        if self.renderer:
            self.renderer.shutdown()
            self.fbo.release()
            self.ctx.release()
        imgui.destroy_context(self.context)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.ui = NativeUI()
        self.addCleanup(self.ui.close)
        self.ui.frame()
        self.ui.frame()

    def test_transport_and_navigation_actions(self):
        ui = self.ui
        ui.click(' Pause ')
        self.assertTrue(ui.app.time_ctrl['paused'])
        ui.click(' Forward ')
        self.assertEqual(ui.app.time_ctrl['time_direction'], -1)
        ui.app.time_ctrl['multiplier'] = 100
        ui.click('1x')
        self.assertEqual(ui.app.time_ctrl['multiplier'], 1)
        ui.click('Go To##goto_btn')
        self.assertEqual(ui.app.camera['tracking_idx'], 1)
        self.assertEqual(ui.app.camera['cam_look'], 'aim')
        ui.click('Tracking##track_btn')
        self.assertIsNone(ui.app.camera['tracking_idx'])
        ui.click('Track##track_btn')
        self.assertEqual(ui.app.camera['tracking_idx'], 1)
        ui.click('Center##center_btn')
        self.assertEqual(ui.app.camera['centered_idx'], 1)
        ui.click('Barycenter##toggle_bary_btn')
        self.assertTrue(ui.app.camera['inspect_bary'])

    def test_outliner_selection_search_crud_and_comparison(self):
        ui = self.ui
        ui.click('Mars##3')
        self.assertEqual(ui.app.camera['inspected_idx'], 3)
        ui.click('+##add_3')
        self.assertEqual(ui.app.camera['add_data']['parent_idx'], 3)
        ui.app.camera['add_mode'] = False
        ui.click('-##del_3')
        self.assertEqual(ui.app.shared_state['crud_queue'], [{'action': 'DELETE', 'idx': 3}])
        ui.app._ui_search_query = 'no such body'
        ui.frame()
        self.assertNotIn('Earth *##1', ui.items)
        ui.app._ui_search_query = ''
        ui.app.comparison_enabled = True
        ui.frame()
        ui.click('Moon##cmp_2')
        self.assertEqual(ui.app.camera['inspected_idx'], 2)
        self.assertTrue(ui.app.camera['inspected_is_cmp'])

    def test_settings_categories_and_immediate_save(self):
        ui = self.ui
        ui.click('Settings')
        self.assertTrue(ui.app.camera['show_settings_modal'])
        for section in SETTINGS_SECTIONS:
            ui.click(section)
            self.assertEqual(ui.app._ui_settings_section, SETTINGS_SECTIONS.index(section))
        ui.click('Align side panels')
        self.assertFalse(ui.app.camera['ui_aligned_panels'])
        self.assertGreater(ui.app.saves, 0)
        ui.click('Reset workspace layout')
        self.assertTrue(ui.app.camera['ui_aligned_panels'])
        ui.click('Close')
        self.assertFalse(ui.app.camera['show_settings_modal'])

    def test_inspector_tabs_capture_and_hidden_ui(self):
        ui = self.ui
        for tab in ('Orbit', 'Atmosphere', 'Rings', 'Surface###Cosmetics', 'Overview'):
            ui.select_tab(tab)
            ui.frame()
        ui.click('Capture (F12)')
        self.assertEqual(ui.app._screenshot_request, (7680, 4320))
        ui.app.ui_visible = False
        ui.frame()
        self.assertEqual(ui.windows, {})

    def test_advanced_settings_branches(self):
        ui = self.ui
        ui.app.camera['show_settings_modal'] = True
        cases = (
            (0, dict(atmo_quality=3)),
            (0, dict(atmo_quality=2, atmo_resolution=.5)),
            (0, dict(atmo_quality=0)),
            (1, dict(bloom_mode=0)),
            (1, dict(bloom_mode=1)),
            (3, dict(ringshine_mode=1)),
            (4, dict(terrain_lod_enabled=True)),
        )
        for section, settings in cases:
            ui.app._ui_settings_section = section
            ui.app.camera.update(settings)
            ui.frame()
            ui.frame()

    def test_render_shortcuts_preserve_visibility_toggle(self):
        ui = self.ui
        ui.click('Render')
        ui.click('Optics Settings...')
        self.assertEqual(ui.app._ui_settings_section, 1)
        self.assertTrue(ui.app.camera['show_settings_modal'])

    def test_timeline_resume_cancel_and_progress(self):
        ui = self.ui
        ui.click('Play Timeline', is_scrubbing=True)
        self.assertTrue(ui.app.time_ctrl['timeline_playing'])
        ui.click('Pause Playback', is_scrubbing=True)
        self.assertFalse(ui.app.time_ctrl['timeline_playing'])
        ui.click('Resume Here', is_scrubbing=True)
        self.assertIn('sync_t', ui.app.time_ctrl)
        ui.click('Cancel', tl_active=True, tl_prog=.3)
        self.assertTrue(ui.app.time_ctrl['cancel_render'])

    def test_hidpi_visibility_and_resizing(self):
        ui = self.ui
        for width, height, scale in ((800, 600, 1), (800, 600, 1.5), (1280, 720, 1), (1920, 1080, 1.25), (2560, 1440, 1.5)):
            with self.subTest(width=width, scale=scale):
                ui.io.display_size = (width, height)
                ui.app.fb_width, ui.app.fb_height = width*2, height*2
                ui.app.camera['ui_scale'] = scale
                ui.frame()
                ui.frame()
                layout = workspace(ui.app)
                self.assertEqual(layout.width, width)
                for title, (position, size) in ui.windows.items():
                    if '###' not in title:
                        continue
                    self.assertGreaterEqual(position[0], 0, title)
                    self.assertGreaterEqual(position[1], 0, title)
                    self.assertLessEqual(position[0]+size[0], width+1, title)
                    self.assertLessEqual(position[1]+size[1], height+1, title)
                outliner = ui.windows['System Outliner###outliner']
                inspector = next(v for k,v in ui.windows.items() if k.endswith('###inspector'))
                self.assertLess(outliner[0][0]+outliner[1][0], inspector[0][0])
                # All transport controls must be within the display, including their labels.
                for label in (' Pause ', ' Forward ', '1x', 'Render Timeline...'):
                    low, high = ui.items[label]
                    self.assertGreaterEqual(low[0], 0)
                    self.assertLessEqual(high[0], width)
                    self.assertLessEqual(high[1], height)
        ui.app.show_outliner = ui.app.show_time_hud = False
        ui.app.camera['inspected_idx'] = None
        ui.frame()
        self.assertIn('##camera_pill', ui.windows)
        self.assertNotIn('System Outliner###outliner', ui.windows)

    def test_size_comparator_and_geometry_overlay(self):
        ui = self.ui
        ui.app.camera.update(scene_mode='size_comparator', show_triangle_count=True)
        ui.frame()
        self.assertIn('##size_comparator_pill', ui.windows)
        ui.click('Second')
        self.assertIn('Second', ui.app.camera['size_comparator_systems'])
        pill = ui.windows['##size_comparator_pill']
        stats = ui.windows['Geometry Statistics###geom_stats']
        self.assertGreaterEqual(stats[0][1], pill[0][1] + pill[1][1])

    def test_preferences_roundtrip_in_both_writers(self):
        from engine.app import App
        for writer in (InputHandlerMixin, App):
            with self.subTest(writer=writer.__name__), tempfile.TemporaryDirectory() as directory:
                app = SimpleNamespace(camera=dict(ui_scale=1.3, ui_aligned_panels=False,
                    ui_outliner_width=310, ui_inspector_width=480, show_gaia_stars=False))
                module = sys.modules[writer.__module__]
                with patch.object(module, 'get_external_path', lambda *p: str(Path(directory).joinpath(*p))):
                    writer.save_settings(app)
                    saved = json.loads((Path(directory)/'data/graphics_settings.json').read_text())
                    for key, value in app.camera.items():
                        self.assertEqual(saved[key], value)
                    loaded = SimpleNamespace(camera={})
                    writer.load_settings(loaded)
                    self.assertEqual(loaded.camera['ui_scale'], 1.3)

    def test_edit_footer_and_tabs_stay_visible_when_scrolling(self):
        ui = self.ui
        ui.click('Edit Mode')
        self.assertTrue(ui.app.camera['edit_mode'])
        for width, height, scale in ((800, 600, 1), (800, 600, 1.5), (1280, 720, 1)):
            ui.io.display_size = (width, height)
            ui.app.camera['ui_scale'] = scale
            ui.frame(); ui.frame()
            before = ui.items['Orbit']
            ui.scroll_children['inspector_details'] = 10000
            ui.frame(); ui.frame()
            self.assertEqual(before, ui.items['Orbit'])
            footer_pos, footer_size = ui.windows['inspector_edit_footer']
            for label in ('Apply Changes##apply_edits', 'Cancel##edit_cancel'):
                low, high = ui.items[label]
                self.assertGreaterEqual(low[1], footer_pos[1])
                self.assertLessEqual(high[1], footer_pos[1]+footer_size[1])
                self.assertLessEqual(high[0], width)
                self.assertLessEqual(high[1], height)
        ui.click('Cancel##edit_cancel')
        self.assertFalse(ui.app.camera['edit_mode'])
        self.assertEqual(ui.app.shared_state['crud_queue'], [])
        ui.click('Edit Mode')
        ui.app.camera['edit_data'].update(a=.000001, e=.9)
        ui.io.display_size = (800, 600)
        ui.app.camera['ui_scale'] = 1.5
        ui.frame(); ui.frame()
        footer_pos, footer_size = ui.windows['inspector_edit_footer']
        self.assertLessEqual(ui.items['Cancel##edit_cancel_disabled'][1][1], footer_pos[1]+footer_size[1])
        self.assertIn('Navigation...', ui.items)
        ui.click('Navigation...')
        self.assertIn('Go To##goto_btn', ui.items)
        footer_pos, footer_size = ui.windows['inspector_edit_footer']
        self.assertLessEqual(ui.items['Cancel##edit_cancel_disabled'][1][1], footer_pos[1]+footer_size[1])

    def test_apply_dispatches_proposed_values_and_roche_guard_is_tab_independent(self):
        ui = self.ui
        ui.click('Edit Mode')
        ui.app.camera['edit_data']['radius'] = 6400.
        ui.click('Apply Changes##apply_edits')
        self.assertFalse(ui.app.camera['edit_mode'])
        update, = ui.app.shared_state['crud_queue']
        self.assertEqual(update['action'], 'UPDATE')
        self.assertEqual(update['idx'], 1)
        self.assertAlmostEqual(ui.state['bodies'][1]['r']*696340, 6400.)
        ui.click('Edit Mode')
        ui.app.camera['edit_data'].update(a=.000001, e=.9)
        for tab in ('Overview', 'Orbit', 'Atmosphere', 'Rings', 'Surface###Cosmetics'):
            ui.select_tab(tab)
            self.assertNotIn('Apply Changes##apply_edits', ui.items)
            self.assertIn('Cancel##edit_cancel_disabled', ui.items)
        ui.click('Cancel##edit_cancel_disabled')
        self.assertFalse(ui.app.camera['edit_mode'])
        self.assertEqual(len(ui.app.shared_state['crud_queue']), 1)

    def test_orbit_details_are_expandable_and_editable(self):
        ui = self.ui
        ui.select_tab('Orbit')
        for label in ('Orbital elements###orbit_elements', 'Gravitational limits###orbit_limits', 'Precession estimates###orbit_precession'):
            ui.click(label)
            ui.frame()
        ui.click('Edit Mode')
        self.assertIn('##Semi-Major Axis (AU)', ui.items)
        self.assertIn('##Mean Anomaly (deg)', ui.items)

    def test_comparison_inspector_uses_numeric_visual_array(self):
        ui = self.ui
        ui.app.comparison_enabled = True
        ui.app.visual_data_cmp = ui.state['visuals'].tolist()
        ui.click('Moon##cmp_2')
        for tab in ('Overview', 'Orbit', 'Atmosphere', 'Rings', 'Surface###Cosmetics'):
            ui.select_tab(tab)
        self.assertNotIn('Edit Mode', ui.items)

    def test_atmosphere_add_pressure_composition_scattering_and_remove(self):
        ui = self.ui
        atmospheres = []
        ui.frame_overrides['atmo_bodies'] = atmospheres
        ui.select_tab('Atmosphere')
        ui.click('Add Atmosphere')
        self.assertEqual(len(atmospheres), 1)
        atmosphere = atmospheres[0]
        released = []
        atmosphere['lut_tex'] = SimpleNamespace(release=lambda: released.append(True))
        original_drag = imgui.drag_float
        def pressure_event(label, value, *args, **kwargs):
            result = original_drag(label, value, *args, **kwargs)
            return (True, 1.8) if label == '##Surface Pressure (atm)' else result
        with patch.object(imgui, 'drag_float', pressure_event):
            ui.frame()
        self.assertEqual(atmosphere['surface_pressure'], 1.8)
        self.assertTrue(atmosphere['_dirty'])
        self.assertEqual(released, [True])
        original_slider = imgui.slider_float
        def gas_event(label, value, *args, **kwargs):
            result = original_slider(label, value, *args, **kwargs)
            return (True, 50.) if label == 'N2##slider' else result
        with patch.object(imgui, 'slider_float', gas_event):
            ui.frame()
        # The physics model normalizes these input weights when deriving properties.
        self.assertEqual(atmosphere['composition']['N2'], .5)
        self.assertTrue(atmosphere['_dirty'])
        # Open scattering despite being below the fold, then deliver a widget event.
        original_header = imgui.collapsing_header
        def open_scattering(label, *args, **kwargs):
            if label == 'Aerosol controls###atmo_scattering':
                imgui.set_next_item_open(True, imgui.ALWAYS)
            return original_header(label, *args, **kwargs)
        def aerosol_event(label, value, *args, **kwargs):
            result = original_drag(label, value, *args, **kwargs)
            return (True, 4.) if label == '##Aerosol Beta (x10^-6)' else result
        with patch.object(imgui, 'collapsing_header', open_scattering), patch.object(imgui, 'drag_float', aerosol_event):
            ui.frame()
        self.assertAlmostEqual(atmosphere['beta_mie'], 4.e-6)
        original_button = imgui.button
        def remove_event(label, *args, **kwargs):
            result = original_button(label, *args, **kwargs)
            return True if label == 'Remove Atmosphere' else result
        with patch.object(imgui, 'button', remove_event):
            ui.frame()
        self.assertEqual(atmospheres, [])
        self.assertNotIn('atmosphere', ui.state['bodies'][1])

    def test_ring_selection_edits_only_one_layer_and_recovers_after_delete(self):
        ui = self.ui
        rings = [dict(body_idx=1, inner_r=.00006+i*.00003, outer_r=.00008+i*.00003,
                      raw_color=(.8,.7,.5), opacity=.8, gradient=[dict(p=0.,a=0.),dict(p=1.,a=1.)]) for i in range(2)]
        ui.frame_overrides['ring_precomputed'] = rings
        ui.select_tab('Rings')
        self.assertIn('##Inner Radius (km)##0', ui.items)
        self.assertNotIn('##Inner Radius (km)##1', ui.items)
        ui.click('Layer 2 / Color##ring_select_1')
        self.assertIn('##Inner Radius (km)##1', ui.items)
        self.assertNotIn('##Inner Radius (km)##0', ui.items)
        original_drag = imgui.drag_float
        def opacity_event(label, value, *args, **kwargs):
            result = original_drag(label, value, *args, **kwargs)
            return (True, .4) if label == '##Opacity##1' else result
        with patch('engine.ui.inspector.rebuild_ring_render_group', return_value={}) as rebuild:
            with patch.object(imgui, 'drag_float', opacity_event):
                ui.frame()
            rebuild.assert_called_once()
        self.assertEqual(rings[0]['opacity'], .8)
        self.assertEqual(rings[1]['opacity'], .4)
        rings.pop(1)
        ui.frame()
        self.assertIn('##Inner Radius (km)##0', ui.items)

    def test_surface_material_events_preserve_preset_and_parameters(self):
        ui = self.ui
        ui.select_tab('Surface###Cosmetics')
        original_header = imgui.collapsing_header
        def open_material(label, *args, **kwargs):
            if label == 'Material controls###surface_materials':
                imgui.set_next_item_open(True, imgui.ALWAYS)
            return original_header(label, *args, **kwargs)
        original_combo = imgui.combo
        def preset_event(label, value, *args, **kwargs):
            result = original_combo(label, value, *args, **kwargs)
            return (True, 2) if label == '##Material preset' else result
        with patch.object(imgui, 'collapsing_header', open_material), patch.object(imgui, 'combo', preset_event):
            ui.frame()
        self.assertEqual(ui.state['bodies'][1]['material']['preset'], 'lunar')
        original_slider = imgui.slider_float
        def brightness_event(label, value, *args, **kwargs):
            result = original_slider(label, value, *args, **kwargs)
            return (True, 1.25) if label == '##Brightness' else result
        with patch.object(imgui, 'collapsing_header', open_material), patch.object(imgui, 'slider_float', brightness_event):
            ui.frame()
        material = ui.state['bodies'][1]['material']
        self.assertEqual(material['preset'], 'custom')
        self.assertEqual(material['brightness'], 1.25)


def render_previews(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for width, height, name in ((1280, 720, 'workspace-1280'), (1920, 1080, 'workspace-1920'), (800, 600, 'workspace-compact')):
        ui = NativeUI(width, height, render=True)
        try:
            ui.frame(); ui.frame()
            ui.save_image(directory/(name+'.png'))
            if width == 1280:
                ui.app.camera['show_settings_modal'] = True
                for index in range(len(SETTINGS_SECTIONS)):
                    ui.app._ui_settings_section = index
                    ui.frame(); ui.frame()
                    ui.save_image(directory/('settings-'+SETTINGS_SECTIONS[index].lower()+'.png'))
        finally:
            ui.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render')
    args, remaining = parser.parse_known_args()
    if args.render:
        render_previews(args.render)
    else:
        unittest.main(argv=[sys.argv[0]] + remaining)
