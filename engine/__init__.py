import sys

from engine.core import constants, math_utils, input_handler
from engine.physics import physics_core, kepler_analytical, atmosphere_physics, star_calc
from engine.rendering import render_utils, imgui_renderer, planetshine, texture_baker, shader_loader, shaders, post_shaders
from engine.ephemeris import system_manager, spice_manager

# Backward-compatibility sys.modules aliases
sys.modules['constants'] = constants
sys.modules['math_utils'] = math_utils
sys.modules['input_handler'] = input_handler
sys.modules['physics_core'] = physics_core
sys.modules['kepler_analytical'] = kepler_analytical
sys.modules['atmosphere_physics'] = atmosphere_physics
sys.modules['star_calc'] = star_calc
sys.modules['render_utils'] = render_utils
sys.modules['imgui_renderer'] = imgui_renderer
sys.modules['planetshine'] = planetshine
sys.modules['texture_baker'] = texture_baker
sys.modules['shader_loader'] = shader_loader
sys.modules['shaders'] = shaders
sys.modules['post_shaders'] = post_shaders
sys.modules['system_manager'] = system_manager
sys.modules['spice_manager'] = spice_manager

from engine.core.constants import *
from engine.core.math_utils import *
from engine.physics.physics_core import *
from engine.rendering.shaders import *
from engine.ephemeris.system_manager import *
from engine.ephemeris.spice_manager import *
