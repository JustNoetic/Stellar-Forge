# -*- mode: python ; coding: utf-8 -*-
# stellar_forge.spec — PyInstaller build configuration for Stellar-Forge
#
# Usage:  pyinstaller stellar_forge.spec --noconfirm
# Or run: build.bat  (handles venv activation and post-build copies)
#
# Layout produced in dist/Stellar-Forge/:
#   Stellar-Forge.exe      ← the launcher (no console window)
#   *.dll / _internal/     ← Python runtime + all Python packages
#   engine/glsl/           ← GLSL shaders (bundled read-only data)
#   [data/]                ← copied by build.bat (external, user-editable)
#   [textures/]            ← copied by build.bat (external, user-editable)
#   [exports/]             ← created by build.bat (screenshot output)

import os
import importlib.util

# GLFW loads its DLL via ctypes — PyInstaller can't auto-detect it,
# so we locate and include it explicitly.
_glfw_binaries = []
try:
    _spec = importlib.util.find_spec('glfw')
    if _spec and _spec.origin:
        _glfw_dir = os.path.dirname(_spec.origin)
        for _dll in ('glfw3.dll', 'msvcr120.dll'):
            _p = os.path.join(_glfw_dir, _dll)
            if os.path.isfile(_p):
                _glfw_binaries.append((_p, 'glfw'))
except Exception:
    pass

if not _glfw_binaries and os.path.isdir(r'venv\Lib\site-packages\glfw'):
    for _dll in ('glfw3.dll', 'msvcr120.dll'):
        _p = os.path.join(r'venv\Lib\site-packages\glfw', _dll)
        if os.path.isfile(_p):
            _glfw_binaries.append((_p, 'glfw'))

block_cipher = None

a = Analysis(
    [r'engine\main.py'],
    pathex=['.'],
    binaries=_glfw_binaries,
    datas=[
        # GLSL shaders — bundled inside the exe (read-only engine internals)
        (r'engine\glsl', r'engine\glsl'),
    ],
    hiddenimports=[
        # Core scientific stack
        'numpy',
        'numpy.core._multiarray_umath',
        'scipy',
        'scipy.spatial',
        'scipy.spatial.transform',
        'scipy.special',
        'scipy.special._ufuncs',
        # Graphics
        'moderngl',
        'moderngl.mgl',
        'glfw',
        'OpenGL',
        'OpenGL.GL',
        'OpenGL.GL.shaders',
        'OpenGL.platform',
        'OpenGL.platform.win32',
        # ImGui
        'imgui',
        'imgui.core',
        'imgui.integrations',
        'imgui.integrations.opengl',
        # Math / geometry
        'pyrr',
        'pyrr.matrix44',
        'pyrr.vector3',
        # Numba JIT (needs all sub-packages collected)
        'numba',
        'numba.core',
        'numba.core.types',
        'numba.typed',
        'numba.np.ufunc',
        'numba.cpython',
        'numba.cuda',
        # SPICE
        'spiceypy',
        'spiceypy.utils',
        # Image loading
        'PIL',
        'PIL.Image',
        'PIL.ImageFilter',
        'PIL._imaging',
        # Networking (for kernel/Horizons downloads)
        'requests',
        'requests.adapters',
        'urllib',
        'urllib.request',
        # Engine packages
        'engine',
        'engine.core',
        'engine.ephemeris',
        'engine.physics',
        'engine.rendering',
        'engine.ui',
        'engine.path_utils',
        # Scripts used at runtime by engine modules
        'scripts',
        'scripts.fetch_artemis2_kernel',
    ],
    collect_all=[
        # These packages need ALL their data files and sub-modules
        'numba',
        'spiceypy',
        'moderngl',
        'glfw',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        '_tkinter',
        'matplotlib',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Stellar-Forge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX can break some native extensions; keep off
    console=False,      # No terminal window — errors go to main_error.txt
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,          # Set to an .ico path here if you want a custom icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Stellar-Forge',
)
