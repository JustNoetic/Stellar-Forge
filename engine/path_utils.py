"""
path_utils.py — centralised path resolution for Stellar-Forge.

Handles the difference between running from source and running as a
PyInstaller-frozen .exe, where ``__file__``-relative paths can break
because ``sys._MEIPASS`` (the extraction temp dir) differs from the
directory that contains the ``.exe``.

Two concepts:
  - **bundled** assets  — read-only engine internals packed *inside* the .exe
                          (GLSL shaders, etc.)
  - **external** assets — user-editable files that live *next to* the .exe
                          (textures, data/, kernels/, exports/, imgui.ini …)
"""

import os
import sys


def _project_root() -> str:
    """Return the project root directory.

    * Frozen (``.exe``): the directory that *contains* the ``.exe`` file.
    * Source: two levels up from this file (``engine/path_utils.py``
      →  ``engine/`` → project root).
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # This file lives at  <root>/engine/path_utils.py
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_bundled_path(*parts: str) -> str:
    """Return the path to a file that is bundled *inside* the ``.exe``.

    When frozen, PyInstaller extracts bundled data to ``sys._MEIPASS``.
    When running from source, falls back to the project root so that the
    same relative layout works in both environments.

    Example::

        get_bundled_path("engine", "glsl", "celestial", "planet.vert")
    """
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, *parts)
    return os.path.join(_project_root(), *parts)


def get_external_path(*parts: str) -> str:
    """Return the path to a file that lives *outside* the ``.exe``.

    Always resolved relative to the project root (or the ``.exe``'s
    directory when frozen), so that user-editable files remain accessible
    regardless of the working directory.

    Example::

        get_external_path("data", "graphics_settings.json")
        get_external_path("textures")
        get_external_path("data", "kernels")
    """
    return os.path.join(_project_root(), *parts)
