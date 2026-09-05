import os
import re

_SHADER_CACHE = {}
_GLSL_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "glsl")
_INCLUDE_PATTERN = re.compile(r'^\s*#include\s+["<]([^">]+)[">]\s*$', re.MULTILINE)

def _resolve_includes(code: str, current_dir: str, visited: set | None = None) -> str:
    if visited is None:
        visited = set()

    def _replace_match(match):
        inc_path = match.group(1).strip()
        candidate1 = os.path.normpath(os.path.join(current_dir, inc_path))
        candidate2 = os.path.normpath(os.path.join(_GLSL_BASE_DIR, inc_path))

        if os.path.isfile(candidate1):
            full_inc_path = candidate1
        elif os.path.isfile(candidate2):
            full_inc_path = candidate2
        else:
            raise FileNotFoundError(
                f"GLSL include file not found: '{inc_path}' "
                f"(searched in '{current_dir}' and '{_GLSL_BASE_DIR}')"
            )

        if full_inc_path in visited:
            return f"// Cyclical include ignored: {inc_path}"

        new_visited = set(visited)
        new_visited.add(full_inc_path)

        with open(full_inc_path, "r", encoding="utf-8") as f_inc:
            inc_code = f_inc.read()

        return _resolve_includes(inc_code, os.path.dirname(full_inc_path), new_visited)

    return _INCLUDE_PATTERN.sub(_replace_match, code)

def load_shader(rel_path: str) -> str:
    """
    Loads a GLSL shader file from the engine/glsl directory and caches it in memory.
    rel_path should be relative to engine/glsl, e.g., 'compute/culling.comp'
    Resolves any `#include "..."` or `#include <...>` directives recursively.
    """
    if rel_path in _SHADER_CACHE:
        return _SHADER_CACHE[rel_path]
    
    full_path = os.path.join(_GLSL_BASE_DIR, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"GLSL shader file not found: {full_path}")
        
    with open(full_path, "r", encoding="utf-8") as f:
        code = f.read()

    resolved_code = _resolve_includes(code, os.path.dirname(full_path))
        
    _SHADER_CACHE[rel_path] = resolved_code
    return resolved_code

def clear_shader_cache():
    """Clears the in-memory shader cache (useful for live shader reloading)."""
    _SHADER_CACHE.clear()
