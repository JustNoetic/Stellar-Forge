import os

_SHADER_CACHE = {}
_GLSL_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "glsl")

def load_shader(rel_path: str) -> str:
    """
    Loads a GLSL shader file from the engine/glsl directory and caches it in memory.
    rel_path should be relative to engine/glsl, e.g., 'compute/culling.comp'
    """
    if rel_path in _SHADER_CACHE:
        return _SHADER_CACHE[rel_path]
    
    full_path = os.path.join(_GLSL_BASE_DIR, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"GLSL shader file not found: {full_path}")
        
    with open(full_path, "r", encoding="utf-8") as f:
        code = f.read()
        
    _SHADER_CACHE[rel_path] = code
    return code

def clear_shader_cache():
    """Clears the in-memory shader cache (useful for live shader reloading)."""
    _SHADER_CACHE.clear()
