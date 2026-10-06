"""Load the native terrain module; never silently fall back to the old terrain."""
try:
    import stellar_terrain as native
except ImportError as exc:
    raise RuntimeError(
        'The Rust terrain backend is missing. Run python scripts/build_terrain.py '
        'with Rust installed, then restart Stellar Forge.'
    ) from exc

if native.PATCH_FLOATS != 24:
    raise RuntimeError('The terrain backend is out of date. Rebuild with scripts/build_terrain.py.')


def terrain_water_level(body):
    """Negative elevation alone is not evidence that a body has oceans."""
    level = body.get('height_water_level_km', 0. if body.get('name', '').lower() == 'earth' else None)
    if level is None:
        return None
    import math
    value = float(level)
    return value if math.isfinite(value) else None

