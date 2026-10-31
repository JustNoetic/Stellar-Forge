import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 268435456


def bake(source, output, width=None, channel="auto"):
    source, output = Path(source), Path(output)
    with Image.open(source) as image:
        target_width = image.width if width is None else width
        if target_width < 2 or target_width > 16384 or target_width % 2:
            raise ValueError("Width must be even and from 2 to 16384")
        if width is None and image.width != 2 * image.height:
            raise ValueError("Native cloud coverage must be a 2:1 image")
        has_alpha = "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info)
        alpha = channel == "alpha" or (channel == "auto" and has_alpha)
        if channel == "alpha" and not has_alpha:
            raise ValueError("The source has no alpha channel")
        mask = image.convert("RGBA").getchannel("A") if alpha else image.convert("L")
        if mask.size != (target_width, target_width // 2):
            mask = mask.resize((target_width, target_width // 2), Image.Resampling.LANCZOS)
    output.mkdir(parents=True, exist_ok=True)
    levels, payload = [], bytearray()
    while True:
        levels.append({"width": mask.width, "height": mask.height, "offset": len(payload)})
        payload.extend(mask.tobytes())
        if mask.size == (1, 1):
            break
        mask = mask.resize((max(1, mask.width // 2), max(1, mask.height // 2)), Image.Resampling.BOX)
    (output / "coverage.r8").write_bytes(payload)
    manifest = {"version": 1, "encoding": "r8-unorm", "projection": "equirectangular",
                "orientation": "north-top-east-right", "levels": levels,
                "source": source.name, "channel": "alpha" if alpha else "luminance",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "payload_sha256": hashlib.sha256(payload).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake a cloud coverage image to a linear grayscale mip pyramid")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, help="Optional even output width up to 16384; default preserves native dimensions")
    parser.add_argument("--channel", choices=("auto", "luminance", "alpha"), default="auto")
    args = parser.parse_args()
    result = bake(args.source, args.output, args.width, args.channel)
    print(f"Baked {len(result['levels'])} cloud coverage mip levels to {args.output}")
