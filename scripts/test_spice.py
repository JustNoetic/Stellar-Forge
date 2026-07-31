import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.ephemeris.spice_manager import SpiceManager

sm = SpiceManager()
sm.load_kernels()

with open('data/system.json', 'r') as f:
    template = json.load(f)

bundle = sm.build_ephemeris_system(0, template)

print("Bodies in bundle:")
for b in bundle:
    print(f"- {b['name']} (parentId: {b.get('parentId')}, has_sv: {'sv' in b})")
