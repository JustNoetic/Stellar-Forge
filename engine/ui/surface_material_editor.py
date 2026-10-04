"""Surface photometry controls in the body inspector's Cosmetics tab."""
import copy
import imgui

from engine.rendering.surface_materials import MATERIAL_PRESETS, resolve_material


def render_surface_material_editor(body):
    imgui.text_colored('Surface Material', 1.0, 0.85, 0.4)
    raw = body.get('material')
    preset = raw if isinstance(raw,str) else raw.get('preset','auto') if isinstance(raw,dict) else 'auto'
    choices = ['auto','lambert','lunar','icy','custom']
    labels = ['Automatic','Lambert','Lunar regolith','Icy regolith','Custom Hapke']
    changed, index = imgui.combo('Material preset', choices.index(preset) if preset in choices else 4, labels)
    if changed:
        selected = choices[index]
        if selected == 'auto':
            body.pop('material',None)
        elif selected == 'custom':
            body['material'] = dict(resolve_material(body),preset='custom',model='hapke')
        else:
            body['material'] = dict(copy.deepcopy(MATERIAL_PRESETS[selected]),preset=selected)
    material = resolve_material(body)
    imgui.text('Resolved: '+('Hapke granular surface' if material['model'] == 'hapke' else 'Lambert surface'))
    if imgui.is_item_hovered():
        imgui.set_tooltip('Automatic uses lunar regolith for Moon/Luna and icy regolith for Europa. Other bodies use Lambert. Presets are editable starting points.')
    controls = [('Brightness','brightness',0.0,2.0,'%.2f')]
    if material['model'] == 'hapke':
        controls += [
            ('Grain scattering albedo','single_scattering_albedo',0.001,0.999,'%.3f'),
            ('Scattering anisotropy','phase_width',0.0,0.85,'%.3f'),
            ('Backward scattering fraction','backscatter_fraction',0.0,1.0,'%.3f'),
            ('Shadow opposition strength','opposition_strength',0.0,2.0,'%.3f'),
            ('Shadow opposition width','opposition_width',0.003,0.5,'%.3f'),
            ('Roughness (degrees)','roughness_deg',0.0,45.0,'%.1f'),
            ('Coherent opposition strength','coherent_strength',0.0,1.0,'%.3f'),
            ('Coherent opposition width','coherent_width',0.003,0.1,'%.3f'),
        ]
    for label, name, low, high, fmt in controls:
        changed, value = imgui.slider_float(label,material[name],low,high,format=fmt)
        if changed:
            material[name] = value
            body['material'] = dict(material,preset='custom' if material['model'] == 'hapke' else 'lambert')
        if imgui.is_item_hovered():
            if name == 'single_scattering_albedo':
                imgui.set_tooltip('Fraction scattered rather than absorbed by a grain. Controls multiple scattering; texture brightness is anchored at phase 30 degrees.')
            elif name == 'backscatter_fraction':
                imgui.set_tooltip('0 prefers forward scattering; 1 returns light toward the source.')
            elif name.endswith('width') and 'opposition' in name:
                imgui.set_tooltip('Dimensionless Hapke h: half-angle tangent scale. Larger values spread the opposition brightening over more phase angles.')
    if material['model'] == 'hapke':
        imgui.text_wrapped('Granular surface scattering. Textures set reference reflectance; cloud and atmosphere scattering are handled separately.')
