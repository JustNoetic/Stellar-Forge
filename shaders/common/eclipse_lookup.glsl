layout(set = 0, binding = 26) uniform sampler2DArray eclipse_shadow_atlas;

vec4 eclipse_lookup_record(uint body, uint star) {
    if (u_scene.eclipse_params.y == 0u) return vec4(0.0);
    uint offset = u_scene.eclipse_params.y+(body*u_scene.num_stars+star)*4u;
    return uintBitsToFloat(uvec4(eclipse_words[offset], eclipse_words[offset+1u], eclipse_words[offset+2u], eclipse_words[offset+3u]));
}

vec4 eclipse_lookup_visibility(vec4 record, float radius, float source, float separation) {
    if (record.x < .5 || source <= 0.0 || radius <= 0.0) return vec4(0.0);
    float q = source/radius, v = separation/(radius+source);
    if (q < .002 || q > 4.0 || v >= record.z) return vec4(0.0);
    float x, inner = abs(1.0-q)/(1.0+q);
    if (v <= 1.0) {
        bool umbra = v < inner;
        float t = umbra ? v/inner : (v-inner)/(1.0-inner);
        float a = sqrt(clamp(t, 0.0, 1.0)), b = sqrt(clamp(1.0-t, 0.0, 1.0));
        x = (umbra ? 0.0 : .375)+.375*a/(a+b);
    }
    else x = .75+.25*sqrt((v-1.0)/(record.z-1.0));
    vec2 size = vec2(textureSize(eclipse_shadow_atlas, 0).xy);
    float pivot = floor(size.y*.75);
    float row = q <= 1.0 ? pivot*log(q/.002)/log(500.0) : pivot+(size.y-1.0-pivot)*log(q)/log(4.0);
    vec2 uv = (vec2(x*(size.x-1.0), row)+.5)/size;
    float domain = smoothstep(.002, .004, q)*(1.0-smoothstep(3.0, 4.0, q));
    vec3 visibility = exp2(textureLod(eclipse_shadow_atlas, vec3(uv, record.x-1.0), 0.0).rgb);
    return vec4(visibility, record.w*domain);
}
