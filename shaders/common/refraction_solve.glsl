#ifndef REFRACTION_SOLVE_GLSL
#define REFRACTION_SOLVE_GLSL

const int REFRACTION_ROWS = 512;
const int REFRACTION_COLUMNS = 256;
const int REFRACTION_PROFILE_WIDTH = 1024;
const int REFRACTION_OBSERVER_WIDTH = 512;
const int REFRACTION_OBSERVER_HEIGHT = 128;
const int REFRACTION_SURFACE_ROW = REFRACTION_OBSERVER_HEIGHT/2;
const float REFRACTION_PI = 3.14159265358979f;
const float REFRACTION_FAR = 1e20f;

struct RefractionGPU {
    vec4 host;
    vec4 pole;
    vec4 frame;
    vec4 mapping;
    vec4 model;
    vec4 profile;
    vec4 escape;
};

vec2 refraction_gauss(int i) {
    if (i == 0) return vec2(-0.9602898564975363f, 0.1012285362903763f);
    if (i == 1) return vec2(-0.7966664774136267f, 0.2223810344533745f);
    if (i == 2) return vec2(-0.5255324099163290f, 0.3137066458778873f);
    if (i == 3) return vec2(-0.1834346424956498f, 0.3626837833783620f);
    if (i == 4) return vec2(0.1834346424956498f, 0.3626837833783620f);
    if (i == 5) return vec2(0.5255324099163290f, 0.3137066458778873f);
    if (i == 6) return vec2(0.7966664774136267f, 0.2223810344533745f);
    return vec2(0.9602898564975363f, 0.1012285362903763f);
}

vec4 refraction_profile_texel(sampler2D table, int i) {
    return texelFetch(table, ivec2(i % REFRACTION_PROFILE_WIDTH, i / REFRACTION_PROFILE_WIDTH), 0);
}

vec2 refraction_profile(sampler2D table, RefractionGPU g, float h) {
    if (h >= g.frame.z) return vec2(0.0f, 0.0f);
    float x = (h-g.model.w)/g.profile.x;
    if (x <= 0.0f) {
        vec4 a = refraction_profile_texel(table, 0);
        return vec2(a.x+a.y*(h-g.model.w), a.y);
    }
    int i = min(int(x), int(g.profile.y)-2);
    float t = x-float(i), step = g.profile.x;
    vec4 a = refraction_profile_texel(table, i);
    vec4 b = refraction_profile_texel(table, i+1);
    float t2 = t*t, t3 = t2*t;
    float value = (2.0f*t3-3.0f*t2+1.0f)*a.x+(t3-2.0f*t2+t)*step*a.y+(-2.0f*t3+3.0f*t2)*b.x+(t3-t2)*step*b.y;
    float slope = ((6.0f*t2-6.0f*t)*a.x+(3.0f*t2-4.0f*t+1.0f)*step*a.y+(-6.0f*t2+6.0f*t)*b.x+(3.0f*t2-2.0f*t)*step*b.y)/step;
    return vec2(value, slope);
}

float refraction_vacuum(RefractionGPU g, float q, float h) {
    if (h >= REFRACTION_FAR) return 0.5f*REFRACTION_PI;
    float gap = h-q;
    if (gap <= 0.0f) return 0.0f;
    float radius = g.frame.y;
    return atan(sqrt(gap*(2.0f*radius+h+q)), radius+q);
}

float refraction_catmull(float y0, float y1, float y2, float y3, float w) {
    return y1+0.5f*w*(y2-y0+w*(2.0f*y0-5.0f*y1+4.0f*y2-y3+w*(3.0f*(y1-y2)+y3-y0)));
}

float refraction_row_position(RefractionGPU g, float q) {
    float coordinate = sqrt(asinh(max(q-g.model.x, 0.0f)/g.model.y)/g.model.z);
    return clamp(coordinate, 0.0f, 1.0f)*float(REFRACTION_ROWS-1);
}

float refraction_row_offset(sampler2D rows, int i) {
    return texelFetch(rows, ivec2(clamp(i, 0, REFRACTION_ROWS-1), 0), 0).y;
}

float refraction_turn(sampler2D rows, sampler2D profile, RefractionGPU g, float q) {
    if (q >= g.frame.z) return q;
    float row = refraction_row_position(g, q);
    int i = min(int(row), REFRACTION_ROWS-2);
    float w = row-float(i);
    float turn = q-refraction_catmull(refraction_row_offset(rows, i-1), refraction_row_offset(rows, i),
                                      refraction_row_offset(rows, i+1), refraction_row_offset(rows, i+2), w);
    if (g.profile.w > 0.1f) {
        vec4 a = texelFetch(rows, ivec2(i, 0), 0), b = texelFetch(rows, ivec2(i+1, 0), 0);
        float lo = a.x-a.y, hi = b.x-b.y;
        for (int k = 0; k < 2; ++k) {
            vec2 medium = refraction_profile(profile, g, turn);
            float radius = g.frame.y+turn;
            float slope = 1.0f+medium.x+medium.y*radius;
            if (slope > 1e-8f) turn = clamp(turn-((turn-q)+medium.x*radius)/slope, lo, hi);
        }
    }
    return turn;
}

float refraction_travel_row(sampler2D travel, int row, int column, float f) {
    int i = clamp(row, 0, REFRACTION_ROWS-1);
    float a = texelFetch(travel, ivec2(column, i), 0).x;
    float b = texelFetch(travel, ivec2(column+1, i), 0).x;
    return a+(b-a)*f;
}

float refraction_table_travel(sampler2D travel, sampler2D rows, sampler2D profile, RefractionGPU g, float q, float h) {
    float top = g.frame.z;
    if (q >= top) return refraction_vacuum(g, q, h);
    float tail = h > top ? refraction_vacuum(g, q, h)-refraction_vacuum(g, q, top) : 0.0f;
    float turn = refraction_turn(rows, profile, g, q);
    float inside = min(h, top);
    if (inside <= turn) return tail;
    float span = sqrt(max(top-turn, 0.0f));
    float x = span > 0.0f ? sqrt(max(inside-turn, 0.0f))/span : 0.0f;
    float column = clamp(x, 0.0f, 1.0f)*float(REFRACTION_COLUMNS-1);
    int j = min(int(column), REFRACTION_COLUMNS-2);
    float f = column-float(j);
    float row = refraction_row_position(g, q);
    int i = min(int(row), REFRACTION_ROWS-2);
    float w = row-float(i);
    float value = refraction_catmull(refraction_travel_row(travel, i-1, j, f), refraction_travel_row(travel, i, j, f),
                                     refraction_travel_row(travel, i+1, j, f), refraction_travel_row(travel, i+2, j, f), w);
    return span*value+tail;
}

float refraction_direct(sampler2D profile, RefractionGPU g, float q, float a, float b) {
    float lo = min(a, b), hi = max(a, b);
    float top = min(hi, g.frame.z), radius = g.frame.y;
    float integral = 0.0f;
    if (top > lo) {
        int panels = g.profile.w > 0.1f ? 8 : 2;
        for (int k = 0; k < panels; ++k) {
            for (int n = 0; n < 8; ++n) {
                vec2 node = refraction_gauss(n);
                float x = (float(k)+0.5f+0.5f*node.x)/float(panels);
                float h = lo+(top-lo)*x*x;
                float refractivity = refraction_profile(profile, g, h).x;
                float r = radius+h;
                float minus = (h-q)+refractivity*r;
                float plus = 2.0f*radius+h+q+refractivity*r;
                integral += node.y/(2.0f*float(panels))*(top-lo)*2.0f*x*(radius+q)/(r*sqrt(max(minus*plus, 1e-30f)));
            }
        }
    }
    if (hi > top) integral += refraction_vacuum(g, q, hi)-refraction_vacuum(g, q, max(lo, top));
    return integral;
}

float refraction_invariant(RefractionGPU g, float zenith) {
    float camera = g.frame.x, radius = g.frame.y;
    float camera_radius = radius+camera;
    float refractivity = g.profile.z;
    float half_angle = sin(0.5f*(0.5f*REFRACTION_PI-zenith));
    return camera+refractivity*camera_radius-(1.0f+refractivity)*camera_radius*2.0f*half_angle*half_angle;
}

vec2 refraction_reach_impact(sampler2D travel, sampler2D rows, sampler2D profile, RefractionGPU g, float q, bool inward, float target, float floor_height, bool outbound) {
    float camera = g.frame.x;
    bool covered = q >= g.model.x;
    bool blocked = q > g.escape.x && target > g.escape.y;
    if (!inward) {
        if (target < camera || blocked) return vec2(0.0f, 0.0f);
        float t = covered ? refraction_table_travel(travel, rows, profile, g, q, target)-refraction_table_travel(travel, rows, profile, g, q, camera)
                          : refraction_direct(profile, g, q, camera, target);
        return vec2(t, 1.0f);
    }
    float turn = covered ? refraction_turn(rows, profile, g, q) : -REFRACTION_FAR;
    bool grounded = !covered || turn < floor_height;
    if (target < camera && !outbound) {
        if (target < (grounded ? floor_height : turn)) return vec2(0.0f, 0.0f);
        float t = covered ? refraction_table_travel(travel, rows, profile, g, q, camera)-refraction_table_travel(travel, rows, profile, g, q, target)
                          : refraction_direct(profile, g, q, target, camera);
        return vec2(t, 1.0f);
    }
    if (grounded || blocked || target < turn) return vec2(0.0f, 0.0f);
    return vec2(refraction_table_travel(travel, rows, profile, g, q, camera)+refraction_table_travel(travel, rows, profile, g, q, target), 1.0f);
}

vec2 refraction_reach(sampler2D travel, sampler2D rows, sampler2D profile, RefractionGPU g, float zenith, float target, float floor_height, bool outbound) {
    return refraction_reach_impact(travel, rows, profile, g, refraction_invariant(g, zenith), zenith > 0.5f*REFRACTION_PI, target, floor_height, outbound);
}

float refraction_deep_fraction(RefractionGPU g) {
    float lower = g.escape.w*sinh(g.mapping.y);
    return g.profile.w < 0.1f ? 0.25f*clamp(2.0f*(1.0f+lower/g.frame.y), 0.0f, 1.0f) : 0.0f;
}

vec3 refraction_chord(RefractionGPU g, float x) {
    float camera = g.frame.x, radius = g.frame.y;
    float camera_radius = radius+camera;
    if (g.frame.w < 0.5f) {
        float horizon = g.mapping.x, band = g.mapping.y;
        float u = 2.0f*x-1.0f;
        float below = u <= 0.75f ? band*u/0.75f : band+(REFRACTION_PI-horizon-band)*(u-0.75f)*(u-0.75f)/0.0625f;
        float theta = x <= 0.5f ? horizon*(1.0f-(1.0f-2.0f*x)*(1.0f-2.0f*x)) : horizon+below;
        float half_angle = sin(0.5f*(0.5f*REFRACTION_PI-theta));
        float impact = camera-camera_radius*2.0f*half_angle*half_angle;
        return vec3(theta, impact, -camera_radius*cos(theta));
    }
    float lower = g.escape.w*sinh(g.mapping.y);
    float fraction = refraction_deep_fraction(g);
    float deep = max(1.0f-x/max(fraction, 1e-20f), 0.0f);
    float impact = x < fraction ? lower-(radius+lower)*deep*deep
        : g.escape.w*sinh(mix(g.mapping.y, g.mapping.z, (x-fraction)/(1.0f-fraction)));
    impact = max(impact, -radius);
    float b = radius+impact;
    float along = sqrt(max((camera_radius-b)*(camera_radius+b), 0.0f));
    return vec3(atan(b, -along), impact, along);
}

vec2 refraction_distance_frame(RefractionGPU g, vec3 chord) {
    float radius = g.frame.y, b = radius+chord.y;
    float surface = sqrt(max(-chord.y*(radius+b), 0.0f));
    float scale = g.escape.w/max(surface/radius, sqrt(g.escape.w/radius));
    return vec2(chord.z-surface, scale);
}

bool refraction_surface_distance(RefractionGPU g) {
    return g.frame.w > 0.5f && ((g.profile.w < 0.1f && g.frame.z < 0.1f*g.frame.y) || g.mapping.w > 8.0f*g.frame.y);
}

float refraction_cell_distance(RefractionGPU g, vec3 chord, float y) {
    if (refraction_surface_distance(g)) {
        if (y >= 1.0f) return 2.0f*REFRACTION_FAR;
        vec2 frame = refraction_distance_frame(g, chord);
        float split = float(REFRACTION_SURFACE_ROW)/float(REFRACTION_OBSERVER_HEIGHT-1);
        float u = y < split ? y/split-1.0f : (y-split)/(1.0f-split);
        return max(frame.x+frame.y*u/max(1.0f-abs(u), 1e-20f), 0.0f);
    }
    return y >= 1.0f ? 2.0f*REFRACTION_FAR : g.mapping.w*y/(1.0f-y);
}

float refraction_born(sampler2D profile, RefractionGPU g, vec3 chord, float distance) {
    float radius = g.frame.y, impact = chord.y, along = chord.z;
    float b = radius+impact;
    float rotation = 0.0f;
    for (int pass = 0; pass < 2; ++pass) {
        float sum = 0.0f;
        for (int k = 0; k < 2; ++k) {
            for (int n = 0; n < 8; ++n) {
                vec2 node = refraction_gauss(n);
                float x = (float(k)+0.5f+0.5f*node.x)/2.0f;
                float s = distance*x;
                float beyond = s-along;
                float squared = impact*(2.0f*radius+impact)+beyond*beyond;
                float r = sqrt(max(radius*radius+squared, 0.0f));
                float h = squared/(r+radius)+rotation*s*(1.0f-x)*b/r;
                vec2 medium = refraction_profile(profile, g, h);
                float curvature = -medium.y*(b/r)/(1.0f+medium.x);
                sum += 0.25f*node.y*(1.0f-x)*curvature;
            }
        }
        rotation = distance*sum;
    }
    return rotation;
}

float refraction_short_visibility(RefractionGPU g, vec3 chord, float distance, float rotation, float floor_height) {
    if (distance <= 0.0f) return 1.0f;
    float radius = g.frame.y+g.frame.x;
    float mu = -chord.z/radius, sine = (g.frame.y+chord.y)/radius;
    float linear = mu+rotation*sine;
    float quadratic = 0.5f*sine*sine/radius-rotation*sine/distance;
    float s = quadratic > 0.0f ? clamp(-linear/(2.0f*quadratic), 0.0f, distance) : 0.0f;
    float minimum = g.frame.x;
    for (int i = 0; i < 2; ++i) {
        float at = i == 0 ? distance : s;
        float squared = g.frame.x*(2.0f*g.frame.y+g.frame.x)+2.0f*radius*mu*at+at*at;
        float r = sqrt(max(g.frame.y*g.frame.y+squared, 0.0f));
        float height = squared/(r+g.frame.y)+rotation*at*(1.0f-at/distance)*sine;
        minimum = min(minimum, height);
    }
    return minimum >= floor_height-1e-5f ? 1.0f : 0.0f;
}

vec3 refraction_preserve_range(vec3 original, vec3 apparent, vec4 pole) {
    vec3 axis = vec3(pole.x, pole.y, pole.z);
    vec3 world = apparent-axis*(dot(apparent, axis)*pole.w/(1.0f+pole.w));
    return normalize(world)*length(original);
}

float refraction_exterior_rotation(RefractionGPU g, float q, float impact, float along) {
    float radius = g.frame.y+g.frame.x;
    float b = g.frame.y+impact, p = g.frame.y+q;
    float apparent_along = sqrt(max((radius-p)*(radius+p), 0.0f));
    float delta = q-impact;
    float cross_value = delta*(along+b*(p+b)/max(along+apparent_along, 1e-20f));
    return atan(cross_value, along*apparent_along+b*p);
}

vec2 refraction_solve(sampler2D travel, sampler2D rows, sampler2D profile, RefractionGPU g, vec3 chord, float distance, bool surface_endpoint) {
    if (distance <= 0.0f) return vec2(0.0f, 1.0f);
    if (chord.y <= -g.frame.y) return vec2(0.0f, chord.z <= 0.0f || distance <= g.frame.x ? 1.0f : -1.0f);
    if (g.frame.w > 0.5f) {
        float entry = chord.z-sqrt(max((g.frame.z-chord.y)*(2.0f*g.frame.y+g.frame.z+chord.y), 0.0f));
        if (distance <= entry) return vec2(0.0f, 1.0f);
    }
    if (distance <= 0.005f*g.frame.y) {
        float rotation = refraction_born(profile, g, chord, distance);
        float valid = refraction_short_visibility(g, chord, distance, rotation, g.model.w);
        float visible = refraction_short_visibility(g, chord, distance, rotation, 0.0f);
        return vec2(clamp(rotation, -g.profile.w, g.profile.w), valid > 0.5f ? 2.0f*visible-1.0f : 0.0f);
    }
    float radius = g.frame.y;
    float theta = chord.x, impact = chord.y, along = chord.z;
    float b = radius+impact;
    float target = REFRACTION_FAR*2.0f, polar = theta;
    if (distance < REFRACTION_FAR) {
        float beyond = distance-along;
        float squared = impact*(2.0f*radius+impact)+beyond*beyond;
        float r = sqrt(max(radius*radius+squared, 0.0f));
        target = squared/(r+radius);
        if (surface_endpoint) target = max(impact, 0.0f);
        polar = atan(along, b)+atan(beyond, b);
    }
    float physical_target = target, physical_polar = polar;
    float lower_q = g.model.x+(g.profile.w > 0.1f ? 0.0f : max(1e-4f, 1e-6f*g.model.y));
    float continuation_floor = g.profile.w > 0.1f ? g.model.w : max(g.model.w, refraction_turn(rows, profile, g, lower_q));
    target = max(target, continuation_floor);
    vec2 limit = refraction_reach_impact(travel, rows, profile, g, lower_q, true, target, -REFRACTION_FAR, true);
    if (limit.y > 0.5f) polar = min(polar, limit.x);
    float geometric_theta = theta, geometric_impact = impact, geometric_along = along;
    if (target != physical_target || polar != physical_polar) {
        if (target >= REFRACTION_FAR) theta = polar;
        else {
            float horizontal = (radius+target)*sin(polar);
            float vertical = (radius+target)*cos(polar)-(radius+g.frame.x);
            float range = length(vec2(horizontal, vertical));
            theta = atan(horizontal, vertical);
            geometric_impact = (radius+g.frame.x)*horizontal/max(range, 1e-20f)-radius;
            geometric_along = -(radius+g.frame.x)*vertical/max(range, 1e-20f);
        }
        geometric_theta = theta;
    }
    bool rising = target >= g.frame.x;
    bool outbound = rising;
    float tangent_zenith = 0.5f*REFRACTION_PI;
    if (!rising) {
        float target_q = target+refraction_profile(profile, g, target).x*(radius+target);
        float camera_q = g.frame.x+g.profile.z*(radius+g.frame.x);
        if (target_q < camera_q && target_q >= g.model.x && refraction_turn(rows, profile, g, target_q) <= target+0.25f*g.profile.x) {
            outbound = polar > refraction_table_travel(travel, rows, profile, g, target_q, g.frame.x);
            tangent_zenith += atan(sqrt(max((camera_q-target_q)*(2.0f*radius+camera_q+target_q), 0.0f)), radius+target_q);
        }
    }
    float floor_height = -REFRACTION_FAR;
    float lo = max(0.0f, theta-g.profile.w-0.02f), hi = min(REFRACTION_PI, theta+0.02f);
    if (!rising) lo = max(lo, tangent_zenith);
    int iterations = g.profile.w > 0.1f ? 26 : 18;
    float best_zenith = theta, best_error = REFRACTION_FAR;
    for (int i = 0; i < iterations; ++i) {
        float z = 0.5f*(lo+hi);
        vec2 reach = refraction_reach(travel, rows, profile, g, z, target, floor_height, outbound);
        float error = abs(reach.x-polar);
        if (reach.y > 0.5f && error < best_error) {
            best_zenith = z;
            best_error = error;
        }
        bool too_far = reach.y < 0.5f || reach.x > polar;
        if (outbound == too_far) hi = z;
        else lo = z;
    }
    float zenith = 0.5f*(lo+hi);
    vec2 solved = refraction_reach(travel, rows, profile, g, zenith, target, floor_height, outbound);
    if (best_error < REFRACTION_FAR && (solved.y < 0.5f || best_error < abs(solved.x-polar))) {
        zenith = best_zenith;
        solved = refraction_reach(travel, rows, profile, g, zenith, target, floor_height, outbound);
    }
    float solved_q = refraction_invariant(g, zenith);
    if (g.frame.w > 0.5f && zenith > 0.5f*REFRACTION_PI) {
        float camera_q = g.frame.x+g.profile.z*(radius+g.frame.x);
        float target_q = target >= REFRACTION_FAR ? camera_q : target+refraction_profile(profile, g, target).x*(radius+target);
        float span = max(0.02f, 2e-6f*(radius+g.frame.x));
        float q_lo = max(outbound ? g.model.x : -radius+1e-3f, solved_q-span);
        float q_hi = min(min(camera_q, target_q), solved_q+span);
        float best_q = solved_q, error_q = REFRACTION_FAR;
        for (int i = 0; i < 12; ++i) {
            float q = 0.5f*(q_lo+q_hi);
            vec2 at = refraction_reach_impact(travel, rows, profile, g, q, true, target, floor_height, outbound);
            float error = abs(at.x-polar);
            if (at.y > 0.5f && error < error_q) { best_q = q; error_q = error; }
            bool too_far = at.y < 0.5f || at.x > polar;
            if (outbound == too_far) q_lo = q;
            else q_hi = q;
        }
        if (error_q < REFRACTION_FAR) {
            solved_q = best_q;
            zenith = 0.5f*REFRACTION_PI+atan(sqrt(max((camera_q-solved_q)*(2.0f*radius+camera_q+solved_q), 0.0f)), radius+solved_q);
            solved = refraction_reach_impact(travel, rows, profile, g, solved_q, true, target, floor_height, outbound);
        }
    }
    if (g.profile.w > 0.1f && outbound && zenith > 0.5f*REFRACTION_PI &&
        (solved.y < 0.5f || abs(solved.x-polar) > 2e-4f*max(1.0f, polar))) {
        float camera_q = g.frame.x+g.profile.z*(radius+g.frame.x);
        float target_q = target >= REFRACTION_FAR ? camera_q : target+refraction_profile(profile, g, target).x*(radius+target);
        float q_lo = max(g.model.x, solved_q-0.02f), q_hi = min(min(camera_q, target_q), solved_q+0.02f);
        float best_q = solved_q, error_q = REFRACTION_FAR;
        for (int i = 0; i < 24; ++i) {
            float q = 0.5f*(q_lo+q_hi);
            vec2 at = refraction_reach_impact(travel, rows, profile, g, q, true, target, floor_height, true);
            float error = abs(at.x-polar);
            if (at.y > 0.5f && error < error_q) { best_q = q; error_q = error; }
            if (at.y < 0.5f || at.x < polar) q_hi = q;
            else q_lo = q;
        }
        if (error_q < abs(solved.x-polar) || solved.y < 0.5f) {
            solved_q = best_q;
            zenith = 0.5f*REFRACTION_PI+atan(sqrt(max((camera_q-solved_q)*(2.0f*radius+camera_q+solved_q), 0.0f)), radius+solved_q);
            solved = refraction_reach_impact(travel, rows, profile, g, solved_q, true, target, floor_height, true);
        }
    }
    vec2 reach = refraction_reach_impact(travel, rows, profile, g, solved_q, zenith > 0.5f*REFRACTION_PI, physical_target, 0.0f, outbound);
    float visible = reach.y > 0.5f && abs(reach.x-physical_polar) <= 2e-4f*max(1.0f, physical_polar) ? 1.0f : 0.0f;
    float rotation = g.frame.w > 0.5f ? refraction_exterior_rotation(g, solved_q, geometric_impact, geometric_along) : geometric_theta-zenith;
    if (distance < 0.02f*radius) {
        float weight = clamp((distance-0.005f*radius)/(0.015f*radius), 0.0f, 1.0f);
        weight = weight*weight*(3.0f-2.0f*weight);
        rotation = mix(refraction_born(profile, g, chord, distance), rotation, weight);
    }
    return vec2(clamp(rotation, -g.profile.w, g.profile.w), 2.0f*visible-1.0f);
}

vec2 refraction_observer_cell(sampler2D travel, sampler2D rows, sampler2D profile, RefractionGPU g, int column, int row) {
    float x = float(column)/float(REFRACTION_OBSERVER_WIDTH-1);
    float y = float(row)/float(REFRACTION_OBSERVER_HEIGHT-1);
    vec3 chord = refraction_chord(g, x);
    if (g.frame.w > 0.5f && chord.y >= g.frame.z) return vec2(0.0f, 1.0f);
    return refraction_solve(travel, rows, profile, g, chord, refraction_cell_distance(g, chord, y),
                            refraction_surface_distance(g) && row == REFRACTION_SURFACE_ROW);
}

float refraction_column(RefractionGPU g, float theta, float impact) {
    if (g.frame.w < 0.5f) {
        float horizon = g.mapping.x;
        if (theta <= horizon) return 0.5f*(1.0f-sqrt(max(1.0f-theta/horizon, 0.0f)));
        float below = theta-horizon, band = g.mapping.y;
        float u = below <= band ? 0.75f*below/band
                                : 0.75f+0.25f*sqrt(clamp((below-band)/max(REFRACTION_PI-horizon-band, 1e-6f), 0.0f, 1.0f));
        return 0.5f+0.5f*u;
    }
    float lower = g.escape.w*sinh(g.mapping.y);
    float fraction = refraction_deep_fraction(g);
    if (fraction > 0.0f && impact < lower) return fraction*(1.0f-sqrt(clamp((lower-impact)/max(g.frame.y+lower, 1e-6f), 0.0f, 1.0f)));
    float band = clamp((asinh(impact/g.escape.w)-g.mapping.y)/(g.mapping.z-g.mapping.y), 0.0f, 1.0f);
    return fraction+(1.0f-fraction)*band;
}

float refraction_row(RefractionGPU g, vec3 chord, float distance) {
    if (refraction_surface_distance(g)) {
        if (distance >= REFRACTION_FAR) return 1.0f;
        vec2 frame = refraction_distance_frame(g, chord);
        float delta = distance-frame.x;
        float u = delta/(abs(delta)+frame.y);
        float split = float(REFRACTION_SURFACE_ROW)/float(REFRACTION_OBSERVER_HEIGHT-1);
        return u < 0.0f ? split*(1.0f+u) : split+(1.0f-split)*u;
    }
    return distance >= REFRACTION_FAR ? 1.0f : distance/(distance+g.mapping.w);
}

bool refraction_unaffected(RefractionGPU g, float theta, float impact) {
    return g.frame.w > 0.5f && (theta <= 0.5f*REFRACTION_PI || impact >= g.frame.z);
}

vec2 refraction_observer_sample(sampler2D observer, RefractionGPU g, vec3 chord, float distance) {
    float x = refraction_column(g, chord.x, chord.y), y = refraction_row(g, chord, distance);
    float fx = x*float(REFRACTION_OBSERVER_WIDTH-1), fy = y*float(REFRACTION_OBSERVER_HEIGHT-1);
    int i = min(int(fx), REFRACTION_OBSERVER_WIDTH-2), j = min(int(fy), REFRACTION_OBSERVER_HEIGHT-2);
    float u = fx-float(i), v = fy-float(j);
    vec4 a = texelFetch(observer, ivec2(i, j), 0);
    vec4 b = texelFetch(observer, ivec2(i+1, j), 0);
    vec4 c = texelFetch(observer, ivec2(i, j+1), 0);
    vec4 d = texelFetch(observer, ivec2(i+1, j+1), 0);
    float wa = (1.0f-u)*(1.0f-v), wb = u*(1.0f-v), wc = (1.0f-u)*v, wd = u*v;
    float visible = wa*max(a.y, 0.0f)+wb*max(b.y, 0.0f)+wc*max(c.y, 0.0f)+wd*max(d.y, 0.0f);
    bool nearby = distance <= 0.005f*g.frame.y;
    float rotation = wa*a.x+wb*b.x+wc*c.x+wd*d.x;
    if (nearby) visible = refraction_short_visibility(g, chord, distance, rotation, 0.0f);
    return vec2(rotation, visible);
}

#endif
