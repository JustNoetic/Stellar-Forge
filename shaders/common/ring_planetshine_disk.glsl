const vec4 RING_DISK_R2 = vec4(0.0694318442,0.3300094782,0.6699905218,0.9305681558);
const vec4 RING_DISK_W = vec4(0.1739274226,0.3260725774,0.3260725774,0.1739274226);
const int RING_DISK_SAMPLES = 64;
vec4 ring_disk_sample(vec3 center, vec3 pole, float sin_radius, int sample_index) {
    vec3 vertical = pole-center*dot(pole,center);
    if (dot(vertical,vertical)<1e-8) {
        vec3 axis = abs(center.x)<0.8 ? vec3(1,0,0) : vec3(0,0,1);
        vertical = axis-center*dot(axis,center);
    }
    vertical = normalize(vertical);
    vec3 horizontal = cross(center,vertical);
    int row = sample_index/16;
    float azimuth = (float(sample_index%16)+0.5)*(2.0*3.141592653589793/16.0);
    float r = min(sin_radius,0.999)*sqrt(RING_DISK_R2[row]);
    float cosine = sqrt(1.0-r*r);
    vec3 direction = center*cosine+r*(vertical*sin(azimuth)+horizontal*cos(azimuth));
    return vec4(direction,RING_DISK_W[row]/(16.0*cosine));
}

vec4 ring_host_disk_sample(uint host_index, vec3 hit_local, vec3 pole, int q, out vec3 incident) {
    uint body_index = ring_hosts[host_index].counts_indices.x;
    BodyInstanceGPU host = bodies[body_index];
    float radius = host.pos_radius.w, distance = length(hit_local);
    incident = vec3(0.0);
    if (distance <= radius) return vec4(0.0);
    float ratio = clamp(radius/distance, 0.0, 0.999);
    vec4 sample_ray = ring_disk_sample(-hit_local/distance, pole, ratio, q);
    vec3 D = sample_ray.xyz;
    float polar = max(1.0-clamp(host.pole_oblateness.w, 0.0, 0.95), 0.05);
    float eta = 1.0/(polar*polar), oq = dot(hit_local, pole), dq = dot(D, pole);
    float c = dot(hit_local, hit_local)+(eta-1.0)*oq*oq-radius*radius;
    float a = 1.0+(eta-1.0)*dq*dq, b = dot(hit_local, D)+(eta-1.0)*oq*dq;
    float discriminant = b*b-a*c;
    if (discriminant <= 0.0) return sample_ray;
    float along = (-b-sqrt(discriminant))/a;
    if (along <= 0.0) return sample_ray;
    vec3 Q = hit_local+along*D;
    vec3 normal = normalize(Q+pole*((eta-1.0)*dot(Q, pole)));
    vec3 host_center = ring_hosts[host_index].center_min_r.xyz;
    for (uint s = 0u; s < u_scene.num_stars; ++s) {
        vec3 delta = (u_scene.stars_pos_radius[s].xyz-host_center)-Q;
        float star_distance = max(length(delta), 1e-12);
        vec3 L = delta/star_distance;
        float response = surface_direct_response(normal, -D, L, u_scene.stars_pos_radius[s].w/star_distance,
            max(dot(normal, L), 0.0), host.material);
        float flux = u_scene.hdr_enabled != 0u ? u_scene.stars_color_lum[s].w/(star_distance*star_distance) : 1.0;
        incident += clamp(host.color_is_star.rgb, 0.0, 1.0)*u_scene.stars_color_lum[s].rgb*(flux*response);
    }
    incident *= ratio*ratio*sample_ray.w;
    return sample_ray;
}
