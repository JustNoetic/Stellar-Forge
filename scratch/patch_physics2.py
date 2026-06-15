import re

def main():
    with open('d:/Files/Coding/OpenGL/Stellar-Forge/physics_core.py', 'r') as f:
        content = f.read()

    new_code = """
# ==============================================================================
# PURE PYTHON NUMBA IAS15 INTEGRATOR
# ==============================================================================

# IAS15 Constants
_IAS15_H = np.array([0.0, 0.0562625605369221464656521910318, 0.180240691736892364987579942780, 0.352624717113169637373907769648, 0.547153626330555383001448554766, 0.734210177215410531523210605558, 0.885320946839095768090359771030, 0.977520613561287501891174488626], dtype=np.float64)
_IAS15_RR = np.array([0.0562625605369221464656522, 0.1802406917368923649875799, 0.1239781311999702185219278, 0.3526247171131696373739078, 0.2963621565762474909082556, 0.1723840253762772723863278, 0.5471536263305553830014486, 0.4908910657936332365357964, 0.3669129345936630180138686, 0.1945289092173857456275408, 0.7342101772154105315232106, 0.6779476166784883850575584, 0.5539694854785181665356307, 0.3815854601022408941493028, 0.1870565508848551485217621, 0.8853209468390957680903598, 0.8290583863021736216247076, 0.7050802551022034031027798, 0.5326962297259261307164520, 0.3381673205085403850889112, 0.1511107696236852365671492, 0.9775206135612875018911745, 0.9212580530243653554255223, 0.7972799218243951369035945, 0.6248958964481178645172667, 0.4303669872307321188897259, 0.2433104363458769703679639, 0.0921996667221917338008147], dtype=np.float64)
_IAS15_C = np.array([-0.0562625605369221464656522, 0.0101408028300636299864818, -0.2365032522738145114532321, -0.0035758977292516175949345, 0.0935376952594620658957485, -0.5891279693869841488271399, 0.0019565654099472210769006, -0.0547553868890686864408084, 0.4158812000823068616886219, -1.1362815957175395318285885, -0.0014365302363708915424460, 0.0421585277212687077072973, -0.3600995965020568122897665, 1.2501507118406910258505441, -1.8704917729329500633517991, 0.0012717903090268677492943, -0.0387603579159067703699046, 0.3609622434528459832253398, -1.4668842084004269643701553, 2.9061362593084293014237913, -2.7558127197720458314421588], dtype=np.float64)
_IAS15_D = np.array([0.0562625605369221464656522, 0.0031654757181708292499905, 0.2365032522738145114532321, 0.0001780977692217433881125, 0.0457929855060279188954539, 0.5891279693869841488271399, 0.0000100202365223291272096, 0.0084318571535257015445000, 0.2535340690545692665214616, 1.1362815957175395318285885, 0.0000005637641639318207610, 0.0015297840025004658189490, 0.0978342365324440053653648, 0.8752546646840910912297246, 1.8704917729329500633517991, 0.0000000317188154017613665, 0.0002762930909826476593130, 0.0360285539837364596003871, 0.5767330002770787313544596, 2.2485887607691597933926895, 2.7558127197720458314421588], dtype=np.float64)

@njit(cache=True)
def ias15_sqrt7(a):
    scale = 1.0
    while a < 1e-7 and a > 0.0:
        scale *= 0.1
        a *= 1e7
    while a > 1e2:
        scale *= 10.0
        a *= 1e-7
    x = 1.0
    for _ in range(20):
        x6 = x*x*x*x*x*x
        x += (a/x6 - x) / 7.0
    return x * scale

@njit(cache=True)
def _ias15_add_cs(val, cs_val, inp):
    y = inp - cs_val
    t = val + y
    new_cs = (t - val) - y
    return t, new_cs

@njit(cache=True)
def _ias15_predict_next_step(ratio, N3, _e, _b, e, b):
    if ratio > 20.0:
        for k in range(N3):
            for i in range(7):
                e[i, k] = 0.0
                b[i, k] = 0.0
    else:
        q1 = ratio
        q2 = q1 * q1
        q3 = q1 * q2
        q4 = q2 * q2
        q5 = q2 * q3
        q6 = q3 * q3
        q7 = q3 * q4

        for k in range(N3):
            be0 = _b[0, k] - _e[0, k]
            be1 = _b[1, k] - _e[1, k]
            be2 = _b[2, k] - _e[2, k]
            be3 = _b[3, k] - _e[3, k]
            be4 = _b[4, k] - _e[4, k]
            be5 = _b[5, k] - _e[5, k]
            be6 = _b[6, k] - _e[6, k]

            e[0, k] = q1*(_b[6, k]* 7.0 + _b[5, k]* 6.0 + _b[4, k]* 5.0 + _b[3, k]* 4.0 + _b[2, k]* 3.0 + _b[1, k]*2.0 + _b[0, k])
            e[1, k] = q2*(_b[6, k]*21.0 + _b[5, k]*15.0 + _b[4, k]*10.0 + _b[3, k]* 6.0 + _b[2, k]* 3.0 + _b[1, k])
            e[2, k] = q3*(_b[6, k]*35.0 + _b[5, k]*20.0 + _b[4, k]*10.0 + _b[3, k]* 4.0 + _b[2, k])
            e[3, k] = q4*(_b[6, k]*35.0 + _b[5, k]*15.0 + _b[4, k]* 5.0 + _b[3, k])
            e[4, k] = q5*(_b[6, k]*21.0 + _b[5, k]* 6.0 + _b[4, k])
            e[5, k] = q6*(_b[6, k]* 7.0 + _b[5, k])
            e[6, k] = q7* _b[6, k]

            b[0, k] = e[0, k] + be0
            b[1, k] = e[1, k] + be1
            b[2, k] = e[2, k] + be2
            b[3, k] = e[3, k] + be3
            b[4, k] = e[4, k] + be4
            b[5, k] = e[5, k] + be5
            b[6, k] = e[6, k] + be6

@njit(cache=True)
def ias15_step_numba(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                     oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles,
                     dt, min_dt, epsilon, dt_last_done,
                     g, e_arr, b, csb, er, br, at, x0, v0, a0, csx, csv,
                     h_const, rr_const, c_const, d_const):
                     
    N3 = 3 * num_bodies
    safety_factor = 0.25
    
    compute_all_accelerations(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                              oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles)

    for k in range(num_bodies):
        x0[3*k+0] = arr[k, 0]
        x0[3*k+1] = arr[k, 1]
        x0[3*k+2] = arr[k, 2]
        v0[3*k+0] = arr[k, 3]
        v0[3*k+1] = arr[k, 4]
        v0[3*k+2] = arr[k, 5]
        a0[3*k+0] = arr[k, 6]
        a0[3*k+1] = arr[k, 7]
        a0[3*k+2] = arr[k, 8]

    for k in range(N3):
        for i in range(7):
            csb[i, k] = 0.0

    for k in range(N3):
        g[0, k] = b[6, k]*d_const[15] + b[5, k]*d_const[10] + b[4, k]*d_const[6] + b[3, k]*d_const[3]  + b[2, k]*d_const[1]  + b[1, k]*d_const[0]  + b[0, k]
        g[1, k] = b[6, k]*d_const[16] + b[5, k]*d_const[11] + b[4, k]*d_const[7] + b[3, k]*d_const[4]  + b[2, k]*d_const[2]  + b[1, k]
        g[2, k] = b[6, k]*d_const[17] + b[5, k]*d_const[12] + b[4, k]*d_const[8] + b[3, k]*d_const[5]  + b[2, k]
        g[3, k] = b[6, k]*d_const[18] + b[5, k]*d_const[13] + b[4, k]*d_const[9] + b[3, k]
        g[4, k] = b[6, k]*d_const[19] + b[5, k]*d_const[14] + b[4, k]
        g[5, k] = b[6, k]*d_const[20] + b[5, k]
        g[6, k] = b[6, k]

    predictor_corrector_error = 1e300
    predictor_corrector_error_last = 2.0
    iterations = 0
    
    dt_new = dt

    while True:
        if predictor_corrector_error < 1e-16:
            break
        if iterations > 2 and predictor_corrector_error_last <= predictor_corrector_error:
            break
        if iterations >= 12:
            break
            
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0
        iterations += 1

        for n in range(1, 8):
            dt_h = dt * h_const[n]
            
            for i in range(num_bodies):
                k0, k1, k2 = 3*i, 3*i+1, 3*i+2
                
                xk0 = -csx[k0] + ((((((((b[6, k0]*7.*h_const[n]/9. + b[5, k0])*3.*h_const[n]/4. + b[4, k0])*5.*h_const[n]/7. + b[3, k0])*2.*h_const[n]/3. + b[2, k0])*3.*h_const[n]/5. + b[1, k0])*h_const[n]/2. + b[0, k0])*h_const[n]/3. + a0[k0])*dt_h/2. + v0[k0])*dt_h
                xk1 = -csx[k1] + ((((((((b[6, k1]*7.*h_const[n]/9. + b[5, k1])*3.*h_const[n]/4. + b[4, k1])*5.*h_const[n]/7. + b[3, k1])*2.*h_const[n]/3. + b[2, k1])*3.*h_const[n]/5. + b[1, k1])*h_const[n]/2. + b[0, k1])*h_const[n]/3. + a0[k1])*dt_h/2. + v0[k1])*dt_h
                xk2 = -csx[k2] + ((((((((b[6, k2]*7.*h_const[n]/9. + b[5, k2])*3.*h_const[n]/4. + b[4, k2])*5.*h_const[n]/7. + b[3, k2])*2.*h_const[n]/3. + b[2, k2])*3.*h_const[n]/5. + b[1, k2])*h_const[n]/2. + b[0, k2])*h_const[n]/3. + a0[k2])*dt_h/2. + v0[k2])*dt_h
                
                arr[i, 0] = xk0 + x0[k0]
                arr[i, 1] = xk1 + x0[k1]
                arr[i, 2] = xk2 + x0[k2]

                vk0 =  -csv[k0] + (((((((b[6, k0]*7.*h_const[n]/8. + b[5, k0])*6.*h_const[n]/7. + b[4, k0])*5.*h_const[n]/6. + b[3, k0])*4.*h_const[n]/5. + b[2, k0])*3.*h_const[n]/4. + b[1, k0])*2.*h_const[n]/3. + b[0, k0])*h_const[n]/2. + a0[k0])*dt_h
                vk1 =  -csv[k1] + (((((((b[6, k1]*7.*h_const[n]/8. + b[5, k1])*6.*h_const[n]/7. + b[4, k1])*5.*h_const[n]/6. + b[3, k1])*4.*h_const[n]/5. + b[2, k1])*3.*h_const[n]/4. + b[1, k1])*2.*h_const[n]/3. + b[0, k1])*h_const[n]/2. + a0[k1])*dt_h
                vk2 =  -csv[k2] + (((((((b[6, k2]*7.*h_const[n]/8. + b[5, k2])*6.*h_const[n]/7. + b[4, k2])*5.*h_const[n]/6. + b[3, k2])*4.*h_const[n]/5. + b[2, k2])*3.*h_const[n]/4. + b[1, k2])*2.*h_const[n]/3. + b[0, k2])*h_const[n]/2. + a0[k2])*dt_h
                
                arr[i, 3] = vk0 + v0[k0]
                arr[i, 4] = vk1 + v0[k1]
                arr[i, 5] = vk2 + v0[k2]

            compute_all_accelerations(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                                      oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles)

            for i in range(num_bodies):
                at[3*i+0] = arr[i, 6]
                at[3*i+1] = arr[i, 7]
                at[3*i+2] = arr[i, 8]

            if n == 1:
                for k in range(N3):
                    tmp = g[0, k]
                    g[0, k] = (at[k] - a0[k]) / rr_const[0]
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], g[0, k] - tmp)
            elif n == 2:
                for k in range(N3):
                    tmp = g[1, k]
                    g[1, k] = ((at[k] - a0[k])/rr_const[1] - g[0, k])/rr_const[2]
                    tmp = g[1, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[0])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp)
            elif n == 3:
                for k in range(N3):
                    tmp = g[2, k]
                    g[2, k] = (((at[k] - a0[k])/rr_const[3] - g[0, k])/rr_const[4] - g[1, k])/rr_const[5]
                    tmp = g[2, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[1])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[2])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp)
            elif n == 4:
                for k in range(N3):
                    tmp = g[3, k]
                    g[3, k] = ((((at[k] - a0[k])/rr_const[6] - g[0, k])/rr_const[7] - g[1, k])/rr_const[8] - g[2, k])/rr_const[9]
                    tmp = g[3, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[3])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[4])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp * c_const[5])
                    b[3, k], csb[3, k] = _ias15_add_cs(b[3, k], csb[3, k], tmp)
            elif n == 5:
                for k in range(N3):
                    tmp = g[4, k]
                    g[4, k] = (((((at[k] - a0[k])/rr_const[10] - g[0, k])/rr_const[11] - g[1, k])/rr_const[12] - g[2, k])/rr_const[13] - g[3, k])/rr_const[14]
                    tmp = g[4, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[6])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[7])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp * c_const[8])
                    b[3, k], csb[3, k] = _ias15_add_cs(b[3, k], csb[3, k], tmp * c_const[9])
                    b[4, k], csb[4, k] = _ias15_add_cs(b[4, k], csb[4, k], tmp)
            elif n == 6:
                for k in range(N3):
                    tmp = g[5, k]
                    g[5, k] = ((((((at[k] - a0[k])/rr_const[15] - g[0, k])/rr_const[16] - g[1, k])/rr_const[17] - g[2, k])/rr_const[18] - g[3, k])/rr_const[19] - g[4, k])/rr_const[20]
                    tmp = g[5, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[10])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[11])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp * c_const[12])
                    b[3, k], csb[3, k] = _ias15_add_cs(b[3, k], csb[3, k], tmp * c_const[13])
                    b[4, k], csb[4, k] = _ias15_add_cs(b[4, k], csb[4, k], tmp * c_const[14])
                    b[5, k], csb[5, k] = _ias15_add_cs(b[5, k], csb[5, k], tmp)
            elif n == 7:
                maxak = 0.0
                maxb6ktmp = 0.0
                for k in range(N3):
                    tmp = g[6, k]
                    g[6, k] = (((((((at[k] - a0[k])/rr_const[21] - g[0, k])/rr_const[22] - g[1, k])/rr_const[23] - g[2, k])/rr_const[24] - g[3, k])/rr_const[25] - g[4, k])/rr_const[26] - g[5, k])/rr_const[27]
                    tmp = g[6, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[15])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[16])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp * c_const[17])
                    b[3, k], csb[3, k] = _ias15_add_cs(b[3, k], csb[3, k], tmp * c_const[18])
                    b[4, k], csb[4, k] = _ias15_add_cs(b[4, k], csb[4, k], tmp * c_const[19])
                    b[5, k], csb[5, k] = _ias15_add_cs(b[5, k], csb[5, k], tmp * c_const[20])
                    b[6, k], csb[6, k] = _ias15_add_cs(b[6, k], csb[6, k], tmp)

                    ak = abs(at[k])
                    if ak > maxak: maxak = ak
                    b6ktmp = abs(tmp)
                    if b6ktmp > maxb6ktmp: maxb6ktmp = b6ktmp
                    
                if maxak > 0.0:
                    predictor_corrector_error = maxb6ktmp / maxak
                else:
                    predictor_corrector_error = 0.0

        # End for Gauss-Radau nodes
    # End Predictor-corrector loop
    
    # Calculate dt_new
    min_timescale2 = 1e300
    for i in range(num_bodies):
        a0i = 0.0
        y2 = 0.0
        y3 = 0.0
        y4 = 0.0
        for k in range(3*i, 3*i+3):
            a0i += a0[k]*a0[k]
            tmp2 = a0[k] + b[0, k] + b[1, k] + b[2, k] + b[3, k] + b[4, k] + b[5, k] + b[6, k]
            y2 += tmp2*tmp2
            tmp3 = b[0, k] + 2.*b[1, k] + 3.*b[2, k] + 4.*b[3, k] + 5.*b[4, k] + 6.*b[5, k] + 7.*b[6, k]
            y3 += tmp3*tmp3
            tmp4 = 2.*b[1, k] + 6.*b[2, k] + 12.*b[3, k] + 20.*b[4, k] + 30.*b[5, k] + 42.*b[6, k]
            y4 += tmp4*tmp4
            
        if a0i == 0.0: continue
        
        timescale2 = 2.0 * y2 / (y3 + math.sqrt(y4 * y2))
        if timescale2 < min_timescale2:
            min_timescale2 = timescale2
            
    if min_timescale2 < 1e299:
        dt_new = math.sqrt(min_timescale2) * dt * ias15_sqrt7(epsilon * 5040.0)
    else:
        dt_new = dt / safety_factor

    if abs(dt_new) < min_dt:
        dt_new = math.copysign(min_dt, dt_new)
        
    if abs(dt_new / dt) < safety_factor:
        for i in range(num_bodies):
            k0, k1, k2 = 3*i, 3*i+1, 3*i+2
            arr[i, 0] = x0[k0]
            arr[i, 1] = x0[k1]
            arr[i, 2] = x0[k2]
            arr[i, 3] = v0[k0]
            arr[i, 4] = v0[k1]
            arr[i, 5] = v0[k2]
            arr[i, 6] = a0[k0]
            arr[i, 7] = a0[k1]
            arr[i, 8] = a0[k2]
            
        if dt_last_done != 0.0:
            ratio = dt_new / dt_last_done
            _ias15_predict_next_step(ratio, N3, er, br, e_arr, b)
            
        return dt_new, 0.0, False

    if abs(dt_new / dt) > 1.0:
        if dt_new / dt > 1.0 / safety_factor:
            dt_new = dt / safety_factor

    for k in range(N3):
        dt2 = dt * dt
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[6, k]/72. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[5, k]/56. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[4, k]/42. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[3, k]/30. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[2, k]/20. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[1, k]/12. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[0, k]/6. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], a0[k]/2. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], v0[k] * dt)

        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[6, k]/8. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[5, k]/7. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[4, k]/6. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[3, k]/5. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[2, k]/4. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[1, k]/3. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[0, k]/2. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], a0[k] * dt)

    for i in range(num_bodies):
        arr[i, 0] = x0[3*i+0]
        arr[i, 1] = x0[3*i+1]
        arr[i, 2] = x0[3*i+2]
        arr[i, 3] = v0[3*i+0]
        arr[i, 4] = v0[3*i+1]
        arr[i, 5] = v0[3*i+2]
        
    for i in range(7 * N3):
        er.flat[i] = e_arr.flat[i]
        br.flat[i] = b.flat[i]
        
    ratio = dt_new / dt
    _ias15_predict_next_step(ratio, N3, e_arr, b, e_arr, b)
    
    return dt_new, dt, True


class Particle:
    def __init__(self, sim, idx):
        self.sim = sim
        self.idx = idx
        
    @property
    def x(self): return self.sim.arr[self.idx, 0]
    @x.setter
    def x(self, val): self.sim.arr[self.idx, 0] = val
    
    @property
    def y(self): return self.sim.arr[self.idx, 1]
    @y.setter
    def y(self, val): self.sim.arr[self.idx, 1] = val
    
    @property
    def z(self): return self.sim.arr[self.idx, 2]
    @z.setter
    def z(self, val): self.sim.arr[self.idx, 2] = val
    
    @property
    def vx(self): return self.sim.arr[self.idx, 3]
    @vx.setter
    def vx(self, val): self.sim.arr[self.idx, 3] = val
    
    @property
    def vy(self): return self.sim.arr[self.idx, 4]
    @vy.setter
    def vy(self, val): self.sim.arr[self.idx, 4] = val
    
    @property
    def vz(self): return self.sim.arr[self.idx, 5]
    @vz.setter
    def vz(self, val): self.sim.arr[self.idx, 5] = val
    
    @property
    def m(self): return self.sim.arr[self.idx, 9]
    @m.setter
    def m(self, val): self.sim.arr[self.idx, 9] = val
    
    @property
    def hash(self): return self.sim.hashes[self.idx]

class ParticleList:
    def __init__(self, sim):
        self.sim = sim
    def __getitem__(self, key):
        if isinstance(key, str):
            for i, h in enumerate(self.sim.hashes):
                if h == key: return Particle(self.sim, i)
            raise KeyError(f"Particle {key} not found")
        elif isinstance(key, int):
            if key < 0 or key >= self.sim.N: raise IndexError()
            return Particle(self.sim, key)
        raise TypeError("Invalid key type")
    def __len__(self):
        return self.sim.N

class Simulation:
    def __init__(self):
        self.G = 1.0
        self.t = 0.0
        self.dt = 1e-4
        self.dt_last_done = 0.0
        self.N = 0
        self.arr = np.zeros((0, 10), dtype=np.float64)
        self.hashes = []
        self.particles = ParticleList(self)
        self.has_j2 = False
        self.has_gr = False
        self.phys_star_idx = -1
        self.oblate_indices = np.empty(0, dtype=np.int32)
        self.oblate_j2 = np.empty(0, dtype=np.float64)
        self.oblate_j4 = np.empty(0, dtype=np.float64)
        self.oblate_req = np.empty(0, dtype=np.float64)
        self.oblate_poles = np.empty((0, 3), dtype=np.float64)
        
        self._g = np.zeros((7, 0), dtype=np.float64)
        self._e = np.zeros((7, 0), dtype=np.float64)
        self._b = np.zeros((7, 0), dtype=np.float64)
        self._csb = np.zeros((7, 0), dtype=np.float64)
        self._er = np.zeros((7, 0), dtype=np.float64)
        self._br = np.zeros((7, 0), dtype=np.float64)
        self._at = np.zeros(0, dtype=np.float64)
        self._x0 = np.zeros(0, dtype=np.float64)
        self._v0 = np.zeros(0, dtype=np.float64)
        self._a0 = np.zeros(0, dtype=np.float64)
        self._csx = np.zeros(0, dtype=np.float64)
        self._csv = np.zeros(0, dtype=np.float64)

    def _resize_buffers(self, n):
        n3 = n * 3
        if n3 > len(self._at):
            self._g = np.zeros((7, n3), dtype=np.float64)
            self._e = np.zeros((7, n3), dtype=np.float64)
            self._b = np.zeros((7, n3), dtype=np.float64)
            self._csb = np.zeros((7, n3), dtype=np.float64)
            self._er = np.zeros((7, n3), dtype=np.float64)
            self._br = np.zeros((7, n3), dtype=np.float64)
            self._at = np.zeros(n3, dtype=np.float64)
            self._x0 = np.zeros(n3, dtype=np.float64)
            self._v0 = np.zeros(n3, dtype=np.float64)
            self._a0 = np.zeros(n3, dtype=np.float64)
            self._csx = np.zeros(n3, dtype=np.float64)
            self._csv = np.zeros(n3, dtype=np.float64)

    def add(self, m=0.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, hash=None, primary=None, a=0.0, e=0.0, inc=0.0, Omega=0.0, omega=0.0, M=0.0):
        if primary is not None:
            mu = self.G * (primary.m + m)
            pos, vel = orbital_to_cartesian(a, e, inc, Omega, omega, M, mu)
            x = primary.x + pos[0]
            y = primary.y + pos[1]
            z = primary.z + pos[2]
            vx = primary.vx + vel[0]
            vy = primary.vy + vel[1]
            vz = primary.vz + vel[2]
            
        new_row = np.array([[x, y, z, vx, vy, vz, 0.0, 0.0, 0.0, m]], dtype=np.float64)
        self.arr = np.vstack([self.arr, new_row])
        self.hashes.append(hash)
        self.N += 1
        self._resize_buffers(self.N)

    def remove(self, index=None, hash=None):
        if hash is not None:
            for i, h in enumerate(self.hashes):
                if h == hash:
                    index = i
                    break
        if index is not None and 0 <= index < self.N:
            self.arr = np.delete(self.arr, index, axis=0)
            self.hashes.pop(index)
            self.N -= 1
            
            # Note: We must reset integrator state if a particle is removed or added
            # because the history buffers `b`, `e` will be invalidated.
            self._resize_buffers(0) # clear it
            self._resize_buffers(self.N)
            self.dt_last_done = 0.0
            
    def move_to_com(self):
        if self.N == 0: return
        M_tot = np.sum(self.arr[:, 9])
        if M_tot == 0: return
        
        com_x = np.sum(self.arr[:, 0] * self.arr[:, 9]) / M_tot
        com_y = np.sum(self.arr[:, 1] * self.arr[:, 9]) / M_tot
        com_z = np.sum(self.arr[:, 2] * self.arr[:, 9]) / M_tot
        com_vx = np.sum(self.arr[:, 3] * self.arr[:, 9]) / M_tot
        com_vy = np.sum(self.arr[:, 4] * self.arr[:, 9]) / M_tot
        com_vz = np.sum(self.arr[:, 5] * self.arr[:, 9]) / M_tot
        
        self.arr[:, 0] -= com_x
        self.arr[:, 1] -= com_y
        self.arr[:, 2] -= com_z
        self.arr[:, 3] -= com_vx
        self.arr[:, 4] -= com_vy
        self.arr[:, 5] -= com_vz

    def copy(self):
        new_sim = Simulation()
        new_sim.G = self.G
        new_sim.t = self.t
        new_sim.dt = self.dt
        new_sim.dt_last_done = self.dt_last_done
        new_sim.N = self.N
        new_sim.arr = self.arr.copy()
        new_sim.hashes = self.hashes.copy()
        new_sim.has_j2 = self.has_j2
        new_sim.has_gr = self.has_gr
        new_sim.phys_star_idx = self.phys_star_idx
        new_sim.oblate_indices = self.oblate_indices.copy()
        new_sim.oblate_j2 = self.oblate_j2.copy()
        new_sim.oblate_j4 = self.oblate_j4.copy()
        new_sim.oblate_req = self.oblate_req.copy()
        new_sim.oblate_poles = self.oblate_poles.copy()
        
        new_sim._resize_buffers(self.N)
        new_sim._g = self._g.copy()
        new_sim._e = self._e.copy()
        new_sim._b = self._b.copy()
        new_sim._csb = self._csb.copy()
        new_sim._er = self._er.copy()
        new_sim._br = self._br.copy()
        new_sim._at = self._at.copy()
        new_sim._x0 = self._x0.copy()
        new_sim._v0 = self._v0.copy()
        new_sim._a0 = self._a0.copy()
        new_sim._csx = self._csx.copy()
        new_sim._csv = self._csv.copy()
        return new_sim

    def integrate(self, t_target):
        c2 = C_AU_YR * C_AU_YR
        while True:
            t_diff = t_target - self.t
            if abs(t_diff) < 1e-12:
                break
                
            dt_step = self.dt
            if t_diff > 0 and dt_step > t_diff:
                dt_step = t_diff
            elif t_diff < 0 and dt_step < t_diff:
                dt_step = t_diff
                
            self.dt, dt_done, success = ias15_step_numba(
                self.arr, self.N, self.G, c2, self.has_j2, self.has_gr, self.phys_star_idx,
                self.oblate_indices, self.oblate_j2, self.oblate_j4, self.oblate_req, self.oblate_poles,
                dt_step, 0.0, 1e-9, self.dt_last_done,
                self._g, self._e, self._b, self._csb, self._er, self._br,
                self._at, self._x0, self._v0, self._a0, self._csx, self._csv,
                _IAS15_H, _IAS15_RR, _IAS15_C, _IAS15_D
            )
            
            if success:
                self.t += dt_done
                self.dt_last_done = dt_done
"""
    content += new_code
    with open('d:/Files/Coding/OpenGL/Stellar-Forge/physics_core.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    main()
