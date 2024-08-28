import numpy as np

def createDOL(filename, DOL):
        dolfile = open(filename + '.dol','w+')
        for i in range(len(DOL)):
               dolfile.write(str(i)+'\t'+str(DOL[i][0])+'\t'+str(DOL[i][1])+'\t'+str(DOL[i][2])+'\t'+str(DOL[i][3])+'\t'+str(DOL[i][4])+'\t'+str(DOL[i][5])+'\n')      
        dolfile.close()
        print('DOL file ' + filename + ' has been created successfully!')

def MoveToGC(Array):
	x_tot = 0
	y_tot = 0
	z_tot = 0
	for i in range(0, len(Array)):
		x_tot += Array[i][0]
		y_tot += Array[i][1]
		z_tot += Array[i][2]
	x_tot = x_tot / len(Array)
	y_tot = y_tot / len(Array)
	z_tot = z_tot / len(Array)
	for i in range(0, len(Array)):
		Array[i][0] = Array[i][0] - x_tot
		Array[i][1] = Array[i][1] - y_tot
		Array[i][2] = Array[i][2] - z_tot
	return Array

def calcLen(t, c, p, R):
    term1 = ((p**2/(4*np.pi**2) + c**2 + R**2) * np.arcsinh((2*c**2*t - 2*R*c) / 
            np.sqrt(4*c**2 * (p**2/(4*np.pi**2) + c**2 + R**2) - 4*R**2*c**2))) / (2*c)
    term2 = (R**2 * np.arcsinh((2*c**2*t - 2*R*c) / 
            np.sqrt(4*c**2 * (p**2/(4*np.pi**2) + c**2 + R**2) - 4*R**2*c**2))) / (2*c)
    term3 = t * np.sqrt(c**2*t**2 - 2*R*c*t + p**2/(4*np.pi**2) + c**2 + R**2) / 2
    term4 = R * np.sqrt(c**2*t**2 - 2*R*c*t + p**2/(4*np.pi**2) + c**2 + R**2) / (2*c)
    return term1 - term2 + term3 - term4

def find_t(t_max, c, p, R_start, R_final,L_pitch, eps=1e-6):
        t_values = [0]
        dt = R_final/L_pitch
        L_max = calcLen(t_max,c,p,R_start) - calcLen(t_values[0],c,p,R_start) + L_pitch
        L_t = calcLen(t_values[0],c,p,R_start) + L_pitch
        while L_max - L_t > L_pitch:
                t_low = t_values[-1]
                t_high = t_low + dt
                while t_high - t_low > eps:
                        t_mid = (t_low + t_high) / 2
                        L_mid = calcLen(t_mid, c, p, R_start) - calcLen(t_values[-1], c, p, R_start)
                        if L_mid > L_pitch:
                                t_high = t_mid
                        else:
                                t_low = t_mid
                        next_t = t_low
                if next_t - t_values[-1] < eps:
                        break
                t_values.append(next_t)
                L_t = calcLen(t_values[-1],c,p,R_start) - calcLen(t_values[0],c,p,R_start) + L_pitch
        return np.array(t_values)  

# =================
# Structures
# =================

def MT2dol(radius=27, pitch=5.1, unitsPerPitch=20, unitsInPitch=20, startAt=0, discreteHeight=10, numHelixStarts=1, superHelicalPitch=0, RingTwistAlpha=0, RingTwistBeta=0, filename=''):
        longitudinalSpacing = (pitch * 2.0 / numHelixStarts)
        angle = 2.0 * np.pi / unitsPerPitch
        if superHelicalPitch > 1e-5:
                angleShift = (2.0 * np.pi * longitudinalSpacing) / (superHelicalPitch * unitsPerPitch)
        else:
                angleShift = 0.0
        res = []
        n = 1
        m = 1
        for i in range(discreteHeight):
                initialLayerShift = np.fmod(i * (2 / numHelixStarts) * angleShift * unitsPerPitch, 2 * np.pi)
                hUnitsPerPitch = 2.0 * np.pi / (angle + angleShift * unitsPerPitch)
                initialZShift = i * longitudinalSpacing
                for j in range(startAt, unitsInPitch):
                        
                        theta = initialLayerShift + j * (angle + angleShift)
                        x = radius * np.cos(theta) 
                        y = radius * np.sin(theta)  
                        z = initialZShift + (j / unitsPerPitch) * pitch
                        alphaBetaFirst = (j * RingTwistAlpha) / 180 * np.pi
                        betaBetaFirst = j * RingTwistBeta / 180 * np.pi
                        gammaBetaFirst = (90 - 180 * theta / np.pi) / 180 * np.pi
                        eps = 1e-5
                        if 1 - abs(np.cos(gammaBetaFirst) * np.sin(betaBetaFirst)) > eps:
                               beta = np.arcsin(np.cos(gammaBetaFirst) * np.sin(betaBetaFirst))
                               alpha = np.arctan2(-(-np.cos(betaBetaFirst) * np.sin(alphaBetaFirst) + np.cos(alphaBetaFirst) * np.sin(betaBetaFirst) * np.sin(gammaBetaFirst)) / np.cos(beta),(np.cos(alphaBetaFirst) * np.cos(betaBetaFirst) + np.sin(alphaBetaFirst) * np.sin(betaBetaFirst) * np.sin(gammaBetaFirst)) / np.cos(beta))
                               gamma = np.arctan2(-(-np.sin(gammaBetaFirst)) / np.cos(beta), (np.cos(betaBetaFirst) * np.cos(gammaBetaFirst)) / np.cos(beta))
                        else:
                               gamma = 0.
                               if 1 - np.cos(gammaBetaFirst) * np.sin(betaBetaFirst) < eps:
                                      beta = np.pi/2
                                      alpha = gamma + np.arctan2(np.sin(alphaBetaFirst) * np.sin(betaBetaFirst) + np.cos(alphaBetaFirst) * np.cos(betaBetaFirst) * np.sin(gammaBetaFirst),-(-np.cos(alphaBetaFirst) * np.sin(betaBetaFirst) + np.cos(betaBetaFirst) * np.sin(alphaBetaFirst) * np.sin(gammaBetaFirst)))
                               else:
                                      beta = -np.pi/2
                                      alpha = -gamma + np.arctan2(- (np.sin(alphaBetaFirst) * np.sin(betaBetaFirst) + np.cos(alphaBetaFirst) * np.cos(betaBetaFirst) * np.sin(gammaBetaFirst)),(-np.cos(alphaBetaFirst) * np.sin(betaBetaFirst) + np.cos(betaBetaFirst) * np.sin(alphaBetaFirst) * np.sin(gammaBetaFirst)))
                        alpha = alpha*180/np.pi
                        beta = beta*180/np.pi
                        gamma = gamma*180/np.pi
                        res.append([x, y, z, alpha, beta, gamma])
                DOL = MoveToGC(res)
        if filename:
                createDOL(filename,DOL)
        return DOL

def helix2dol(R, L_dim, T_dim, N_p, h, pitchPerUnit=0, pitchPerSlab=0, startAt=0, filename=''):
        unitPitch = pitchPerUnit + T_dim
        p = N_p * unitPitch + pitchPerSlab
        L_turn = np.sqrt((2*np.pi * R)**2 + p**2)
        N_turn = int(L_turn/L_dim)
        N_ring = int((h-unitPitch*N_p)/(p)) # Number of slabs
        angle = 2 * np.pi / N_turn
        
        c = 0
        N_remain = int((h - p*N_ring - unitPitch*N_p)/(p/N_turn))
        DOL = np.zeros([N_turn*N_ring*N_p + N_remain*N_p, 6])
        for helix in range(N_p):
                for k in range(N_ring + 1):
                        z_start = startAt + unitPitch * helix + p * k
                        j_max = N_turn if k < N_ring else N_remain
                        for j in range(j_max):
                                theta = j * angle
                                alpha = np.arctan(p/(2*np.pi*R)) * 180 / np.pi
                                beta = 0
                                gamma = theta * 180 / np.pi
                                z = z_start + j*p/N_turn
                                x = R * np.cos(theta)
                                y = R * np.sin(theta)
                                DOL[c] = [x, y, z, alpha, beta, gamma]
                                c += 1
        if filename:
                createDOL(filename, DOL)
        return DOL

def IT2dol(R, L_dim, T_dim, h, pitch=0, startAt=0, alpha_shift=0, beta_shift=0, gamma_shift=0, stand=False, filename=''):
        p = L_dim + pitch if stand else T_dim + pitch
        L_turn = np.sqrt((2*np.pi * R)**2 + p**2) 
        i_max = int((h* L_turn)/(p * L_dim) + 1) 
        theta_dim = L_dim/ R
        alpha = alpha_shift if stand else 90 - np.arcsin(p/L_turn) * 180/ np.pi + alpha_shift
        c = 0
        DOL = np.zeros([i_max, 6])
        for i in range(i_max):
                theta = np.remainder(i * theta_dim, np.pi * 2)
                x = R * np.cos(theta)
                y = R * np.sin(theta)
                z = startAt + i * p * L_dim / L_turn
                beta = beta_shift if stand else (180 * theta / np.pi) + beta_shift
                DOL[c] = [x, y, z, alpha, beta, gamma_shift]
                c += 1
        if filename:
                createDOL(filename, DOL)
        return DOL

def CS2dol(R_start, R_final, L_dim, T_dim, loops=3, pitchPerLoop=0, startAt=0, alpha_shift=0, beta_shift=0, gamma_shift=0, filename=''):
        p = T_dim + pitchPerLoop
        c_value = (R_start - R_final)/(2 * np.pi * loops)
        t = find_t(2*np.pi*loops,c_value,p,R_start,R_final,L_dim, eps=1e-14)
        dt = np.diff(t)
        t = t[:-1]
        N_tot = len(t)
        DOL = np.zeros([N_tot, 6])
        c = 0
        for i in range(N_tot):
                x = (R_start - c_value*t[i])*np.cos(t[i])
                y = (R_start - c_value*t[i])*np.sin(t[i])
                z = startAt + p/(2*np.pi) * t[i]
                alpha = 90 - np.arcsin((p*dt[i])/(2*np.pi * L_dim)) * 180/ np.pi + alpha_shift
                beta = (180 * t[i] / np.pi) + beta_shift
                DOL[c] = [x, y, z, alpha, beta, gamma_shift]
                c += 1
        if filename:
                createDOL(filename, DOL)
        return DOL

# R is array, more R (you can use R=[r1, r2, r3] for 3 closed rings or R=r1 for on closed ring)
def Rings2dol(R, L_dim, T_dim, pitchPerRing=0, startAt=0, alpha_shift=0, beta_shift=0, gamma_shift=0, stand=False, filename=''):
        p = L_dim + pitchPerRing if stand else T_dim + pitchPerRing
        R = np.array([R]) if np.isscalar(R) else np.array(R)
        rings = len(R)
        L_turn = np.sqrt((2*np.pi * R)**2) # TODO: Change L_turn to S
        N_turn = np.int64(L_turn/L_dim) # TODO: Change N_turn to n
        angle = 2 * np.pi / N_turn
        c = 0
        DOL = np.zeros([np.sum(N_turn), 6])
        z_start = startAt
        for i in range(rings):
                for j in range(N_turn[i]):
                        theta = j * angle[i]
                        x = R[i] * np.cos(theta)
                        y = R[i] * np.sin(theta)
                        z = z_start + p * i
                        alpha = alpha_shift if stand else 90 + alpha_shift
                        beta = beta_shift if stand else (180 * theta / np.pi) + beta_shift
                        DOL[c] = [x, y, z, alpha, beta, gamma_shift]
                        c += 1
        if filename:
                createDOL(filename, DOL)
        return DOL

#=========================================
# Grid-like placement / DOL Manimpulations
#=========================================

def VerticalRep(DOL, repetitions=1, repeatVerticlaDistance=0, tiltPerRep=0, antiParallel=False, filename=''):
    unitHeight = np.max(DOL[:, 2]) - np.min(DOL[:, 2])
    p = unitHeight + repeatVerticlaDistance
    VDOL = DOL.copy()
    lx, lz = 0, 0
    for i in range(1, repetitions):
        DOL2 = DOL.copy()
        if tiltPerRep != 0:
            DOL2 = rotateDOL(DOL2, beta_shift=tiltPerRep*i, fixHeight=True)
        if antiParallel and i % 2 == 1:
            DOL2 = rotateDOL(DOL2, 180, fixHeight=True)
        lx += p * np.sin(tiltPerRep * np.pi / 180 * i)
        lz += p * np.cos(tiltPerRep * np.pi / 180 * i)
        DOL2[:, 2] += lz
        DOL2[:, 0] += lx

        VDOL = combineDOL(VDOL, DOL2)
    if filename:
        createDOL(filename, VDOL)
    return VDOL

def Hexagonal_Symmetry(DOL, layers=3, pitchPerUnit=0, unitRad=False, antiParallel=False, beta_shift=0,filename=''):
        if layers < 1:
                raise Exception('At least 2 layers is neccesery!')

        if not unitRad:
                unitRad = np.max(np.sqrt(DOL[:, 0]**2 + DOL[:, 1]**2))
        unitPitch = unitRad*2 + pitchPerUnit

        stand = True if len(np.unique(DOL[:, 4])) == 1 else False
        DOL_Mirror = rotateDOL(DOL, 180)

        N_IT = np.insert(np.arange(6, layers*6,6,dtype=int), 0, 1)
        HDOL = np.zeros([len(DOL)*np.sum(N_IT), 6])
        c = 0
        for IT in range(len(N_IT)):
                if IT < 2:
                        ringAngles = np.linspace(0, np.pi*2, N_IT[IT], endpoint=False) if N_IT[IT] > 0 else np.array([0])
                        cx = unitPitch * IT * np.cos(ringAngles)
                        cy = unitPitch * IT * np.sin(ringAngles)
                else:
                        ringAngles = np.linspace(0, np.pi*2, 6, endpoint=False)
                        corner_cx = unitPitch * IT * np.cos(ringAngles)
                        corner_cy = unitPitch * IT * np.sin(ringAngles)

                        cx = np.array([])
                        cy = np.array([])
                        for i in range(6):
                                x1, y1 = corner_cx[i], corner_cy[i]
                                x2, y2 = corner_cx[(i + 1) % 6], corner_cy[(i + 1) % 6]
                                for j in range(IT):
                                        frac = j / IT
                                        ring_x = x1 * (1 - frac) + x2 * frac
                                        ring_y = y1 * (1 - frac) + y2 * frac
                                        cx = np.append(cx, ring_x)
                                        cy = np.append(cy, ring_y)          
                for i in range(N_IT[IT]):
                        theta_shift = np.arctan2(cy[i], cx[i]) * 180 / np.pi
                        DOL_i = rotateDOL(DOL,0,0,theta_shift, False)
                        DOL_Mirror_i = rotateDOL(DOL_Mirror,0,0,theta_shift, False)
                        for j in range(len(DOL)):
                                if antiParallel and IT % 2 == 1:
                                        x, y, z, alpha, beta, gamma = DOL_Mirror_i[j]
                                else:
                                        x, y, z, alpha, beta, gamma = DOL_i[j]
                                x += cx[i]
                                y += cy[i]
                                HDOL[c] = [x, y, z, alpha, beta, gamma]
                                c += 1
        if filename:
                createDOL(filename, HDOL)
        return HDOL

def Square_Symmetry(DOL, layer_x=3, layer_y=3, tilt=0, pitchPerUnit=0, unitRad=False, antiParallel=False, beta_shift=0, filename=''):
        if layer_x + layer_y < 2:
                raise Exception('At least 2 layers is neccesery!')
        if tilt > 30:
                raise Exception('Tilt can not exceed 30 degrees')
        if layer_x == 1 and layer_y == 1:
               if filename:
                      createDOL(filename, DOL)
               return DOL
        if not unitRad:
                unitRad = np.max(np.sqrt(DOL[:, 0]**2 + DOL[:, 1]**2))
        unitPitch = unitRad*2 + pitchPerUnit
        x_arr = np.array([i for i in range(-(layer_x // 2), (layer_x // 2) + 1)])
        y_arr = np.array([i for i in range(-(layer_y // 2), (layer_y // 2) + 1)])
        if layer_x % 2 == 0:
                x_arr = x_arr[x_arr != 0]
        if layer_y % 2 == 0:
                y_arr = y_arr[y_arr != 0]        
        Ax = unitPitch/2 if layer_x % 2 == 0 else 0
        Ay = unitPitch/2 if layer_y % 2 == 0 else 0

        DOL_Mirror = rotateDOL(DOL, 180)

        SDOL = np.zeros([len(DOL)*layer_x*layer_y, 6])
        c = 0
        for x_ind, xi in enumerate(x_arr):
                for y_ind, yi in enumerate(y_arr):
                        cx = xi * unitPitch + yi * unitPitch * np.sin(tilt * np.pi / 180) - Ax * np.sign(xi) 
                        if layer_y % 2 == 0 and tilt != 0:
                                cx -= (unitPitch * np.sin(tilt * np.pi / 180)) / 2 * np.sign(yi)
                        cy = (yi * unitPitch - Ay * np.sign(yi)) * np.cos(tilt * np.pi / 180)
                        theta_shift = np.arctan2(cy, cx) * 180 / np.pi
                        DOL_i = rotateDOL(DOL,0,0,theta_shift, False)
                        DOL_Mirror_i = rotateDOL(DOL_Mirror,0,0,theta_shift, False)
                        for j in range(len(DOL)):
                                if antiParallel and (x_ind + y_ind) % 2 == 1:
                                        x, y, z, alpha, beta, gamma = DOL_Mirror_i[j]
                                else:
                                        x, y, z, alpha, beta, gamma = DOL_i[j]

                                x += cx
                                y += cy

                                SDOL[c] = [x, y, z, alpha, beta, gamma]
                                c += 1  

        if filename:
                createDOL(filename, SDOL)
        return SDOL              

def combineDOL(DOL1, DOL2, filename=''):
        DOL1 = np.array(DOL1)
        DOL2 = np.array(DOL2)
        DOL = np.vstack((DOL1,DOL2))

        if filename:
                createDOL(filename,DOL)
        return DOL

def rotateDOL(DOL, alpha_shift=0, beta_shift=0, gamma_shift=0, fixHeight=True, filename=''):
    def A_x(angle):
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])

    def A_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])

    def A_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    DOL_rotated = DOL.copy()

    alpha_shift, beta_shift, gamma_shift = np.radians([alpha_shift, beta_shift, gamma_shift])
    A = A_x(alpha_shift) @ A_y(beta_shift) @ A_z(gamma_shift)
    for i in range(len(DOL_rotated)):
        x, y, z = np.dot(A, DOL[i, :3])
        alpha, beta, gamma = np.radians(DOL[i, 3:])

        A_o = A_x(alpha) @ A_y(beta) @ A_z(gamma)  # Orientation Mat
        A_o_tag = np.dot(A, A_o)
        
        # Euler Angles
        beta_new = np.arcsin(A_o_tag[0, 2]) * 180 / np.pi
        gamma_new = np.arctan2(-A_o_tag[0, 1], A_o_tag[0, 0]) * 180 / np.pi
        alpha_new = np.arctan2(-A_o_tag[1, 2], A_o_tag[2, 2]) * 180 / np.pi

        if gamma_new < 0:
            gamma_new += 360
        if beta_new < 0:
            beta_new += 360
        if alpha_new < 0:
            alpha_new += 360

        DOL_rotated[i, :] = [x, y, z, alpha_new, beta_new, gamma_new]
    if fixHeight:
        DOL_rotated[:, 2] -= np.min(DOL_rotated[:, 2])

    if filename:
        createDOL(filename, DOL_rotated)
    return DOL_rotated
