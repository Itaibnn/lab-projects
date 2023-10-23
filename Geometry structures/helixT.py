import numpy as np

# Usage example:
# dol2helix_norm(R=24,L_dim=8.3,T_dim=5.2,N_p=14,h=200,'newvarb')
# R - Radius of the helix
# L_dim, T_dim - Length and Width of the dimer
# N_p - protofilments that turn around togther
# h - Total height of the helix (not fixed, it's not the final height)

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

# =================
# Structures
# =================
def helix2dol(R, L_dim, T_dim, N_p, h, filename='what', dR=1, R_phase=0, savefile=True):
        p = N_p * T_dim
        L_turn = np.sqrt((2 * np.pi * R)**2 + p**2)
        N_turn = L_turn/L_dim
        i_max = int(h/p*N_turn)
        k_max = N_p - 1
        DOL = np.zeros([i_max* k_max, 6])
        c = 0
        for k in range(k_max):
                z_start = k*p/N_p
                for j in range(i_max):
                        alpha = np.arctan(p/(2*np.pi*R)) * 180 / np.pi
                        beta = 0
                        gamma = j * 2 * np.pi / N_turn
                        z = z_start + (j * p / N_turn)
                        r = R - dR * np.abs(np.sin((np.pi*z)/p + R_phase* np.pi/180))
                        x = r * np.cos(gamma)
                        y = r * np.sin(gamma)
                        DOL[c] = [x, y, z, alpha, beta, gamma]
                        c += 1
        if savefile:
                createDOL(filename, DOL)
        return DOL

def ring2dol(R, T_dim, L_dim, h, startAt=0, alpha=0, beta=0, pitch=0, filename='ringtest', savefile=True):
        p = T_dim + pitch
        L_turn = np.sqrt((2*np.pi * R)**2 + p**2) # Length of one complete turn
        N_turn = int(L_turn/L_dim) # Number of dimers for one turn
        rings = int(h/p) # Number of complete rings
        angle = 2 * np.pi / N_turn
 
        c = 0
        N_remain = int((h - p*rings)/(p/N_turn))
        DOL = np.zeros([rings*N_turn + N_remain, 6])
        for k in range(rings + 1):
                z_start = startAt + k*p
                j_max = N_turn if k < rings else N_remain
                for j in range(j_max):
                        theta = j * angle
                        x = R * np.sin(theta)
                        y = R * np.cos(theta)
                        z = z_start + (j * p/ N_turn)
                        gamma = 90 - (180 * theta / np.pi)
                        DOL[c] = [x, y, z, alpha, beta, gamma]
                        c += 1
        if savefile:
                createDOL(filename, DOL)
        return DOL

def CS2dol(R, T_dim, L_dim, h, startAt=0, alpha=0, beta=0,filename='ringtest', savefile=True):
        p = T_dim  
        L_turn = np.sqrt((2*np.pi * R)**2 + p**2) # Length of one complete turn
        N_turn = int(L_turn/L_dim) # Number of dimers for one turn
        rings = int(h/p) # Number of complete rings
        angle = 2 * np.pi / N_turn
        DOL = np.zeros([rings*N_turn, 6])
        c = 0
        for k in range(rings):
                z_start = startAt + k*p   
                for j in range(N_turn):
                        theta = j * angle
                        x = R * np.sin(theta)
                        y = R * np.cos(theta)
                        z = z_start + (j * p/ N_turn)
                        gamma = 90 - (180 * theta / np.pi)
                        DOL[c] = [x, y, z, alpha, beta, gamma]
                        c += 1
        if savefile:
                createDOL(filename, DOL)
        return DOL

def MT2dol(radius=27, pitch=5.1, unitsPerPitch=20, unitsInPitch=20, startAt=0, discreteHeight=10, numHelixStarts=1, superHelicalPitch=0, RingTwistAlpha=0, RingTwistBeta=0, filename='test', savefile=True):
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
        if savefile:
                createDOL(filename,DOL)
        return DOL

def combineDOL(DOL1, DOL2, filename='combine', savefile=True):
        DOL1 = np.array(DOL1)
        DOL2 = np.array(DOL2)
        DOL = np.vstack((DOL1,DOL2))
        if savefile:
                createDOL(filename,DOL)
        return DOL