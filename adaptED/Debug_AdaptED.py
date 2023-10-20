import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,TextBox
import os
from dplus.CalculationInput import CalculationInput
from inspect import signature
from dplus.DataModels.models import UniformHollowCylinder, Sphere, Helix, SymmetricLayeredSlabs, AsymmetricLayeredSlabs
import re
from tkinter.filedialog import asksaveasfile
from scipy.interpolate import interp1d

step = []

# ===== Example functions =====

'''
Smoothing functions should get 3 parameters:
1. r - value of the function at r aka f(r)
2. par - vector of arguments for the function
    a. First two arguments r1,r2 must be added to it
    b. Last argument is the polED
        
    ### Example par = [r1, r2, a1, a2, ...., an, polED] ###
3. **solED - Solvent electron density** (By default in water solED=333)
'''

# Par needs to be 2-D array:
# incorrect par = [a1,...]
# correct par = [[a1,...]]

def tanhED(r, par, solED=333):
    if np.ndim(par) == 1:
        par = np.array([par])
    if np.size(par, axis=1) != 5:
        raise Exception("Parameters are not vailed!\npar = [r1, r2, a1, a2, polED]") #TODO: Fix error message!
    r1, r2, a1, a2, polED = np.transpose(par)
    x1 = (r - r1) * a1
    x2 = (r - r2) * a2
    multi_tanh = np.sum((polED-solED)*(np.tanh(x1)/2 - np.tanh(x2)/2))
    return multi_tanh + solED

def gaussianED(r, par, solED=333): #TODO: test new algo
    if np.ndim(par) == 1:
        par = np.array([par])
    if np.size(par, axis=1) != 3:
        raise Exception("Parameters are not vailed!\npar = [radius, tau(center of the gaussian), polED]") #TODO: Fix error message!
    radius, tau, polED = np.transpose(par)
    multi_gauss = np.sum((polED-solED)*(np.exp(-((4*np.log(2)*(r - radius)**2)/(tau**2)))))
    return multi_gauss + solED

#TODO: fix if's in parabolaED function
def parabolaED(r, par, solED=333):
    #par = [r1, r2, a1, a2, a3, polED]
    if len(par) != 6:
        raise Exception("Parameters are not vailed!\npar = [r1, r2, a1, a2, a3, polED]")
    r1, r2, a1, a2, a3, polED = par
    if type(r) == np.ndarray:
        y = np.zeros(np.size(r))
        y[r >= r1] =  polED*(a1*(r[r >= r1]**2) + a2*r[r >= r1] + a3) + solED
        y[r <= r2] =  polED*(a1*(r[r <= r2]**2) + a2*r[r <= r2] + a3) + solED
        y[r > r2] = solED
        y[r < r1] = solED
        return y
    if r < r1 or r > r2:
        return solED
    return polED*(a1*(r**2) + a2*r + a3) + solED
    
def interpolatedED(r, par, solED=333):
    z, rho = par
    max_peak = z[rho == np.max(rho)]
    if len(max_peak) > 0:
        max_peak = max_peak[0]
    a = 387.5
    rho[(z > max_peak) & (rho <= a)] = solED + ((a - solED) * (rho[(z > max_peak) & (rho <= a)] - np.min(rho[(z > max_peak) & (rho <= a)]))) / (np.max(rho[(z > max_peak) & (rho <= a)]) - np.min(rho[(z > max_peak) & (rho <= a)]))
    
    z_vec = np.concatenate([-z[::-1], z])
    #dz = z_vec[0] - z_vec[1]
    rho = np.concatenate([rho[::-1], rho])
    #z_vec = np.append(z_vec, np.array([np.max(z)+dz,np.max(z)+2*dz,np.max(z)+3*dz,np.max(z)+4*dz,np.max(z)+5*dz,np.max(z)+6*dz,np.max(z)+7*dz,np.max(z)+8*dz,np.max(z)+9*dz]))
    #rho = np.append(rho, np.array([solED,solED,solED,solED,solED,solED,solED,solED,solED]))
    f = interp1d(z_vec, rho, kind='linear')
    try:
        return np.float(f(r))
    except:
        return solED
    
def interpolatedED_box(r, par, solED=333):
    z, rho = par
    max_peak = z[rho == np.max(rho)]
    a = 387.5
    #rho[(z > max_peak) & (rho <= a)] = solED + ((a - solED) * (rho[(z > max_peak) & (rho <= a)] - np.min(rho[(z > max_peak) & (rho <= a)]))) / (np.max(rho[(z > max_peak) & (rho <= a)]) - np.min(rho[(z > max_peak) & (rho <= a)]))
    
    z_vec = np.concatenate([-z[::-1], z])
    #dz = z_vec[0] - z_vec[1]
    rho = np.concatenate([rho[::-1], rho])
    #z_vec = np.append(z_vec, np.array([np.max(z)+dz,np.max(z)+2*dz,np.max(z)+3*dz,np.max(z)+4*dz,np.max(z)+5*dz,np.max(z)+6*dz,np.max(z)+7*dz,np.max(z)+8*dz,np.max(z)+9*dz]))
    #rho = np.append(rho, np.array([solED,solED,solED,solED,solED,solED,solED,solED,solED]))
    #======
    for i in range(len(rho)):
        if rho[i] < solED:
            rho[i] = solED
    #======
    f = interp1d(z_vec, rho, kind='linear')
    try:
        return np.float(f(r))
    except:
        return solED
# ===== Example functions =====

# ===== Addons =====
def checkRadiusOverlap(a):
    if np.ndim(a) < 2:
        return False
    for i in range(len(a)):
        if a[i,0] > a[i,1]:
            return True
        for j in range(i+1,len(a)):
            if a[i, 1] > a[j, 0]:
                raise Exception('Error: Radius overlaps in the provided parameters!') #TODO: Check message
    return False

def upperBoundT(parm):
    if np.ndim(parm) == 1:
        parm = np.array([parm])
    r = parm[:,1]
    alpha = parm[:,3]
    upper = r + 10.36/alpha
    return np.max(upper)

def upperBoundG(parm):
    if np.ndim(parm) == 1:
        parm = np.array([parm])
    sigma = 3*(parm[:,1])/(2*np.sqrt(2*np.log(2)))+parm[:,0] 
    return max(sigma)
# ===== Addons =====

# ===== Adaptive =====
def AdaptiveSmooth(fun, fun_args,  a, b, epsilon= 0.001, solED=333):
    mid = (a + b)/2
    if not step:
         step.append(a)
         step.append(b)
    step.append(mid)
    midpoint = (b-a)*fun(mid, fun_args, solED)
    simpson = ((b-a)/6)*(fun(a, fun_args, solED) + 4*fun(mid, fun_args, solED) + fun(b, fun_args, solED))
    if np.abs(simpson-midpoint) < epsilon:    
        return simpson
    else:
        return AdaptiveSmooth(fun, fun_args, a, mid, epsilon/2, solED) + AdaptiveSmooth(fun, fun_args, mid, b, epsilon/2, solED)

# TODO:
#       - Fix asymmetricslabs creation

def adaptED_to_state(radius, ED, filename='target.state',  model='Sphere', load=False ,extra_params=[], save_state=False):
    sphericaltype = True
    if load:
        CI = CalculationInput().load_from_state_file(filename)
    else:
        CI = CalculationInput()
    if model == 'Sphere' or model == 'sphere':
        state = Sphere()
        if extra_params:
            state.extra_params["scale"].value = extra_params[0]
            state.extra_params["background"].value = extra_params[1]
    elif model == 'UniformHollowCylinder' or model == 'cylinder' or model == 'Cylinder' or model == 'uniformhollowcylinder':
        state = UniformHollowCylinder()
        if extra_params:
            state.extra_params["scale"].value = extra_params[0]
            state.extra_params["background"].value = extra_params[1]
            state.extra_params["height"].value = extra_params[2]
    elif model == 'SymmetricLayeredSlabs' or model == 'SymmetricSlabs' or model == 'Sslab': #TODO: names 
        state = SymmetricLayeredSlabs()
        sphericaltype = False
        if extra_params:
            state.extra_params["scale"] = extra_params[0]
            state.extra_params["background"] = extra_params[1]
            state.extra_params["x_domain_size"] = extra_params[2]
            state.extra_params["y_domain_size"] = extra_params[3]
    elif model == 'AsymmetricLayeredSlabs' or model == 'asymslab' or model == 'ASslab': #TODO: names
        state = AsymmetricLayeredSlabs()
        sphericaltype = False
        if extra_params:
            state.extra_params["scale"] = extra_params[0]
            state.extra_params["background"] = extra_params[1]
            state.extra_params["x_domain_size"] = extra_params[2]
            state.extra_params["y_domain_size"] = extra_params[3]
    else:    
        raise Exception(str(model) + 'is not supported!') # TODO: Check if this message is good
    if sphericaltype:
        state.layer_params[1].radius.value = radius[0]
    else: 
        state.layer_params[1].width.value = radius[0]
    state.layer_params[1].ed.value = ED[0]
    for i in range(1,len(radius)):
        state.add_layer()
        if sphericaltype:
            state.layer_params[i+1].radius.value = radius[i] - radius[i - 1]
        else:
            state.layer_params[i+1].width.value = radius[i] - radius[i - 1]
        state.layer_params[i+1].ed.value = ED[i]
    
    CI.Domain.populations[0].add_model(state)
    if save_state:
        CI.export_all_parameters(filename)
        print('Electron density profile has been created!\n' + 'Saved state file:', filename)
    return CI

# TODO:
#       - Fix extra parameter for min bound
def adaptED(params, funED=tanhED, model='Sphere', convergence=0.01, solED=333, extra_params=[], maxbound=False, filename='target.state', save_file=False):
    layer_y = []
    loadfile = False
    if maxbound: 
        if type(maxbound) != Sphere():
            AdaptiveSmooth(funED, params, 0, maxbound, convergence, solED)
        else:
            AdaptiveSmooth(funED, params, 0, maxbound(params), convergence, solED)
            if len(signature(maxbound).parameters) == 1:
                AdaptiveSmooth(funED, params, 0, maxbound(params), convergence, solED)
            else:
                AdaptiveSmooth(funED, params, 0, maxbound, convergence, solED)
    else: 
        if funED == tanhED:
            checkRadiusOverlap(params)
            AdaptiveSmooth(funED, params,0,upperBoundT(params), convergence, solED)
        elif funED == gaussianED:
            AdaptiveSmooth(funED, params,0,upperBoundG(params), convergence, solED)
        else:
            raise Exception("Maximum bound of the profile is missing!")
    
    layer_x = np.unique(np.concatenate(step,axis=None))
    for i in range(len(layer_x)):
        layer_y.append(funED(layer_x[i], params, solED))
    
    if os.path.isfile(filename):
        loadfile = True
    CI = adaptED_to_state(layer_x, layer_y, filename, model, loadfile, extra_params,save_file)
    return CI
# ===== Adpative =====

# TODO:
#       - Fix asymetric Slabs graph
#       - ?
def plotEDProfile(filename, models=['Sphere', 'UniformHollowCylinder', 'SymmetricLayeredSlabs', 'AsymmetricLayeredSlabs'], gx=0, gy=0):
    global UI_R,UI_ED,ind
    UI_R,UI_ED,ind = [],[],0
    UI_fig, UI_ax = plt.subplots()
    if type(models) != list:
        models = [models]
    for i in range(len(models)):
        models[i] = re.sub(r"(\w)([A-Z])", r"\1 \2", models[i])
    if type(filename) == str:
        CI = CalculationInput().load_from_state_file(filename).get_models_by_type(models)
    else:
        CI = filename.get_models_by_type(models)
    for i in range(len(CI)):
        isRadius = True
        solED = CI[i].layer_params[0]["ed"].value
        TW = 0
        UI_R.append(np.zeros(len(CI[i].layer_params) + 1))
        UI_ED.append(np.zeros(len(CI[i].layer_params) + 1))
        try:
                dummy = CI[i].layer_params[1]["radius"].value
        except:
            isRadius = False
        for j in range(len(CI[i].layer_params)):
            UI_ED[i][j] = CI[i].layer_params[j]["ed"].value
            if isRadius:
                UI_R[i][j] = CI[i].layer_params[j]["radius"].value + TW
                TW += CI[i].layer_params[j]["radius"].value
            else:
                UI_R[i][j] = CI[i].layer_params[j]["width"].value + TW
                TW += CI[i].layer_params[j]["width"].value
        UI_R[i][len(CI[i].layer_params)] = TW
        UI_ED[i][len(CI[i].layer_params)] = solED
    UI_ax.step(UI_R[0],UI_ED[0])
    plt.title("Graph number: " + str(ind+1))
    def nextProfile(val):
        global ind
        ind += 1
        ind = ind % len(UI_R)
        UI_ax.clear()
        UI_ax.step(UI_R[ind], UI_ED[ind]) 
        drawSub()
            
    def backProfile(val):
        global ind
        ind -= 1
        ind = abs(ind % len(UI_R))    
        UI_ax.clear()
        UI_ax.step(UI_R[ind], UI_ED[ind])
        drawSub()
            
    def savef(val):
        files = [('CSV', '*.csv'),
                ('All Files', '*.*')] 
        file = asksaveasfile(filetypes = files, defaultextension = files)
        createCSV(file.name, np.array([UI_R[ind] , UI_ED[ind]]))

    def createCSV(filename, params):
        f = open(filename, 'w')
        f.write('Radius,Electron Density' + '\n')
        for i in range(len(params[0,:])):
            f.write(str(params[0,i])+ ',' +str(params[1,i]))
            f.write('\n')
        print('The file has been created!')

    def graphDesign(first=False):
        solED = UI_ED[ind][0]
        plt.axhline(y=solED, color="black", linewidth=3)
        plt.xlim([-0.1,max(UI_R[ind])+1])
        plt.grid()
        locs, labels = plt.yticks()
        plt.yticks(np.append(locs,solED),np.append(labels, "Solvent"),fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylabel('Electron Density', fontsize=15)
        plt.xlabel('Radius', fontsize=15)
        axnext = plt.axes([0.385, 0.02, 0.05, 0.05])
        axback = plt.axes([0.335, 0.02, 0.05, 0.05])
        axexport = plt.axes([0.125, 0.02, 0.2, 0.05])
        if first:
            bnext = Button(axnext, 'Next')
            bback = Button(axback, 'Back')
            bexport = Button(axexport, 'export...')
            bnext.on_clicked(nextProfile)   
            bback.on_clicked(backProfile)
            bexport.on_clicked(savef)
            plt.show()

    def drawSub():
        plt.draw()
        solED = UI_ED[ind][0]
        UI_ax.grid()
        UI_ax.axhline(y=solED, color="black", linewidth=3)
        Ylocs, Ylabels = UI_ax.get_yticks(), UI_ax.get_yticklabels()
        Xlocs, Xlabels = UI_ax.get_xticks(), UI_ax.get_xticklabels()
        UI_ax.set_yticks(np.append(Ylocs,solED),np.append(Ylabels, "Solvent"),fontsize=12)
        UI_ax.set_xticks(Xlocs,labels=Xlabels,fontsize=12)
        UI_ax.set_xlim([-0.1,max(UI_R[ind])+1])
        UI_ax.set_title("Graph number: " + str(ind+1))
        UI_ax.set_ylabel('Electron Density', fontsize=15)
        UI_ax.set_xlabel('Radius', fontsize=15)
    graphDesign(True)
    plt.close()
#plotEDProfile('states/g1.state',['Sphere', 'UniformHollowCylinder', 'SymmetricLayeredSlabs', 'AsymmetricLayeredSlabs'])
