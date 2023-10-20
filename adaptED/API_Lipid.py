import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r"\\raviv_backup\raviv_group\Homes\Itai\Projects\PythonScripts")
import chemConvertor as conv
from chemConvertor import atomic_radii_A, number_to_symbol, atomic_numbers
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import Debug_AdaptED as dub
import csv
from dplus.CalculationRunner import EmbeddedLocalRunner




# ==========================
# Calculation functions
# ==========================

# TODO: 
#       - add FC (Formal Charge)
#       - extract FC chemConvertor
# TODO: 
#       - add FC (Formal Charge)
#       - extract FC chemConvertor

# Debugg function
def cal_height(z, n):
    zmin, zmax = np.zeros(len(z)), np.zeros(len(z))
    for i in range(len(n)):
        rad = atomic_radii_A[number_to_symbol[n[i]]]/100
        zmin[i] += z[i] - rad
        zmax[i] += z[i] + rad
    return np.min(zmin), np.max(zmax)

def electron_positions_Vol(n, z, FC=0):
    FC = np.zeros(len(z), dtype=int)
    FC[0] = -1
    FC[19] = -1
    FC[21] = 1
    FC[20] = 1
    electron_positions = []
    for i in range(len(n)):
        num_electrons = n[i] + FC[i]  # atomic number equals the number of electrons for neutral atoms
        radius = atomic_radii_A[number_to_symbol[n[i]]]
        if num_electrons == 1:
            spread = [z[i]]
        else:
            theta1 = 0
            spread = []
            for j in range(1, num_electrons+1):
                theta2 = np.arccos(np.cos(theta1) - 2/num_electrons)
                spread.append(z[i] + radius/2*(np.cos(theta2) + np.cos(theta1)))
                theta1 = theta2 - 1e-15
        electron_positions.extend(spread)
    return np.array(electron_positions)

def electron_positions_nuc(n, z, FC=0):
    FC = np.zeros(len(z), dtype=int)
    FC[0] = -1
    FC[19] = -1
    FC[21] = 1
    FC[20] = 1
    electron_positions = []
    for i in range(len(n)):
        num_electrons = n[i] + FC[i]  # atomic number equals the number of electrons for neutral atoms
        radius = atomic_radii_A[number_to_symbol[n[i]]]    
        if num_electrons == 1:
            spread = [z[i]]
        else:
            spread = []
            for j in range(1, num_electrons+1):
                spread.append(z[i])
        electron_positions.extend(spread)
    return np.array(electron_positions)


def electron_positions_box(n, z, nFC):
    FC = np.zeros(len(z), dtype=int)
    electron_positions = []

    for i in range(len(n)):
        if nFC[i] == 'P':
            FC[i] += -1
        elif n[i] == 'N':
            FC[i] += -1
        elif nFC[i] == 'O11' or nFC == 'O12':
            FC[i] += 1
        num_electrons = n[i] + FC[i]  # atomic number equals the number of electrons for neutral atoms
        radius = atomic_radii_A[number_to_symbol[n[i]]] 
        if num_electrons == 1:
            spread = [z[i]]
        else:
            theta1 = 0
            spread = []
            for j in range(1, num_electrons+1):
                theta2 = np.arccos(np.cos(theta1) - 2/num_electrons)
                spread.append(z[i] + radius/2*(np.cos(theta2) + np.cos(theta1)))
                theta1 = theta2 - 1e-15
        electron_positions.extend(spread)
    return np.array(electron_positions)

def electron_positions_gen(n, z, nFC=0):
    FC = np.zeros(len(z), dtype=int)
    electron_positions = []

    for i in range(len(n)):
        num_electrons = n[i] + FC[i]  # atomic number equals the number of electrons for neutral atoms
        radius = atomic_radii_A[number_to_symbol[n[i]]]/100
        if num_electrons == 1:
            spread = [z[i]]
        else:
            theta1 = 0
            spread = []
            for j in range(1, num_electrons+1):
                theta2 = np.arccos(np.cos(theta1) - 2/num_electrons)
                spread.append(z[i] + radius/2*(np.cos(theta2) + np.cos(theta1)))
                theta1 = theta2 - 1e-15
        electron_positions.extend(spread)
    return np.array(electron_positions)

def atomConvexHull(x, y, z):
    if len(x) == 0:
        return 1
    else:
        hull = ConvexHull(np.column_stack([x, y, z]))
        V = hull.volume
        return V


def calculateVolume(x, y, z, n, res=10):
    theta = np.linspace(0, np.pi, res)
    phi = np.linspace(0, 2*np.pi, res)
    theta, phi = np.meshgrid(theta, phi)
    num_atoms = len(x)
 
    x = np.reshape(x, (num_atoms, 1, 1))
    y = np.reshape(y, (num_atoms, 1, 1))
    z = np.reshape(z, (num_atoms, 1, 1))
    n = np.reshape(n, (num_atoms, 1, 1))

    r = np.array([atomic_radii_A[number_to_symbol[i]] for i in n.flatten()])/100
    r = np.reshape(r, (num_atoms, 1, 1))
    
    x_s = x + r * np.sin(theta) * np.cos(phi)
    y_s = y + r * np.sin(theta) * np.sin(phi)
    z_s = z + r * np.cos(theta)
    
    x_s = x_s.flatten()
    y_s = y_s.flatten()
    z_s = z_s.flatten()
    return ConvexHull(np.column_stack([x_s, y_s, z_s])).volume , z_s
    #return (np.max(x_s) - np.min(x_s)) * (np.max(y_s) - np.min(y_s)) * (np.max(z_s) - np.min(z_s))

def calculateVolume_single(x, y, z, n, dz, rho):
    dv = np.zeros(len(rho))

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)
    return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

def monteCarloVolume(x, y, z, N=10000):
    if len(x) == 0:
        return 0

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)
    inside_count = N
    bounding_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    return (inside_count / N) * bounding_volume

def rho_of_zN(x, y, z, n, dz, V, electron_pos_fun=electron_positions_Vol):
    z_vec = np.arange(np.min(z), np.max(z), dz)
    e_pos = electron_pos_fun(n,z)
    rho = np.zeros(len(z_vec))
    V = calculateVolume(x,y,z,n)
    dv = V/len(rho)
    
    for i in range(len(rho)):
        for j in range(len(e_pos)):
            if e_pos[j] >= z_vec[i]  and e_pos[j] < z_vec[i] + dz:
                rho[i] += 1
    rho *= 1/dv
    
    return z_vec, rho

def rho_of_zN2(x, y, z, n, dz, V, nFC, electron_pos_fun=electron_positions_Vol):
    zmin, zmax = cal_height(z,n)
    e_pos = electron_pos_fun(n,z, nFC)
    z_vec = np.arange(0, zmax, dz)
    rho = np.zeros(len(z_vec))
    #V = 128*128*1100#calculateVolume(x,y,z,n)
    dv = V/len(rho)
    
    for i in range(len(rho)):
        for j in range(len(e_pos)):
            if e_pos[j] >= z_vec[i]  and e_pos[j] < z_vec[i] + dz:
                rho[i] += 1
    rho *= 1/dv
    
    return z_vec, rho


def rho_of_z_single(x, y, z, n, dz, V, electron_pos_fun=electron_positions_Vol):
    z_vec = np.arange(np.min(z), np.max(z), dz)
    e_pos = electron_pos_fun(n,z)
    rho = np.zeros(len(z_vec))
    print(n)
    for i in range(len(rho)):
        x_s = x[(z > z_vec[i]) & (z < z_vec[i] + dz)]
        y_s = y[(z > z_vec[i]) & (z < z_vec[i] + dz)]
        z_s = z[(z > z_vec[i]) & (z < z_vec[i] + dz)]
        n_s = n[(z > z_vec[i]) & (z < z_vec[i] + dz)]
        dv = calculateVolume(x_s,y_s,z_s,n_s)
        for j in range(len(e_pos)):
            if e_pos[j] >= z_vec[i]  and e_pos[j] < z_vec[i] + dz:
                
                rho[i] += 1
        rho[i] *= 1/dv
    #rho *= 1/dv
    
    return z_vec, rho


# ==========================
# Plot functions
# ==========================
def plot_electron_distribution(n, z, electron_pos_fun=electron_positions_Vol):
    electron_z = electron_pos_fun(n, z)
    fig, ax = plt.subplots()
    ax.scatter([0]*len(z), z, color='blue', marker='o', label='Nucleus')    
    ax.scatter([0]*len(electron_z), electron_z, color='red', marker='.', label='Electrons')

    # Change this for loop
    for i, atom_n in enumerate(n):
        radius = atomic_radii_A[number_to_symbol[atom_n]]
        circle = plt.Circle((0, z[i]), radius, color='black', fill=False)
        ax.add_artist(circle)
    
    ax.set_xlabel('x (arbitrary)')
    ax.set_ylabel('z')
    ax.set_title('Electron Distribution in z-direction')
    ax.legend()


# ==========================
# Tests functions (X+ implementation)
# ==========================
# TODO: 
# - fix title (first line in ini)
# - test it
# - make it from blank (without using exsiting file)
# - Add more options to play with
width_default_attributes = """\
Width{i} = {value:.6f}
Width{i}mut = 0
Width{i}Cons = 0
Width{i}min = 0.000000
Width{i}max = 0.000000
Width{i}minind = -1
Width{i}maxind = -1
Width{i}linkind = -1
Width{i}sigma = 0.000000
"""

ed_default_attributes = """\
E.D.{i} = {value:.6f}
E.D.{i}mut = 0
E.D.{i}Cons = 0
E.D.{i}min = 0.000000
E.D.{i}max = 0.000000
E.D.{i}minind = -1
E.D.{i}maxind = -1
E.D.{i}linkind = -1
E.D.{i}sigma = 0.000000
"""
def update_layers_alternate_with_saving_name(input_file_path, saving_name, width_array, ed_array):
    with open(input_file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    in_slabs_section = False
    original_layer_count = 0
    done_with_width = False
    done_with_ed = False
    print(width_array)
    for i in range(len(width_array) - 1, 1, -1):
        width_array[i] += -width_array[i-1]
    print(width_array)
    print(ed_array)
    for line in lines:
        if "[Symmetric Uniform Slabs]" in line:
            in_slabs_section = True
        elif "layers =" in line and in_slabs_section:
            original_layer_count = int(line.split('=')[1].strip())
            new_lines.append(f"layers = {original_layer_count + len(width_array)}\n")
            continue
        elif "Width" in line and in_slabs_section and not done_with_width:
            new_lines.append(line)
            if f"Width{original_layer_count}" in line:
                done_with_width = True
                for i, value in enumerate(width_array, start=original_layer_count+1):
                    new_lines.append(width_default_attributes.format(i=i, value=value))
            continue
        elif "E.D." in line and in_slabs_section and not done_with_ed:
            new_lines.append(line)
            if f"E.D.{original_layer_count}" in line:
                done_with_ed = True
                for i, value in enumerate(ed_array, start=original_layer_count+1):
                    new_lines.append(ed_default_attributes.format(i=i, value=value))
            continue
        else:
            new_lines.append(line)
    
    with open(saving_name, 'w') as f:
        f.writelines(new_lines)

# TODO: remove this function
def extract_xy_from_csv(csv_file_path):
    """
    Extract x and y values from a CSV file where x is in the first column and y is in the second column.
    The first row (header) is skipped.
    
    Parameters:
        csv_file_path (str): The path to the CSV file.
        
    Returns:
        np.ndarray: Numpy array of x values.
        np.ndarray: Numpy array of y values.
    """
    x_values = []
    y_values = []
    
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        

        next(csv_reader, None)
        
        for row in csv_reader:
            x_values.append(float(row[0]))
            y_values.append(float(row[1]))
            
    return np.array(x_values), np.array(y_values)

# ==========================
# Additional functions
# ==========================

def binConfN(lipid='DMPC', dz=0.2, res=500, holerad=0, electron_pos_fun=electron_positions_Vol):
    lipid_name = 'CRD/' + lipid + '/'
    lipid_lower = lipid.lower()
    plt.figure(figsize=(8, 8))
    z_mat = []
    rho_mat = []
    Vtot = np.zeros(2000)
    C = 0
    for conf in ['conf1', 'conf2']:
        for i in range(1, 1001):
            filepath = f"{lipid_name}{conf}/{lipid_lower}_{i}.crd"
            x, y, z, n = conv.crd_arrays(filepath)
            z -= np.min(z) + atomic_radii_A['H'] + holerad
            Vtot[C] = calculateVolume(x,y,z,n,res)
            z_vec, rho = rho_of_zN(x, y, z, n, dz, Vtot[C], electron_pos_fun)
            
            z_vec = np.concatenate([-z_vec[::-1], z_vec])
            rho = np.concatenate([rho[::-1], rho])
            
            z_mat.append(z_vec)
            rho_mat.append(rho)
            #print(C)
            #print(f"{conf}: {i}\n- Volume: {Vtot[C]}")
            C += 1
    z_max = 0.
            
    for j in range(len(z_mat)):
        if z_max < max(z_mat[j]):
            z_max = max(z_mat[j])

    z_len = np.linspace(-z_max,z_max,100000)
    samp_rho = np.zeros([len(rho_mat),len(z_len)])
    for i in range(len(samp_rho)):
        interp_rho = interp1d(z_mat[i],rho_mat[i],kind='linear', bounds_error=False,fill_value=0)
        samp_rho[i] += interp_rho(z_len)
    rho_final = np.mean(samp_rho,axis=0)*1000
    print('avg vol', np.mean(Vtot))
    plt.plot(z_len, rho_final)
    plt.xlabel('z (Å)')
    plt.ylabel('ρ(z) (e/nm³)')
    plt.title('Charge Density Profile of ' + lipid)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()    
    return z_len, rho_final

def binConf_single(filename='DMPC', dz=0.2, res=25):
    plt.figure(figsize=(8, 8))
    x, y, z, n, FC = conv.pdb_arrays2(filename)
    zmin, zmax = cal_height(z,n)
    z -= zmin
    Vtot, z_s = calculateVolume(x,y,z,n,res)
    z_vec, rho = rho_of_zN2(x, y, z, n, dz, Vtot, FC, electron_pos_fun=electron_positions_gen)
    print((np.max(x) - np.min(x)) * (np.max(y) - np.min(y)) * (np.max(z) - np.min(z)), Vtot)
    plt.plot(z_vec/10, rho*1000)
    plt.figure(figsize=(8, 8))
    #Vtot = calculateVolume(x,y,z,n)
    #z_vec, rho =  rho_of_zN(x, y, z, n, dz, 0)
    plt.plot(z_vec/10, rho*1000)
    return z_vec/10, rho*1000
#x,y = extract_xy_from_csv('a.csv')
#update_layers_alternate_with_saving_name('simu3.ini', 'fina3.ini', np.array(x), np.array(y))
    

#plot_electron_distribution(n,z)
#plot_charge_density_profile(n, z, 1000,1000, 1000,1000,0.45)
#z1, rho1 = binConf_single(filename='pdbs/testers/half.pdb',res=20,dz=0.25)
#plt.show()
#CI = dub.adaptED(np.array([z1,rho1]), dub.interpolatedED_box, 'SymmetricSlabs',1e-2,maxbound=np.max(z1), filename='Missions of Wen/181023/symetric2.state', save_file=True)
z1, rho1 = binConf_single(filename='Missions of Wen/191023/pdbasym2.pdb',res=20,dz=0.25)
plt.show()
CI = dub.adaptED(np.array([z1,rho1]), dub.interpolatedED_box, 'AsymmetricLayeredSlabs',1e-2,maxbound=np.max(z1), filename='Missions of Wen/191023/asymetricpdb2.state', save_file=True)
#dub.plotEDProfile(CI)
