import os
import subprocess
import numpy as np
import time
import glob
import csv
import tkinter as tk
from tkinter import filedialog
import cv2
from math import sqrt, radians, cos, sin, radians
import matplotlib.pyplot as plt
from scipy.stats import (wasserstein_distance, wasserstein_distance_nd)
from threading import Thread
import sys
import warnings
warnings.filterwarnings('ignore')
#from mpl_toolkits import mplot3d
from PIL import Image
from pymks import (
    #generate_multiphase,
    plot_microstructures,
    PrimitiveTransformer,
    TwoPointCorrelation,
    paircorr_from_twopoint,
    #FlattenTransformer
)
import dask.array as da
import GooseEYE
import porespy as ps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd


# Here you should define the paths on your computer 
working_dir = '/Users/tmr96/Documents/Automatic'
images_path = '/Users/tmr96/projects/my_files/images' # Images folder
convert_script_path = working_dir + "/convert_moose.sh" # Script .sh
moose_dir = '/Users/tmr96/projects/my_files' 
moose_app = 'solid_mechanics' # Moose app to run (assumed to be in moose_dir)
convert_path = '/Users/tmr96/projects/my_files/images/convert_moose.i' # Moose input files
destination_moose = working_dir + '/moose_output'  # Where to store the images converted by Moose
destination_coreform = working_dir + '/coreform_output' # Where to store the meshes produced by Coreform
journals_path = working_dir + '/journals' # Where all the journal files are going to be created
meshes_path = destination_coreform # Run the simulation on this folder
to_run_path = working_dir + '/to_run' # Moose scripts to run on the meshes
run_script_path = working_dir + '/run_moose.sh' # Runs the Moose scripts
results_path = working_dir + '/simulation_results' # Where to save the results
consistent_path = working_dir + '/consistent_tensors'  # Where the consistent tensors are stored.
imgchar_path = working_dir + '/images_characteristics' # Where the characteristics of the images are stored
errorpropag_path = working_dir + '/error_propag' # Where the results computation of the error propagation are stored
graph_path = working_dir + '/graph_results' # Where the graphs of the error propag are going to be saved
graphstiffness_path = working_dir + '/graph_stiffness' # Where the graphs of the stiffness as a function of the moments are going to be saved
extracted_path = working_dir + '/extracted_inclusions'

cutoff_input = 160

# Solidity is removed for the moment
descriptors_name = ['Aspect ratio', 'Extent', 'Size', 'Orientation', 'Solidity'] # List of descriptors used. Be careful with the order
descriptors_max = {'aspect_ratio' : 1, 'extent' : 1, 'size': 50, 'orientation' : 180, 'solidity' : 1, 's2' : 1, 'cluster' : 1, 'lp' : 1, 'cld angular' : 1}  # The maximum value of a descriptor. Useful for plotting. Maybe replace it with a case by case max search. 
# Max size to be discussed
descriptor_sampling = 20 # Number of classes in which to divide each descriptor
correlation_descriptors = ['S2', 'Cluster', 'Lineal Path', 'Angular Chord Length Distribution']

# It is necessary to specify the cutoff beforehand if we want to calculate the stiffness as a function of S2 in a given direction


def display_working_dirs() :
    text = 'Working directory : ' + working_dir + '\n\nImages folder : ' + images_path + '\n\nMOOSE application : ' + moose_dir + '/' + moose_app + '\n\n' \
        'The following files are supposed to be in the Working directory : convert_moose.sh, run_moose.sh.\n\n'\
            'The following directories are used in the Working directory : moose_output, coreform_output, journals, to_run, simulation_results, consistent_tensors, images_characteristics, error_propag, graph_stiffness, graph_results, extracted_inclusions.\n\n' \
                'The images folder has to be there because otherwise MOOSE can\'t detect the images (I don\'t know why).\n'
    popup(text)
    return


def init_dirs() :
    create_dir(destination_moose)
    create_dir(destination_coreform)
    create_dir(results_path)
    create_dir(consistent_path)
    create_dir(imgchar_path)
    create_dir(journals_path)
    create_dir(errorpropag_path)
    create_dir(errorpropag_path + '/pairwise')
    create_dir(errorpropag_path + '/groundtruth')
    create_dir(graph_path)
    create_dir(graphstiffness_path)
    create_dir(extracted_path)

# Returns the list of images in a folder
def images_folder(images_path) :
    return [os.path.basename(image) for image in glob.glob(images_path + '/*.png')] # Retrieves all the names of the images in the images folder

#Returns the list of meshes in a folder
def mesh_folder(meshes_path) :
    return [os.path.basename(mesh) for mesh in glob.glob(meshes_path + '/*.e')] 

#Returns the name of a file without the extension
def rm_ext(file) :
    name, ext = os.path.splitext(file)
    return name

# Returns the name of a file with a new extension
def change_ext(file, ext_name) :
    tmp, _ = os.path.splitext(file) # Name without extension
    new_name = tmp + ext_name
    return new_name

# Creates a file if it does not exist
def create_file(file) :
    if not os.path.exists(file) :
        with open(file, 'w') as f :
            f.write('')


# Creates a directory if it does not exist
def create_dir(path) :
    if not os.path.exists(path) :
        os.makedirs(path)

# Returns the number of files with a given extension in a given directory
def count_files(dir, ext) :
    nb_files = 0
    if not os.path.exists(dir) :
        return 0
    dirs = os.listdir(dir)
    if '.DS_Store' in dirs :
        dirs.remove('.DS_Store')
    for e in dirs :
        path = os.path.join(dir, e)
        if os.path.isdir(path) :
            nb_files += count_files(path, ext)
        else :
            if e.endswith(ext) :
                nb_files += 1
    return nb_files


# Counts the number of non empty directories in a directory
def count_non_empty_dirs(dir) :
    dirs = [d for d in os.listdir(dir) if d != '.DS_Store' and os.path.isdir(d)]
    nb = 0
    for d in dirs :
        files = [f for f in os.listdir(d) if os.path.isfile(f)]
        if len(files) > 0 :
            nb += 1
    return nb

# Retrieves all files with a certain extension in a folder and its children
def retrieve_files_recursive(dir, ext) :
    files = [os.path.basename(f) for f in glob.glob(dir + '/*' + ext)]
    for e in os.listdir(dir) :
        path = os.path.join(dir, e)
        if os.path.isdir(path) :
            files_child = retrieve_files_recursive(path)
            files += files_child
    return files    


# Removes any element of list1 that already is in list2 
def filter_list(list1, list2) :
    index = []
    \
    for x, element in enumerate(list1) :
        for k in list2 :
            if rm_ext(element) == rm_ext(k) :
                if not x in index :
                    index.append(x)
    
    index.sort()
    
    for i in reversed(index) :
        list1.pop(i)
      
    return list1
  

# Edits the MOOSE input files to convert images to meshes
def write_input(image) :
    with open(convert_path, 'r+') as convert :
            # The lines to modify have index 4 and 9
            lines = convert.readlines()
            lines[4] = '\t\tfile = ' + images_path + '/' + image + '\n'
            lines[9] = '\t\tfile = ' + images_path + '/' + image + '\n'
            lines = ''.join(lines)
            convert.seek(0)
            convert.write(lines)


# Edits the shell file to move MOOSE output to the correct destination
def edit_convert_script(image) :
    with open(convert_script_path, 'r+') as script :
        lines = script.readlines()
        result = change_ext(image, '.e')
        lines[5] = 'mv rename_moose_in.e ' + destination_moose + '/' + result + '\n'
        lines = ''.join(lines)
        script.seek(0)
        script.write(lines)
        

# Starts the shell to convert images to meshes        
def convert_moose() :
    # Allow the script to run 
    os.chmod(convert_script_path, 0o755) 
    process = subprocess.run(["bash", convert_script_path])
        

# Creates all the journal files that Coreform Cubit will use
def journal(image) :
    name = change_ext(image, '.jou')
    mesh = change_ext(image, '.e')
    with open(journals_path + '/' + name, 'w') as journal :
        journal.seek(0)
        journal.write('reset\n')
        journal.write('import mesh geometry \"' + destination_moose + '/' + mesh + '\" feature_angle 135 merge\n')
        journal.write('delete mesh\n')
        journal.write('surface all size auto factor 5\n')
        journal.write('surface all scheme trimesh\n')
        journal.write('mesh surface all\n')
        journal.write('block all element type TRI6\n')
        journal.write('export mesh \"' + destination_coreform + '/' + mesh + '\" overwrite')


# Creates the .txt file to copy in Cubit command line
def to_copy(image) :
    name = change_ext(image, '.jou')
    with open(working_dir + '/to_copy.txt', 'a') as f :
        f.write('cubit.cmd(\"Playback \\\"' + journals_path + '/' + name + '\\\"\")\n')


# Not called from the application
def prepapre_to_copy() :
    images = images_folder(images_path)
    for image in images :
        to_copy(image)

       
# Main function to convert images to meshes (called by a button)
def mesh_images(images) :
    
    time_start = time.time()       
    for image in images :
        write_input(image)
        edit_convert_script(image)
        convert_moose()
        journal(image)
        to_copy(image)
   
    time_end = time.time()
    popup('Meshing done !', time_end - time_start)


def mesh_images_init(fen) :
    if os.path.exists(working_dir + '/to_copy.txt') :
        os.remove(working_dir + '/to_copy.txt')
        
    overwrite_bool = overwrite(fen)
        
    images = images_folder(images_path)
    meshes = mesh_folder(destination_moose)
   
    if not overwrite_bool :
        images = filter_list(images, meshes)
        
    p = Thread(target=mesh_images, args=(images,))
    p.start()

# Edits the MOOSE input files to run the simulations on the meshes
def edit_run_script(mesh) :
    scripts = [os.path.basename(script) for script in glob.glob(to_run_path + '/*.i')]
    mesh_name = rm_ext(mesh)
    for script in scripts :
        script_name = rm_ext(script)
        with open(to_run_path + '/' + script, 'r+') as s :
            lines = s.readlines()
            lines[8] = 'file = \'' + meshes_path + '/' + mesh + '\'\n'
            lines[153] = 'file_base = ' + results_path + '/' + mesh_name + '/' + mesh_name + '_' + script_name + '\n'
            lines[159] = 'file_base = ' + results_path + '/' + mesh_name + '/' + mesh_name + '_' + script_name + '\n'
            lines = ''.join(lines)
            s.seek(0)
            s.write(lines)
    return


# Starts the shell to run the simulations
def run_moose() :
    os.chmod(run_script_path, 0o755) 
    process = subprocess.run(["bash", run_script_path])
    

# Solves Y = A X for A by providing n2 equtions (n loading directions * nb of components)
def solve_system_2D(y_xx, y_yy, y_xy, x_xx, x_yy, x_xy) :  
    Y = y_xx + y_yy + y_xy 
   
    B = np.array([[x_xx[0], x_yy[0], x_xy[0]],             # We want to write the system in the form M A' = R
                  [x_xx[1], x_yy[1], x_xy[1]],             # Where A' is the column vector of the coefficients of A
                  [x_xx[2], x_yy[2], x_xy[2]]])
     
    B_inv = np.linalg.inv(B)   
    M_inv = np.block([[B_inv, np.zeros((3,6))],
                  [np.zeros((3,3)), B_inv, np.zeros((3,3))],
                  [np.zeros((3,6)), B_inv]])
    
    R = np.dot(M_inv, Y)
    
    
    A = np.zeros((3,3))
    
    for i in range(3) :
        for j in range(3) :
            A[i][j] = R[j+3*i]
    return A


# Check if the tensors obtained are consistent (symmetric and inverse of each other)
def is_consistent(t1, t2, e) :
    tolerance_symmetry = 0.05
    tolerance_importance = 0.01  # If the 21 coefficients are too small compared to the others, the symmetry is not checked for them
    tolerance_bc = 0.1
    shape = t1.shape
    if shape != t2.shape :
        return (False, 'Incoherent shapes')
    
    tensors = [t1, t2]
    
    cumulative_symmetry_error = [0,0]
    return_code = 0 # Tells which tensor to keep
    
    for k, t in enumerate(tensors) :
        max = np.max(t)
        for i in range(shape[0]) :
            for j in range(shape[1]) :
                if t[i][j] / max  < tolerance_importance :
                    continue 
                symmetry_error = (t[i][j] - t[i][j])/t[i][j]
                cumulative_symmetry_error[k] += symmetry_error
                if symmetry_error > tolerance_symmetry :
                    return (False, 'Symmetry error on one of the tensors', return_code)
            
    for i in range(shape[0]) :
        for j in range(shape[1]) :
            if e[i][j] > tolerance_bc :
                    for t in tensors :
                        max = np.max(t)
                        if t[i][j] / max  > tolerance_importance :
                            return (False, 'Difference between BCs analysis is too important', return_code)
    
    
    if cumulative_symmetry_error[0] < cumulative_symmetry_error[1] :
        return_code = 1
    else :
        return_code = 2
    
    return (True, '', return_code)


# Computes the stiffness of a mesh and copy the tensor in consistent_tensors if the results are consistent
def compute_cell(dir) :
    path = os.path.join(results_path, dir)
    files = [file for file in glob.glob(path + '/*.csv')] 
    
    if len(files) < 6 :
        popup('Not enough csv files for mesh ' + dir)
        return
    
    eps_xx_t = [0,0,0] # Average of eps_xx depending on the loading conditions (sigma_xx, sigma _yy, sigma_xy) under homogeneous traction BCs
    eps_xy_t = [0,0,0]
    eps_yy_t = [0,0,0]
    sig_xx_t = [0,0,0]
    sig_xy_t = [0,0,0]
    sig_yy_t = [0,0,0]
    
    eps_xx_d = [0,0,0]
    eps_xy_d = [0,0,0] 
    eps_yy_d = [0,0,0]
    sig_xx_d = [0,0,0]
    sig_xy_d = [0,0,0]
    sig_yy_d = [0,0,0]
    
    for file in files :
        
        name = os.path.basename(file)
        name = rm_ext(name)
        name = name.split('_')
            
            
        with open(file) as f :
            lecteur = csv.reader(f)
            
            ligne = list(lecteur)[1]
            ligne = list(ligne)
            ligne = [float(l) for l in ligne]
            
            loading_dir = name[2]
            index = 0
            if loading_dir == '11' :
                index = 0
            elif loading_dir == '22' :
                index = 1
            elif loading_dir == '21' :
                index = 2
            else :
                popup('Results reading eror')
                return
            
            loading_case = name[1]
            if loading_case == 'sigma' :
                eps_xx_t[index] += ligne[1]
                eps_xy_t[index] += ligne[2]
                eps_yy_t[index] += ligne[3]
                sig_xx_t[index] += ligne[7]
                sig_xy_t[index] += ligne[8]
                sig_yy_t[index] += ligne[9]
            elif loading_case == 'epsilon' :
                eps_xx_d[index] += ligne[1]
                eps_xy_d[index] += ligne[2]
                eps_yy_d[index] += ligne[3]
                sig_xx_d[index] += ligne[7]
                sig_xy_d[index] += ligne[8]
                sig_yy_d[index] += ligne[9]
            
    
    tolerance = 1e-12
    
    eps_xx_t = [nb if abs(nb) > tolerance else 0 for nb in eps_xx_t]
    eps_xy_t = [nb if abs(nb) > tolerance else 0 for nb in eps_xy_t]
    eps_yy_t = [nb if abs(nb) > tolerance else 0 for nb in eps_yy_t]
    sig_xx_t = [nb if abs(nb) > tolerance else 0 for nb in sig_xx_t]
    sig_xy_t = [nb if abs(nb) > tolerance else 0 for nb in sig_xy_t]
    sig_yy_t = [nb if abs(nb) > tolerance else 0 for nb in sig_yy_t]
    
    eps_xx_d = [nb if abs(nb) > tolerance else 0 for nb in eps_xx_d]
    eps_xy_d = [nb if abs(nb) > tolerance else 0 for nb in eps_xy_d]
    eps_yy_d = [nb if abs(nb) > tolerance else 0 for nb in eps_yy_d]
    sig_xx_d = [nb if abs(nb) > tolerance else 0 for nb in sig_xx_d]
    sig_xy_d = [nb if abs(nb) > tolerance else 0 for nb in sig_xy_d]
    sig_yy_d = [nb if abs(nb) > tolerance else 0 for nb in sig_yy_d]
    
    
    
    C_d = solve_system_2D(sig_xx_d, sig_yy_d, sig_xy_d, eps_xx_d, eps_yy_d, [e * 2 for e in eps_xy_d])
    
    S = solve_system_2D(eps_xx_t, eps_yy_t, [e * 2 for e in eps_xy_t], sig_xx_t, sig_yy_t, sig_xy_t)
    C_t = np.linalg.inv(S)
    
    E = np.zeros((3,3))
    
    for i in range(3) :
        for j in range(3) :
            E[i][j] = (C_d[i][j] - C_t[i][j]) / C_d[i][j]
    
    txt_name = name[0] + '_effective_stiffness'
    
    tensors = [C_t, C_d, E]
    
    txt_path = path + '/' + txt_name + '.txt'
    
    create_file(txt_path)
    
    with open(txt_path, 'w') as f :
        f.seek(0)
        for t in tensors :
            for i in range(t.shape[0]) :
                for j in range(t.shape[1]) :
                    f.write(str(t[i][j]) + '\t')
                f.write('\n')
            f.write('\n----------\n')
    
    consistency , error_txt, to_keep = is_consistent(C_t, C_d, E)
    
            
    if consistency :
        '''
        average_stiffness = np.zeros((3,3))
        shape = average_stiffness.shape
        for i in range(shape[0]) :
            for j in range(shape[1]) :
                average_stiffness[i][j] = (C_t[i][j] + C_d[i][j])/2
        '''
        if to_keep == 1 :
            tensor_to_keep = C_t
        elif to_keep == 2 :
            tensor_to_keep = C_d
        shape = tensor_to_keep.shape
        consistent_tensor_name = consistent_path + '/' + txt_name + '.txt'
        create_file(consistent_tensor_name)
        with open(consistent_tensor_name, 'w') as f :
            f.seek(0)
            for i in range(shape[0]) :
                for j in range(shape[1]) :
                    f.write(str(tensor_to_keep[i][j]) + '\t')
                f.write('\n')
            f.write('\n')
    else :
        with open(txt_path, 'a') as f :
            f.write(error_txt +'\n')             
       
# Main function for running the simulations on the meshes (called by a button)        
def run_simulations(meshes) :
    
    time_start = time.time()
    
    nb_meshes = len(meshes)
    nb_computed = 0
    
    
    for mesh in meshes :
        edit_run_script(mesh)
        run_moose()
        nb_computed += 1
    
    
    # I think this test does not work. Should look into it.
    while nb_computed < nb_meshes : # If we do not do this, the python script continues even when the shell is not done
        time.sleep(1)
        nb_computed = count_non_empty_dirs(results_path)
    
    time_end = time.time()
    popup('Simulations done !', time_end - time_start)



def run_simulations_init(fen) :
    meshes = mesh_folder(meshes_path)
    
    results = os.listdir(results_path)
    if '.DS_Store' in results :
        results.remove('.DS_Store')
    done = []
    for r in results :
        name = results_path + '/' + r
        nb_csv = count_files(name, '.csv')
        if nb_csv == 6 :
            done.append(r)
    
    overwrite_bool = overwrite(fen)
    if not overwrite_bool :
        meshes = filter_list(meshes, done)
        
    p = Thread(target=run_simulations, args=(meshes,))
    p.start()    
        
    
 
# Calculate the stiffness and creates the consistent tensors for ever mesh  
def compute_results(fen) :
    dir_content = os.listdir(results_path)
    dirs = [dir for dir in dir_content if os.path.isdir(os.path.join(results_path, dir)) and dir != '.DS_Store']
    
    computed = [dir for dir in dirs if count_files(dir, '.txt') > 0]
    
    overwrite_bool = overwrite(fen)
    if not overwrite_bool :
        dirs = filter_list(dirs, computed)
    
    
    for dir in dirs :
        compute_cell(dir)
    


def hsw() :
    tensor_fullname = filedialog.askopenfilename(title='Select a consistent tensor', initialdir=consistent_path)
    
    tensor_name = os.path.basename(tensor_fullname)
    
    if not 'effective_stiffness' in tensor_name :
        popup('Invalid input !')
        return
    
    tensor = read_tensor(tensor_name)
    
    name = tensor_name.split('_')[0]
    vf_name = os.path.join(imgchar_path, name, name + '_characteristics.txt')
    
    v1 = read_vf(vf_name)
    
    # Index 1 is for the inclusion and Index 0 is for the matrix.
    # The inclusions are assumed to be stiffer. Otherwise see Walpole 1966.
    # Young's moduli are given in MPa.
    
    E0, nu0 = 7000, 0.3
    K0 = E0/3/(1-2*nu0)
    G0 = E0/2/(1+nu0)
    v0 = 1-v1
    
    E1, nu1 = 70000, 0.22  
    K1 = E1/3/(1-2*nu1)
    G1 = E1/2/(1+nu1)
    
    K_top = K1 + v0/( 1/(K0-K1) + 3*v1/(3*K1+4*G1) )
    K_bottom = K0 + v1/( 1/(K1-K0) + 3*v0/(3*K0+4*G0) )
    
    G_top = G1 + v0/( 1/(G0-G1) + 6*(K1+2*G1)*v1/(5*G1*(3*K1+4*G1)) )
    G_bottom = G0 + v1/( 1/(G1-G0) + 6*(K0+2*G0)*v0/(5*G0*(3*K0+4*G0)) )
    
    C11 = float(tensor[0][0])
    C12 = float(tensor[0][1])
    nu_t = C12/(C11+C12)
    E_t = (1+nu_t)*(1-2*nu_t)*(C11 + C12)
    Kt = E_t/3/(1-2*nu_t)
    Gt = E_t/2/(1+nu_t)
    
    if K_bottom <= Kt <= K_top and G_bottom <= Gt <= G_top :
        popup('Tensor ' + name + ' lies within Hashin-Shtrikman\'s bounds')
    else :
        popup('Tensor ' + name + ' does not lie within Hashin-Shtrikman\'s bounds')
    
def help() :
    return   


# Returns the repartition of a list as a Gaussian-like function
def distribution_descriptor(list, max) :
    nb_steps = descriptor_sampling  # Number of steps to discretize the descriptors. A number too big results in a list with many zeros
    n = len(list)
    
    distribution = nb_steps*[0]
    
    step = max / nb_steps # We assume that the lists start from 0
    
    for i in range(nb_steps) :
        for e in list :
            if i*step <= e < (i+1)*step :
                distribution[i] += 1   
    
    distribution = [i/n for i in distribution]

    return distribution


# Removes all the zeros before plotting
def clean_distribution(list_input, max) :
    x = np.linspace(0, max, descriptor_sampling)
    index = [i for i,e in enumerate(list_input) if e == 0]
    for i in reversed(index) :
        list_input = np.delete(list_input, i)
        x = np.delete(x, i)
    return list(x), list(list_input)      # A descriptor should not be separated from the values where it has been calculated


# Plots the distribution as points
def plot_distribution() :
    name = filedialog.askopenfilename(title='Choose a distribution file', initialdir=imgchar_path)
    if not ('distributions' in name) :
        popup('Invalid input')
        return
    
    nb_descriptors = len(descriptors_name)
    
    distributions = read_distributions(name)
    
    name = os.path.basename(name)
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    
    aspect_ratio_max = descriptors_max['aspect_ratio']
    extent_max = descriptors_max['extent']
    size_max = descriptors_max['size']
    orientation_max = descriptors_max['orientation']
    
    # Find a way to improve this (standadize the names as key to dictionnary)
    # [descriptor][moments][order 0]
    x_aspect_ratio, aspect_ration_distr = clean_distribution(distributions[0][1][0], aspect_ratio_max)  
    x_extent, extent_distr = clean_distribution(distributions[1][1][0], extent_max)
    x_size, size_distr = clean_distribution(distributions[2][1][0], size_max)
    x_orientation, orientation_distr = clean_distribution(distributions[3][1][0], orientation_max)
    
    distributions_plot = [['Aspect ratio', x_aspect_ratio, aspect_ration_distr],['Extent', x_extent, extent_distr], ['Size', x_size, size_distr], ['Orientation', x_orientation, orientation_distr]]
    
    
    for k in range(nb_descriptors) :  
        x = distributions_plot[k][1]
        y = distributions_plot[k][2]
        
        
        h = k // 2
        if h == 0 :
            g = k-h
        elif h == 1 :
            g = k-h-1
        
        
        axs[h, g].plot(x, y, 'r+')
        axs[h, g].set_title(descriptors_name[k])
    plt.tight_layout()
    fig.suptitle(name.split('_')[0])
    plt.show()  
    

# Plots the histograms of the descriptors of one image
def plot_hist() :
    name = filedialog.askopenfilename(title='Choose a characteristics file', initialdir=imgchar_path)
    if not ('characteristics.txt' in name) :
        popup('Invalid input')
        return
    
    nb_descriptors = len(descriptors_name)
    descriptors = []
    basename = os.path.basename(name)
    
    with open(name, 'r') as f :
        lines = f.readlines()
        lines.pop(0)
        for i, l in enumerate(lines) :
            if i % 2 == 1 :
                descriptors.append(eval(l))
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    for k in range(nb_descriptors) :  
        descriptors_numeric = np.array(descriptors[k], dtype = float)
        counts, bins = np.histogram(descriptors_numeric, bins=descriptor_sampling)
        counts = counts.astype(float)
        counts /= np.sum(counts)
        
        h = k // 2
        if h == 0 :
            g = k-h
        elif h == 1 :
            g = k-h-1
        
        
        axs[h, g].stairs(counts, bins)
        axs[h, g].set_title(descriptors_name[k])
    plt.tight_layout()
    fig.suptitle(basename.split('_')[0])
    plt.show()  
    

def plot_polar() :
    name = filedialog.askopenfilename(title='Choose a characteristics file', initialdir=imgchar_path)
    if not ('characteristics.txt' in name) :
        popup('Invalid input')
        return
    
   
    angles = []
    
    
    with open(name, 'r') as f :
        lines = f.readlines()
        lines.pop(0)
        for i, l in enumerate(lines) :
            if 'Orientation' in l :
                angles = eval(lines[i+1])
                break
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 16))
    
    counts, bins = np.histogram(angles, bins=descriptor_sampling)
    
    counts = counts.astype(float)
    counts /= np.sum(counts)
    theta = np.linspace(0, radians(descriptors_max['orientation']), descriptor_sampling)
    bins_middle = [(bins[i] + bins[i+1])/2 for i in range(len(bins) - 1)]
    bins_middle = [radians(e) for e in bins_middle]
    ax.bar(bins_middle, counts, width = 0.1, color='blue')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    title = os.path.basename(name).split('_')[0] + ' : Orientation'
    plt.title(title)
    plt.show()

# Computes the characteristics of one image
def img_char(image_arg) :
    image = cv2.imread(images_path + '/' + image_arg, cv2.IMREAD_UNCHANGED)
    shapeMask = cv2.inRange(image, 0, 0)
    contours_init, hierarchy = cv2.findContours(shapeMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Find the contours of the inclusions by passing shapeMask as argument
    
    image_filtered = np.zeros(image.shape, np.uint8)
    for number, cnt in enumerate(contours_init) : 
        area = cv2.contourArea(cnt)
        if area != 0 :
            cv2.drawContours(image_filtered, contours_init, number, (255,255,255), cv2.FILLED)
            
    contours_draw, _ = cv2.findContours(image_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(image.shape, np.uint8)
    mask.fill(255)
    thisMask = mask.copy()
    cv2.drawContours(thisMask, contours_draw, -1, (0, 0, 0), cv2.FILLED)
    
    # Satistical descripors we want to compute
    if len(thisMask.shape) == 3 :
        volume_fraction_contours = np.count_nonzero(thisMask[:,:,0] == 0) / (thisMask.shape[0]*thisMask.shape[1])
    elif len(thisMask.shape) == 2 :
        volume_fraction_contours = np.count_nonzero(thisMask == 0) / (thisMask.shape[0]*thisMask.shape[1])
    else :
        popup('Error computing the characteristics of image ' + image_arg)
        return
    aspect_ratio = []
    extent = []
    orientation = []
    size = []
    solidity = []
    
    nb_solidity = 0
    nb_ar = 0
    
    for cnt in contours_draw :
        
        x,y,w,h = cv2.boundingRect(cnt)
        
        # Computing the extent
        area = cv2.contourArea(cnt)
        extent_cnt = float(area) / w /h
        
        # Computing the solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        try :
            solidity_cnt = float(area)/hull_area
        except ZeroDivisionError :
            nb_solidity += 1
            solidity_cnt = 0
            
        # Computing the orientation and the size
        # Need to investigate if the choice of the interpolation figure does a lot of difference
        if len(cnt) > 4 :
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
            try :
                aspect_ratio_cnt = MA / ma
                size_cnt = ma
            except ZeroDivisionError :
                nb_ar += 1
                
                (x,y),(MA,ma),angle = cv2.minAreaRect(cnt) 
                size_cnt = sqrt(MA*ma)
                
                x,y,w,h = cv2.boundingRect(cnt)
                aspect_ratio_cnt = float(w)/h
                
                
        else :
            (x,y),(MA,ma),angle = cv2.minAreaRect(cnt) # if fitEllipse does not work
            size_cnt = sqrt(MA*ma)
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio_cnt = float(w)/h
        
        aspect_ratio.append(aspect_ratio_cnt)
        extent.append(extent_cnt)
        size.append(size_cnt)
        orientation.append(angle)
        solidity.append(solidity_cnt)
    
    #print(len(contours_draw), nb_solidity, nb_ar)
     
    #  I make sure that every orientation value is positive
    orientation = [e + 180 if e < 0 else e for e in orientation]
   
    name = rm_ext(image_arg)
    create_dir(imgchar_path + '/' + name)
    txt_name = imgchar_path + '/' + name + '/' + name + '_characteristics.txt'
    with open(txt_name, 'w') as f :
        f.write('Number of inclusions : ' + str(len(contours_draw)) + '\tVolume fraction : ' + str(volume_fraction_contours) + '\n')
        f.write('Aspect ratio :\n')
        f.write(str(aspect_ratio) + '\n')
        f.write('Extent :\n')
        f.write(str(extent) + '\n')
        f.write('Size :\n')
        f.write(str(size) + '\n')
        f.write('Orientation :\n')
        f.write(str(orientation) + '\n')
        f.write('Solidity :\n')
        f.write(str(solidity) + '\n')
    
    
    
    
    
    # Max values of the descriptors
    aspect_ratio_max = descriptors_max['aspect_ratio']
    extent_max = descriptors_max['extent']
    size_max = descriptors_max['size']
    orientation_max = descriptors_max['orientation']
    solidity_max = descriptors_max['solidity']
    
    
    # In the end, I make the choice not to interpolate the distributions because the Wasserstein distance takes as parameters two lists anyways
    aspect_ratio_distr = distribution_descriptor(aspect_ratio, aspect_ratio_max)
    extent_distr = distribution_descriptor(extent, extent_max)
    size_distr = distribution_descriptor(size, size_max)
    orientation_distr = distribution_descriptor(orientation, orientation_max)
    solidity_distr = distribution_descriptor(solidity, solidity_max)
    
    aspect_ratio_moments = compute_moments(aspect_ratio_distr, 'aspect_ratio')
    extent_moments = compute_moments(extent_distr, 'extent')
    size_moments = compute_moments(size_distr, 'size')
    orientation_moments = compute_moments(orientation_distr, 'orientation')
    solidity_moments = compute_moments(solidity_distr, 'solidity')
    
    txt_name = imgchar_path + '/' + name + '/' + name + '_distributions.txt'
    with open(txt_name, 'w') as f :
        f.write('Aspect ratio :\n')
        f.write(str(aspect_ratio_moments) + '\n')
        f.write('Extent :\n')
        f.write(str(extent_moments) + '\n')
        f.write('Size :\n')
        f.write(str(size_moments) + '\n')
        f.write('Orientation :\n')
        f.write(str(orientation_moments) + '\n')
        f.write('Solidity :\n')
        f.write(str(solidity_moments) + '\n')
      

# Computes the characteristics of the images 
def compute_img_char(fen) :
    
    images = images_folder(images_path)
    
    char = os.listdir(imgchar_path)
    if '.DS_Store' in char :
        char.remove('.DS_Store')
    
    overwrite_bool = overwrite(fen)
    
    if not overwrite_bool :
        images = filter_list(images, char)
    
    
    
    time_start = time.time()
    
    for image in images :
        img_char(image) 
    
    time_end = time.time()
    popup('Images characteristics computed', time_end - time_start)


def compute_mean(distr, descriptor_name) :
    total = 0
    max = descriptors_max[descriptor_name]
    n = len(distr)
    step = max / descriptor_sampling
    for i in range(n) :
        total += (i + 0.5) * step * distr[i]

    return total

def compute_moments_sup(distr,descriptor_name, mean, order) :
    total = 0
    max = descriptors_max[descriptor_name]
    n = len(distr)
    step = max / descriptor_sampling
    for i in range(n) :
        total += ((i + 0.5) * step - mean)**order * distr[i]

    return total

# Returns a list of the distribution and its three first moments
def compute_moments(distr, descriptor_name) :
    mean = compute_mean(distr, descriptor_name)
    std = compute_moments_sup(distr, descriptor_name, mean, 2)
    skewness = compute_moments_sup(distr, descriptor_name, mean, 3)
    return [distr, mean, std, skewness]



# Reads the descriptors pdfs from the .txt file
# Returns the list of each descriptors represented by a list of its name and the list of its moments (0th one being a list)
def read_distributions(full_name) :
    distributions = []
    txt = 'undefined'
    with open(full_name, 'r') as f :
        lines = f.readlines()
        tmp = [0,0]
        for i,l in enumerate(lines) :
            if i%2 == 0 :
                txt = l
                tmp[0] = txt.replace(' :\n', '')
            elif i%2 == 1 :
                tmp[1] = eval(l)
                distributions.append(tmp)
                tmp = [0,0]
    return distributions



def read_vf(full_name) :
    vf = 0
    with open(full_name, 'r') as f :
        lines = f.readlines()
        split1 = lines[0].split('\t')
        split2 = split1[1].split(':')[1]
        split2.replace(' ', '')
        vf = float(split2)
    return vf



def read_vf_all() :
    images = os.listdir(imgchar_path)
    if '.DS_Store' in images :
        images.remove('.DS_Store')
    vf = []
    for e in images :
        name = e + os.path.basename(e) + '_characteristics.txt'
        vf.append(read_vf(name))
    return vf


def compute_s2_r(full_name) :
    
    image = Image.open(full_name)
    
    basename = os.path.basename(full_name)
    
    imarray = np.expand_dims(np.array(image), axis=0)
    data = PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0).transform(imarray)
    auto_correlation = TwoPointCorrelation(
        periodic_boundary = False,
        cutoff = 160, # cutoff length should depend on the dimensions of my image
        correlations=[(0,0)]
    ).transform(data)
    
    
    probsS2, radiiS2 = paircorr_from_twopoint(auto_correlation, cutoff_r = None, interpolate_n = None)
    
    path_probs = os.path.join(imgchar_path, rm_ext(basename), rm_ext(basename) + '_s2_r.npy')
    
    np.save(path_probs, probsS2)
    
    path_rad = os.path.join(imgchar_path, rm_ext(basename), rm_ext(basename) + '_rads2.txt')
    create_file(path_rad)
    with open(path_rad, 'w') as f :
        f.write(str(list(radiiS2)))
    
 
 
def compute_s2(full_name) :
    image = Image.open(full_name)
    
    basename = os.path.basename(full_name)
    basename_rm = rm_ext(basename)
    
    imarray = np.expand_dims(np.array(image), axis=0)
    data = PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0).transform(imarray)
    
    auto_correlation = TwoPointCorrelation(
        periodic_boundary = False,
        cutoff = cutoff_input, # cutoff length should depend on the dimensions of my image
        correlations=[(0,0)]
    ).transform(data)   
   
    np.save(os.path.join(imgchar_path, basename_rm, basename_rm + '_s2.npy'), auto_correlation)
    

def compute_lp(full_name) :
    img = cv2.imread(full_name, cv2.IMREAD_UNCHANGED)
    shapeMask = cv2.inRange(img, 0, 0)
    L = GooseEYE.L((shapeMask.shape[0], shapeMask.shape[1]), shapeMask)
    
    basename = os.path.basename(full_name)
    basename_rm = rm_ext(basename)
    
    np.save(os.path.join(imgchar_path, basename_rm, basename_rm + '_lp.npy'), L)


def rotate_contour(cnt, angle_in_degrees):
    M = cv2.moments(cnt)
    assert M['m00'] != 0., "Cannot compute inclusion's centroid, exiting rotation function"

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    # Move the contour's centroid to the origin of the coordinate system
    cnt_norm = cnt - [cx, cy]
    # Form the rotation matrix
    rot_2D = np.zeros(shape = (2,2))
    rot_2D[0,0] = cos(radians(angle_in_degrees))
    rot_2D[0,1] = sin(radians(angle_in_degrees))
    rot_2D[1,0] = -sin(radians(angle_in_degrees))
    rot_2D[1,1] =  cos(radians(angle_in_degrees)) 
    # Rotate the contour
    coords = cnt_norm[:, 0, :] # first and last column are the x, y coords
    # Apply the rotation
    cnt_rotated = np.transpose(np.matmul(rot_2D, np.transpose(coords)))
    # Expand "cnt_rotated" to match the format of contours
    cnt_rotated = np.expand_dims(cnt_rotated, axis=1)
    # Move the contour back to its original position
    cnt_rotated = cnt_rotated + [cx, cy]
    # Adjust integer type of new position
    cnt_rotated = cnt_rotated.astype(np.int32) 
    
    return cnt_rotated


def compute_cld_angular(full_name) :
    image = cv2.imread(full_name, cv2.IMREAD_UNCHANGED)
    basename = os.path.basename(full_name)
    basename_rm = rm_ext(basename)
    
    # Padding
    top_border = 50
    bottom_border = 50
    left_border = 50
    right_border = 50

    border_color = [255, 255, 255]  # white color
    padded_image = cv2.copyMakeBorder(image, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=border_color)
    
    shapeMask = cv2.inRange(padded_image, 0, 0) 
    contours_draw, hierarchy = cv2.findContours(shapeMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    

    mask = np.zeros(padded_image.shape, np.uint8) 
    mask.fill(255)
    
    
    data_x_list = []
    
    for theta in np.arange(0,361): 
        theta_radians = np.radians(theta)

        plotRotMask = mask.copy()
        # ROTATE ALL THE INCLUSIONS BY THE SAME ANGLE
        for i in range(len(contours_draw)):  
            M = cv2.moments(contours_draw[i])
            if M['m00'] != 0:
                #print("index of contour to be rotated: " + str(i))
                cnt_rotated = rotate_contour(contours_draw[i], theta)
                cv2.drawContours(plotRotMask, [cnt_rotated], 0, (0, 0, 0), cv2.FILLED)
    
        shapeMask = cv2.inRange(plotRotMask, 0, 0) 

        # Draw the chords along a fixed direction, I will use axis = 1 as the fixed direction
        crds_x = ps.filters.apply_chords(im = shapeMask, spacing = 1, axis = 1, trim_edges = False)

        data_x = ps.metrics.chord_length_distribution(crds_x, bins = 20) # check if need to adjust bins
        data_x_list.append(data_x)

    np.save(os.path.join(imgchar_path, basename_rm, basename_rm + '_cld_angular.npy'), data_x_list)
        
        
        

def plot_cld_angular(data_x_list) :
    
    fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
    
    Rmax = 0
    R_cutoff = 55 # corresponding to Rmax or smaller, if I notice any outliers in my data 
    # If R_cutoff different than Rmax, my plotting of the line segments beyond the cutoff should be adjusted
    prob_max = 0
    pdf_cutoff = 0.18 # from prob max, would have to be adjusted based on the probability levels of the image being read
        
    for theta in np.arange(0,361): 
        theta_radians = np.radians(theta)

        data_x = data_x_list[theta]

        if  Rmax < (np.max(data_x.L) + np.mean(data_x.bin_widths)/2):
            Rmax = (np.max(data_x.L) + np.mean(data_x.bin_widths)/2)

        if prob_max < np.max(data_x.pdf):
            prob_max = np.max(data_x.pdf)
            
        probabilities = data_x.pdf

        # Create colormap based on probabilities
        norm = Normalize(vmin=0, vmax=pdf_cutoff)
        cmap = plt.get_cmap('viridis')
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Plot histogram bars with colors based on probabilities
        for i in range(len(data_x.pdf)):
            color = sm.to_rgba(probabilities[i])
            ax2.plot([theta_radians, theta_radians], [ data_x.bin_centers[i] - data_x.bin_widths[i]/2, data_x.bin_centers[i] + data_x.bin_widths[i]/2 ] , color = color, linewidth = 1.2, alpha = 0.8)
            
            
        if (np.max(data_x.L) + np.mean(data_x.bin_widths)/2) < R_cutoff:
            ax2.plot([theta_radians, theta_radians], [ (np.max(data_x.L) + np.mean(data_x.bin_widths)/2), R_cutoff ] , color = sm.to_rgba(0), linewidth = 1.2, alpha = 0.8)
            
        
    cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', pad=0.1) # what is the padding parameter?
    cbar.set_label('Probability')

    plt.title('Angularly Resolved Chord Length Distribution')
    plt.show()
    
    
def compute_cld_dir(full_name) :
    image = cv2.imread(full_name, cv2.IMREAD_UNCHANGED)
    
    basename = os.path.basename(full_name)
    basename_rm = rm_ext(basename)
    
    # Padding
    top_border = 50
    bottom_border = 50
    left_border = 50
    right_border = 50

    border_color = [255, 255, 255]  # white color
    padded_image = cv2.copyMakeBorder(image, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=border_color)
    
    shapeMask = cv2.inRange(padded_image, 0, 0) 
    contours_draw, hierarchy = cv2.findContours(shapeMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    mask = np.zeros(padded_image.shape, np.uint8) 
    mask.fill(255)
    
    data_x_list = []
    
    for theta in [0,90]: 
        plotRotMask = mask.copy()
        # ROTATE ALL THE INCLUSIONS BY THE SAME ANGLE
        for i in range(len(contours_draw)):  
            M = cv2.moments(contours_draw[i])
            if M['m00'] != 0:
                cnt_rotated = rotate_contour(contours_draw[i], theta)
                cv2.drawContours(plotRotMask, [cnt_rotated], 0, (0, 0, 0), cv2.FILLED) 
        shapeMask = cv2.inRange(plotRotMask, 0, 0) 
        
        crds_x = ps.filters.apply_chords(im = shapeMask, spacing = 1, axis = 1, trim_edges = False)
        
        data_x = ps.metrics.chord_length_distribution(crds_x, bins = 20) # check if need to adjust bins
        
        data_x_list.append(data_x)
        
    np.save(os.path.join(imgchar_path, basename_rm, basename_rm + '_cld_dir.npy'), data_x_list)


def plot_cld_dir(data_x_list) :
    
    
    for i, theta in enumerate([0,90]) :
        
        data_x = data_x_list[i]
            
        probabilities = data_x.pdf

        fig2, ax2 = plt.subplots()
        prob_cutoff = 0.14 # this has to be changed based on the maximum probability of the image being read

        # Create colormap based on probabilities
        norm = Normalize(vmin=0, vmax=prob_cutoff)
        cmap = plt.get_cmap('viridis')
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for i in range(len(data_x.pdf)):
            color = sm.to_rgba(probabilities[i])
            if i == 0:
                ax2.bar(data_x.L[i], data_x.pdf[i], width=data_x.bin_widths, color=color, edgecolor='k', alpha=0.5, label = r'$\theta$=' + str(theta) + '$^{\mathrm{o}}$')
            else:
                ax2.bar(data_x.L[i], data_x.pdf[i], width=data_x.bin_widths, color=color, edgecolor='k', alpha=0.5)
                

    ax2.legend()
    ax2.set_xlabel("Chord length")
    ax2.set_ylabel("PDF") 
    plt.show()
    
    

def compute_correlation(fen) :
    
    images = images_folder(images_path)
    if '.DS_Store' in images :
        images.remove('.DS_Store')
    
    images_s2 = images.copy()
    images_c = images.copy()
    images_c_r = images.copy()
    images_s2_r = images.copy()
    images_lp = images.copy()
    images_cld_angular = images.copy()
    images_cld_dir = images.copy()
    
    already_done_s2 = []
    already_done_c = []
    already_done_c_r = []
    already_done_s2_r = []
    already_done_lp = []
    already_done_cld_angular = []
    already_done_cld_dir = []
    
    directories = os.listdir(imgchar_path)
    
    if '.DS_Store' in directories :
        directories.remove('.DS_Store')
    
    for d in directories :
        for f in os.listdir(os.path.join(imgchar_path, d)) :
            if 's2.npy' in f :
                if not d + '.png' in already_done_s2 :
                    already_done_s2.append(d + '.png')
            elif 'cluster.npy' in f :
                if not d + '.png' in already_done_c :
                    already_done_c.append(d + '.png')
            elif 'cluster_r.npy' in f :
                if not d + '.png' in already_done_c_r :
                    already_done_c_r.append(d + '.png')
            elif 's2_r.npy' in f :
                if not d + '.png' in already_done_s2_r :
                    already_done_s2_r.append(d + '.png')
            elif 'lp' in f :
                if not d + '.png' in already_done_lp :
                    already_done_lp.append(d + '.png')
            elif 'cld_angular' in f :
                if not d + '.png' in already_done_cld_angular :
                    already_done_cld_angular.append(d + '.png')
            elif 'cld_dir' in f :
                if not d + '.png' in already_done_cld_dir :
                    already_done_cld_dir.append(d +'.png')
            
    overwrite_bool = True
    if len(already_done_s2) > 0 or len(already_done_c) > 0 :        
        overwrite_bool = overwrite(fen)
    
    if not overwrite_bool :
        images_s2 = filter_list(images_s2, already_done_s2)   
        images_c = filter_list(images_c, already_done_c)
        images_s2_r = filter_list(images_s2_r, already_done_s2_r)
        images_c_r = filter_list(images_c_r, already_done_c_r)
        images_lp = filter_list(images_lp, already_done_lp)
        images_cld_angular = filter_list(images_cld_angular, already_done_cld_angular)
        images_cld_dir = filter_list(images_cld_dir, already_done_cld_dir)
   
    time_start = time.time() 
       
    for image in images_s2 :
        compute_s2(os.path.join(images_path, image))
    for image in images_s2_r :
        compute_s2_r(os.path.join(images_path, image))
    for image in images_c_r :
        compute_cluster_r(rm_ext(image))
    for image in images_c :
        compute_cluster(rm_ext(image))
    for image in images_lp :
        compute_lp(os.path.join(images_path, image))
    for image in images_cld_angular :
        compute_cld_angular(os.path.join(images_path, image))
    for image in images_cld_dir :
        compute_cld_dir(os.path.join(images_path, image))
    
    time_end = time.time()
    popup('Computation done !', time_end - time_start)
    



def extract_inclusion(name) :
    image = images_path + '/' + name + '.png'
    path = extracted_path + '/'  + name
    
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    mask = cv2.inRange(img, 0, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for x, cnt in enumerate(contours) :
        tmp = np.zeros(mask.shape, np.uint8)
        tmp.fill(255)
        cv2.drawContours(tmp, [cnt], -1, (0, 0, 0), cv2.FILLED)
        cv2.imwrite(path + '/inclusion' + str(x) + '.png', tmp)
    
    
def extract_inclusion_all(fen) :
    
    images = images_folder(images_path)
    
    images_rm_ext = [rm_ext(i) for i in images]
    already_extracted = os.listdir(extracted_path)
    
    if '.DS_Store' in already_extracted :
        already_extracted.remove('.DS_Store')
    
    already_extracted = [os.path.basename(d) for d in already_extracted]
    overwrite_bool = overwrite(fen)
    
    
    if not overwrite_bool :
        images_rm_ext = filter_list(images_rm_ext, already_extracted)
    
    
    for image in images_rm_ext :
        create_dir(extracted_path + '/' + image)
        extract_inclusion(image)
    

def compute_cluster_r(basename) :
    img_dir = extracted_path + '/' + basename
    
    if not os.path.exists(img_dir) :
        popup('The inclusions have not been extracted for image ' + basename)
        return
    
    
    imageFiles = images_folder(img_dir)
    imageFiles = [extracted_path + '/' + basename + '/' + img for img in imageFiles]
    
    im_array = np.array(Image.open(imageFiles[0])) 
    im_array_all = np.zeros( shape = (len(imageFiles), im_array.shape[0], im_array.shape[1]) )

    for i in range(len(imageFiles)):
        im_array = np.array(Image.open(imageFiles[i]))
        im_array_all[i,:,:] = im_array
    

    data = PrimitiveTransformer(n_state = 2, min_=0.0, max_=1.0).transform(im_array_all)

    
    data_corr = TwoPointCorrelation(
        periodic_boundary = False,
        cutoff = 160, # I can tune the cutoff so that it is just sufficient for the computed two-point cluster function to drop to 0 at large distances
        correlations=[(0, 0)]  
    ).transform(data)
    
    summed_array = da.sum(data_corr, axis = 0)
    cluster_expanded = da.expand_dims(summed_array, axis = 0)
    probsC2, radiiC2 = paircorr_from_twopoint(cluster_expanded, cutoff_r = None, interpolate_n = None)
    
    path_probs = os.path.join(imgchar_path, basename, basename + '_cluster_r.npy')
 
    np.save(path_probs, probsC2)
    
    path_rad = os.path.join(imgchar_path, basename, basename + '_radcluster.txt')
    create_file(path_rad)
    with open(path_rad, 'w') as f :
        f.write(str(list(radiiC2)))


def compute_cluster(basename) :
    img_dir = extracted_path + '/' + basename
    
    if not os.path.exists(img_dir) :
        popup('The inclusions have not been extracted for image ' + basename)
        return
    
    
    imageFiles = images_folder(img_dir)
    imageFiles = [extracted_path + '/' + basename + '/' + img for img in imageFiles]
    
    im_array = np.array(Image.open(imageFiles[0])) 
    im_array_all = np.zeros( shape = (len(imageFiles), im_array.shape[0], im_array.shape[1]) )

    for i in range(len(imageFiles)):
        im_array = np.array(Image.open(imageFiles[i]))
        im_array_all[i,:,:] = im_array
    

    data = PrimitiveTransformer(n_state = 2, min_=0.0, max_=1.0).transform(im_array_all)

    
    data_corr = TwoPointCorrelation(
        periodic_boundary = False,
        cutoff = 160, # I can tune the cutoff so that it is just sufficient for the computed two-point cluster function to drop to 0 at large distances
        correlations=[(0, 0)]  
    ).transform(data)
    
    summed_array = da.sum(data_corr, axis = 0)
    
    np.save(os.path.join(imgchar_path, basename, basename + '_cluster.npy'), summed_array)


def read_correlation_r(basename, correlation_name) :
    path = imgchar_path + '/' + basename
    r = []
    with open(os.path.join(path, basename + '_rad' + correlation_name + '.txt')) as f :
        lines = f.readlines()
        r = eval(lines[0])

    p = np.load(os.path.join(path, basename + '_' + correlation_name + '_r.npy'))
    
    return r, p

# correlation_name should be either s2 or cluster    
def plot_correlation_r(correlation_name) :
    nb_descr = len(descriptors_name)
    
    images = os.listdir(extracted_path)
    if '.DS_Store' in images :
        images.remove('.DS_Store')
    
    
    # To adapt depending on the number of descriptors
    x_array = [[],[],[],[],[]]
    y_array = [[],[],[],[],[]]
    z_array = [[],[],[],[],[]]
    
    
    
    for image in images :
        basename = os.path.basename(image)
        full_name = os.path.join(images_path, basename + '.png')
        distr_name = os.path.join(imgchar_path, basename, basename + '_distributions.txt')
        distributions = read_distributions(distr_name)
        
        for k in range(nb_descr) :

            radii, probs = read_correlation_r(basename, correlation_name)
            
            
            for i in range(len(probs[0])) :
                x_array[k].append(radii[i])
                y_array[k].append(distributions[k][1][1])  # Let's plot S2 as a function of the average
                z_array[k].append(probs[0][i])
    
    for k in range(nb_descr) :
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = x_array[k]
        y = y_array[k]
        z = z_array[k]
        
        ax.scatter(x, y ,z, marker='.')
        ax.view_init(elev=20, azim=-20)
        
        ax.set_xlabel('r')
        ax.set_ylabel('Mean of ' + descriptors_name[k])
        ax.set_zlabel(correlation_name)
        
        plt.show()
        
# Plots S2 and C as a function of the other descriptors
def plot_correlation_all() :
    plot_correlation_r('s2')
    plot_correlation_r('cluster')
    
# Does not work    
def plot_correlation_img() :
    name = filedialog.askdirectory(title='Choose an image', initialdir=imgchar_path)
    basename = os.path.basename(name)
    
    try :
        auto_correlation = np.load(os.path.join(name, basename + '_s2.npy'))
        fig = plot_microstructures(auto_correlation[0,:,:,0], showticks=True, titles='S2', colorbar=True, cmap = 'viridis')
        fig.show()
    except FileNotFoundError :
        popup('S2 not computed for image : ' + basename)
        
    try :
        cluster = np.load(os.path.join(name, basename + '_cluster.npy'))
        fig = plot_microstructures(cluster[:,:,0], showticks=True, titles='Cluster', colorbar=True, cmap = 'viridis')
        fig.show()
    except FileNotFoundError :
        popup('Cluster not computed for image : ' + basename)
        
    try :
        lp = np.load(os.path.join(name, basename + '_lp.npy'))
        fig = plot_microstructures(lp, showticks=True, titles='Lineal Path', colorbar=True, cmap = 'viridis')
        fig.show()
    except FileNotFoundError :
        popup('Lineal path not computed for image : ' + basename)

    try :
        data_x_list = np.load(os.path.join(name, basename + '_cld_angular.npy'), allow_pickle=True)
        plot_cld_angular(data_x_list)
    except FileNotFoundError :
        popup('CLD angular not computed for image : ' + basename)
        
    try :
        data_x_list = np.load(os.path.join(name, basename + '_cld_dir.npy'), allow_pickle=True)
        plot_cld_dir(data_x_list)
    except FileNotFoundError :
        popup('CLD directional not computed for image : ' + basename)


def plot_stiffness() :
    tensors = os.listdir(consistent_path)
    if '.DS_Store' in tensors :
        tensors.remove('.DS_Store')
    
    nb_descriptors = len(descriptors_name)
    
    #[descriptor[moments]]
    # To adapt depending on the number of descriptors        
    descriptors = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
    stiffness = [[[],[],[]],[[],[],[]],[[],[],[]]]
    
    vf = []
      
    for tensor in tensors :
        
        name = tensor.split('_')[0]
        
        t = read_tensor(tensor)
        
        for i in range(3) :
            for j in range(3) :
                stiffness[i][j].append(float(t[i][j]))
        
        distr_name = imgchar_path + '/' + name + '/' + name + '_distributions.txt'
        distr = read_distributions(distr_name)

        vf.append(read_vf(imgchar_path + '/' + name + '/' + name + '_characteristics.txt'))
        
        for k in range(nb_descriptors) :
            descriptors[k][0].append(distr[k][1][1])
            descriptors[k][1].append(distr[k][1][2])
            descriptors[k][2].append(distr[k][1][3])
    
    for k in range(nb_descriptors) :
        for moment in range(3) :
            fig, axs = plt.subplots(3, 3, figsize=(14, 12))
            for m in range(3) :
                for n in range(3) :
                    x = descriptors[k][moment]
                    y = stiffness[m][n]
                    
                    
                    a, b = indexm_to_indext(m,n)
                    
                    axs[m,n].plot(x, y, 'r+')
                    axs[m,n].set_title('C' + str(a) + str(b))
            title = descriptors_name[k] + ' : Moment of order ' + str(moment+1)
            fig.suptitle(title)
            plt.tight_layout()
            plt.savefig(graphstiffness_path + '/' + title + '.png')
            plt.show()
    
    
    fig, axs = plt.subplots(3, 3, figsize=(14, 12))
    for m in range(3) :
        for n in range(3) :
            x = vf
            y = stiffness[m][n]
            
            
            a, b = indexm_to_indext(m,n)
            
            axs[m,n].plot(x, y, 'r+')
            axs[m,n].set_title('C' + str(a) + str(b))
    title = 'Volume fraction'
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(graphstiffness_path + '/' + title + '.png')
    plt.show()
           


# Plots the stiffness a function of the norms of S2
def plot_stiffness_correlation(mode) :
    
    images_char = os.listdir(imgchar_path)
    if '.DS_Store' in images_char :
            images_char.remove('.DS_Store')
            
    s2_computed = []
    
    for i in images_char :
        files = os.listdir(imgchar_path + '/' + i)
        if '.DS_Store' in files :
            files.remove('.DS_Store')
        for f in files :
            if '_s2.npy' in f :
                s2_computed.append(i)
                break
    
    norms_to_compute = ['fro', 1, 2]
    
    stiffness = [[[],[],[]],[[],[],[]],[[],[],[]]]
    
    tensors = os.listdir(consistent_path)
    if '.DS_Store' in tensors :
        tensors.remove('.DS_Store')
    
        
    if len(s2_computed) != len(tensors) :
        popup('Warning ! Nnumber of S2 computed : ' + str(len(s2_computed)) + '\tNnumber of tensors computed : ' + str(len(tensors)))
    
    to_plot = []
    
    # Let's assume for now that s2_computed = cluster_computed = lp_computed
    for s in s2_computed :
        for t in tensors :
            if s in t :
                to_plot.append(s)
                continue
    
    for t in to_plot :
        tensor = t + '_effective_stiffness.txt'
        tensor = read_tensor(tensor)
        
        for i in range(3) :
            for j in range(3) :
                
                stiffness[i][j].append(float(tensor[i][j]))
    
    
    if mode == 'norm' :
    
        for k in range(len(norms_to_compute)) :
            
            # See if something more efficient could be made
            norms_s2 = []
            norms_c = []
            norms_lp = []
            norms_cld_angular = []
            
            for img in to_plot :
                auto_correlation = np.load(os.path.join(imgchar_path, img, img + '_s2.npy'))
                norm_s2 = np.linalg.norm(auto_correlation[0,:,:,0], ord=norms_to_compute[k])
                norms_s2.append(norm_s2)
                
                cluster = np.load(os.path.join(imgchar_path, img, img + '_cluster.npy'))
                norm_c = np.linalg.norm(cluster[:,:,0], ord=norms_to_compute[k])
                norms_c.append(norm_c)
                
                lp = np.load(os.path.join(imgchar_path, img, img + '_lp.npy'))
                norm_lp = np.linalg.norm(lp, ord=norms_to_compute[k])
                norms_lp.append(norm_lp)
            
                cld_angular = np.load(os.path.join(imgchar_path, img, img + '_cld_angular.npy'), allow_pickle=True)
                pdfs_cld_angular = [d.pdf for d in cld_angular]
                norm_cld_angular = np.linalg.norm(pdfs_cld_angular, ord=norms_to_compute[k])
                norms_cld_angular.append(norm_cld_angular)
            
            # Number of correlation descriptors    
            for p in range(len(correlation_descriptors)) :
                
                if p == 0 :
                    x = norms_s2
                    
                elif p == 1 :
                    x = norms_c
                
                elif p == 2 :
                    x = norms_lp
                
                elif p == 3 :
                    x = norms_cld_angular
                    
                name = correlation_descriptors[p]
                
                fig, axs = plt.subplots(3, 3, figsize=(14, 12))
                for m in range(3) :
                    for n in range(3) :
                        y = stiffness[m][n]
                        
                        a, b = indexm_to_indext(m,n)
                        
                        axs[m,n].plot(x, y, 'r+')
                        axs[m,n].set_title('C' + str(a) + str(b))
                title = name + ' norm : ' + str(norms_to_compute[k])
                fig.suptitle(title)
                plt.tight_layout()
                #plt.savefig()
                plt.show() 
    
    
    elif mode == 'dir' :
        
        # Direction, moment
        x_s2 = [[[],[],[]],[[],[],[]]]
        x_c = [[],[],[]],[[],[],[]]
        x_lp = [[],[],[]],[[],[],[]]
        
        for img in to_plot :
            
                auto_correlation = np.load(os.path.join(imgchar_path, img, img + '_s2.npy'))
                cluster = np.load(os.path.join(imgchar_path, img, img + '_cluster.npy'))
                lp = np.load(os.path.join(imgchar_path, img, img + '_lp.npy'))
                
                s2_max = descriptors_max['s2']
                cluster_max = descriptors_max['cluster']
                lp_max = descriptors_max['lp']
                
                s2_distr_1 = distribution_descriptor(auto_correlation[0,:,cutoff_input,0], s2_max)
                cluster_distr_1 = distribution_descriptor(cluster[:,cutoff_input,0], cluster_max)
                lp_distr_1 = distribution_descriptor(lp[:,lp.shape[1]//2], lp_max)
                
                s2_moments_1 = compute_moments(s2_distr_1,'s2')
                cluster_moments_1 = compute_moments(cluster_distr_1, 'cluster')
                lp_moments_1 = compute_moments(lp_distr_1, 'lp')
                
                s2_distr_2 = distribution_descriptor(auto_correlation[0,cutoff_input,:,0], s2_max)
                cluster_distr_2 = distribution_descriptor(cluster[cutoff_input,:,0], cluster_max)
                lp_distr_2 = distribution_descriptor(lp[lp.shape[0]//2,:], lp_max)
                
                s2_moments_2 = compute_moments(s2_distr_2,'s2')
                cluster_moments_2 = compute_moments(cluster_distr_2, 'cluster')
                lp_moments_2 = compute_moments(lp_distr_2, 'lp')
                
                for moment in range(3) :
                    x_s2[0][moment].append(s2_moments_1[moment+1])
                    x_s2[1][moment].append(s2_moments_2[moment+1])
                
                    x_c[0][moment].append(cluster_moments_1[moment+1])
                    x_c[1][moment].append(cluster_moments_2[moment+1])
                    
                    x_lp[0][moment].append(lp_moments_1[moment+1])
                    x_lp[1][moment].append(lp_moments_2[moment+1])
                    
        
        # No plot for CLD dir at the moment       
        for p in range(len(correlation_descriptors)-1) :
            for direction in range(2) :        
                for moment in range(3) : 
                    if p == 0 :
                        x = x_s2[direction][moment]
                        
                    elif p == 1 :
                        x = x_c[direction][moment]
                    
                    elif p == 2 :
                        x = x_lp[direction][moment]
                        
                    name = correlation_descriptors[p]
                    
                    fig, axs = plt.subplots(3, 3, figsize=(14, 12))
                    for m in range(3) :
                        for n in range(3) :
                            y = stiffness[m][n]
                            
                            a, b = indexm_to_indext(m,n)
                            
                            axs[m,n].plot(x, y, 'r+')
                            axs[m,n].set_title('C' + str(a) + str(b))
                    title = name + ' direction : ' + str(direction+1) + ' moment : ' + str(moment+1)
                    fig.suptitle(title)
                    plt.tight_layout()
                    #plt.savefig()
                    plt.show()        
        
                 
        
# Reads the consistent stiffness tensor from the .txt file
# Please note that this is a string
def read_tensor(t) :
    with open(consistent_path + '/' + t) as f :
        lines = f.readlines()
        for i in range(len(lines)) :
            lines[i] = lines[i].split('\t')
            lines[i].remove('\n')
    return lines[0:3]




# Returns the distances between the p-th order moments and the assiocated difference between stiffnesses
# The distances variable is in the form [[descriptor1_0th, descriptor2_1th, ...], [descriptor2_0th, ...], ...]
def compute_error(t1_name, t2_name, m, n) :
    name1 = t1_name.split('_')[0]
    name2 = t2_name.split('_')[0]

    name1 = imgchar_path + '/' + name1 + '/' + name1 + '_distributions.txt'
    name2 = imgchar_path + '/' + name2 + '/' + name2 + '_distributions.txt'
    
    distr1 = read_distributions(name1)
    distr2 = read_distributions(name2)
    
    nb_descriptors = len(descriptors_name)
    
    # Defining it as nb_descriptors*[4*[0]] does not work
    # To adapt depending on the number of descriptors
    distances = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
   
    
    # For each descriptor
    for i in range(nb_descriptors) :
        #For each moment
        for k in range(4) :
            if k == 0 :
                distances[i][k] = wasserstein_distance(distr1[i][1][k], distr2[i][1][k])
                #distances[i][k] = abs(distr1[i][1][k][0] - distr2[i][1][k][0])
            else :
                distances[i][k] = abs(distr1[i][1][k] - distr2[i][1][k])
        
    tensor1 = read_tensor(t1_name)
    tensor2 = read_tensor(t2_name)
    
    delta = abs(float(tensor1[m][n]) - float(tensor2[m][n]))
    
    
    return [distances,delta]


def compute_error_vf(t1_name, t2_name, m, n) :
    name1 = t1_name.split('_')[0]
    name2 = t2_name.split('_')[0]

    name1 = imgchar_path + '/' + name1 + '/' + name1 + '_characteristics.txt'
    name2 = imgchar_path + '/' + name2 + '/' + name2 + '_characteristics.txt'
    
    v1 = read_vf(name1)
    v2 = read_vf(name2)
    
    delta_v = abs(v2-v1)
    
    tensor1 = read_tensor(t1_name)
    tensor2 = read_tensor(t2_name)
    
    delta_t = abs(float(tensor1[m][n]) - float(tensor2[m][n]))
    
    return [delta_v, delta_t]


def compute_error_correlation(t1_name, t2_name, m, n) :
    name1 = t1_name.split('_')[0]
    name2 = t2_name.split('_')[0]
    
    nb_correlation_descriptors = len(correlation_descriptors)
    
    auto_correlation1 = np.load(os.path.join(imgchar_path, name1, name1 + '_s2.npy'))
    auto_correlation2 = np.load(os.path.join(imgchar_path, name2, name2 + '_s2.npy'))
    cluster1 = np.load(os.path.join(imgchar_path, name1, name1 + '_cluster.npy'))
    cluster2 = np.load(os.path.join(imgchar_path, name2, name2 + '_cluster.npy'))
    lp_1 = np.load(os.path.join(imgchar_path, name1, name1 + '_lp.npy'))
    lp_2 = np.load(os.path.join(imgchar_path, name2, name2 + '_lp.npy'))
    cld_angular_1 = np.load(os.path.join(imgchar_path, name1, name1 + '_cld_angular.npy'), allow_pickle=True)
    cld_angular_2 = np.load(os.path.join(imgchar_path, name2, name2+ '_cld_angular.npy'), allow_pickle=True)
    
    pdfs_cld_1 = [d.pdf for d in cld_angular_1]
    pdfs_cld_2 = [d.pdf for d in cld_angular_2]
    
    
    emd = nb_correlation_descriptors * [0]
    emd[0] = wasserstein_distance_nd(auto_correlation1[0,:,:,0], auto_correlation2[0,:,:,0])
    emd[1] = wasserstein_distance_nd(cluster1[:,:,0], cluster2[:,:,0])
    emd[2] = wasserstein_distance_nd(lp_1, lp_2)
    emd[3] = wasserstein_distance_nd(pdfs_cld_1, pdfs_cld_2)
    
    tensor1 = read_tensor(t1_name)
    tensor2 = read_tensor(t2_name)
    
    delta_t = abs(float(tensor1[m][n]) - float(tensor2[m][n]))
    
    return [emd, delta_t]
    
# Convert the matrix index to tensor index in the 2D case
def indexm_to_indext(line, column) :
    a = 0
    b = 0
    if line == 0 :
        a = 11
    elif line == 1 :
        a = 22
    elif line == 2 :
        a = 21
    if column == 0 :
        b = 11
    elif column == 1 :
        b = 22
    elif column == 2 :
        b = 21
        
    return (a,b)


# Takes a string iklj and returns (line, column)
def indext_to_indexm(index) :
    a = index[0:2]
    b = index[2:4]
    line = 0
    column = 0
    
    if a == '11' :
        line = 0
    elif a == '22' :
        line = 1
    elif a == '21' :
        line = 2
        
    if b == '11' :
        column = 0
    elif b == '22' :
        column = 1
    elif b == '21' :
        column = 2  
        
    return (line, column)      
        
        

def compute_errorpropag_ref(fen) :
    
    consistent_tensors = os.listdir(consistent_path)
    if '.DS_Store' in consistent_tensors :    
        consistent_tensors.remove('.DS_Store')
    
    name = errorpropag_path + '/groundtruth'
    
    done = os.listdir(name)
    if '.DS_Store' in done :
        done.remove('.DS_Store')
    
    
    
    if len(done) > 0 :
        overwrite_bool = overwrite(fen)
        if not overwrite_bool :
            popup('Nothing done !')
            return

    
    name_ref = filedialog.askopenfilename(title='Choose a ground truth', initialdir=consistent_path)
    if not ('effective_stiffness' in name_ref and 'consistent_tensors' in name_ref) :
        popup('Invalid input')
        return
    
    
    basename = os.path.basename(name_ref)
    consistent_tensors.remove(basename)
    
    name_dir = basename.split('_')[0]
    
    nb_descriptors = len(descriptors_name)
    nb_tensors = len(consistent_tensors)
    
    create_dir(name + '/' + name_dir)
    
    time_start = time.time()
    
    for m in range(3) :
        for n in range(3) :
            
            a, b = indexm_to_indext(m,n)
            coefficient = 'C' + str(a) + str(b)
            name_coeff = name + '/' + name_dir + '/' + coefficient
            create_dir(name_coeff)
            
            for k in range(nb_descriptors) :
                # I don't know why but x = 4*[[]] does not work (the elements are appened to every list)
                x = [[], [], [], []]
                y = []
                for i in range(nb_tensors) :
                    
                    point = compute_error(consistent_tensors[i], basename, m, n)
                    # Up to third moment
                    for p in range(4) :
                        x[p].append(point[0][k][p])
                    y.append(point[-1])

                name_coeff_descr = name_coeff + '/' + descriptors_name[k] + '.txt'
                create_file(name_coeff_descr)
                
                with open(name_coeff_descr, 'w') as f :
                    f.write(str(x) + '\n')
                    f.write(str(y) + '\n')
                    
            x = []
            y = []        
            for i in range(nb_tensors) :
                point = compute_error_vf(consistent_tensors[i], basename, m, n)
                x.append(point[0])
                y.append(point[1])
            name_coeff_descr = name_coeff + '/Volume fraction.txt'
            create_file(name_coeff_descr)
            
            with open(name_coeff_descr, 'w') as f :
                f.write(str(x) + '\n')
                f.write(str(y) + '\n')    
                
            # To adapt depending on the number of correlation descriptors
            x = [[],[],[],[]]
            y = []
            for i in range(nb_tensors) :
                
                point = compute_error_correlation(consistent_tensors[i], basename, m, n)
                x[0].append(point[0][0])
                x[1].append(point[0][1])
                x[2].append(point[0][2])
                x[3].append(point[0][3])
                y.append(point[1])    
                
                print( (m, n, i) )
                
            for k in range(len(correlation_descriptors)) :
                
                name_coeff_descr = name_coeff + '/' + correlation_descriptors[k] + '.txt'
                create_file(name_coeff_descr)
                 
                with open(name_coeff_descr, 'w') as f :
                    f.write(str(x[k]) + '\n')
                    f.write(str(y) + '\n')    
                    
    time_end = time.time()
    
    
    popup('Computing distances between moments done.', time_end - time_start)



# Computes the error propagation and stores it in a file
# Creates one folder per coefficient. Each folder contains one .txt file for each descriptor. 
# Each text file contains the differences between the p-th order moment (as a list of lists) and the difference of stiffnesses associated
def compute_errorpropag_pairwise() :
    
    name = errorpropag_path + '/pairwise'
    
    consistent_tensors = os.listdir(consistent_path)
    if '.DS_Store' in consistent_tensors :
        consistent_tensors.remove('.DS_Store')
    
    nb_descriptors = len(descriptors_name)
    nb_tensors = len(consistent_tensors)
    
    time_start = time.time()
    
    for m in range(3) :
        for n in range(3) :
            
            a, b = indexm_to_indext(m,n)
            coefficient = 'C' + str(a) + str(b)
            name_coeff = name + '/' + coefficient
            create_dir(name_coeff)
            
            for k in range(nb_descriptors) :
                # I don't know why but x = 4*[[]] does not work (the elements are appened to every list)
                x = [[], [], [], []]
                y = []
                for i in range(nb_tensors) :
                    for j in range(nb_tensors) :
                        if j > i :
                            point = compute_error(consistent_tensors[i], consistent_tensors[j], m, n)
                            # Up to third moment
                            for p in range(4) :
                                x[p].append(point[0][k][p])
                            y.append(point[-1])

                name_coeff_descr = name_coeff + '/' + descriptors_name[k] + '.txt'
                create_file(name_coeff_descr)
                
                with open(name_coeff_descr, 'w') as f :
                    f.write(str(x) + '\n')
                    f.write(str(y) + '\n')
            
            
            x = []
            y = []
            for i in range(nb_tensors) :
                    for j in range(nb_tensors) :
                        if j > i :
                            point = compute_error_vf(consistent_tensors[i], consistent_tensors[j], m, n)
                            x.append(point[0])
                            y.append(point[1])
                            print((m,n,i,j))
            
            name_coeff_descr = name_coeff + '/Volume fraction.txt'
            create_file(name_coeff_descr)
            
            with open(name_coeff_descr, 'w') as f :
                f.write(str(x) + '\n')
                f.write(str(y) + '\n')                
                            
                            
            # To adapt depending on the number of correlation descriptors
            x = [[], [], []]
            y = []
            for i in range(nb_tensors) :
                for j in range(nb_tensors) :
                    if j > i :
                        point = compute_error_correlation(consistent_tensors[i], consistent_tensors[j], m, n)
                        x[0].append(point[0][0])
                        x[1].append(point[0][1])
                        x[2].append(point[0][2])
                        y.append(point[1])    
                        
            for k in range(len(correlation_descriptors)) :
                name_coeff_descr = name_coeff + '/' + correlation_descriptors[k] + '.txt'
                create_file(name_coeff_descr)
                 
                with open(name_coeff_descr, 'w') as f :
                    f.write(str(x[k]) + '\n')
                    f.write(str(y) + '\n')              
                                 
                    
    time_end = time.time()
    popup('Computing distances between moments done.', time_end - time_start)
                    

                 
def compute_pairwise_init(fen) :
    
    name = errorpropag_path + '/pairwise'
    
    done = os.listdir(name)
    if '.DS_Store' in done :
        done.remove('.DS_Store')
    
    if len(done) > 0 :
        overwrite_bool = overwrite(fen)
        if not overwrite_bool :
            popup('Nothing done !')
            return        
    
    p = Thread(target=compute_errorpropag_pairwise)
    p.start()
    
# Plots the difference of stiffness as a function of the distance between pdfs
def plot_errorpropag(fen, mode) :
    consistent_tensors = os.listdir(consistent_path)
    if '.DS_Store' in consistent_tensors :
        consistent_tensors.remove('.DS_Store')
    
    name = errorpropag_path + '/' + mode
    
    percentage = False
    
    if mode == 'groundtruth' :
        gt = filedialog.askdirectory(title='Choose a ground truth', initialdir=name)
        if not ('groundtruth' in name) :
            popup('Invalid input')
            return
        
        gt = os.path.basename(gt)
        
        name = name + '/' + gt
    
    
        percentage = askPercentage(fen)
    
    coefficients = os.listdir(name)
    if '.DS_Store' in coefficients :
        coefficients.remove('.DS_Store')
    
    for c in coefficients :
        name_c = name + '/' + c
        descriptors = os.listdir(name_c)
        if '.DS_Store' in descriptors :
            descriptors.remove('.DS_Store')
        
        for d in descriptors :
            
            name_d = name_c + '/' + d
            
            with open(name_d, 'r') as f :
                
                read_lines = f.readlines()
                lines = []
                for l in read_lines :
                    lines.append(eval(l))
                
                
                if 'Volume fraction' in d or 'Cluster' in d or 'S2' in d or 'Lineal Path' in d or 'Chord Length' in d :
                    x = lines[0]
                    y = lines[1]
                    
                    if percentage : 
                        t = read_tensor(gt + '_effective_stiffness.txt')
                        code = c[1:5]
                        coord = indext_to_indexm(code)
                        coeff = float(t[coord[0]][coord[1]])
                        
                        y = [e/coeff * 100 for e in y]
                    
                    if percentage :
                        plt.xlabel('Percentage of error')
                        title = title = c + ' : ' + rm_ext(d) + ' percentage'
                    else :
                        plt.xlabel('Difference of error')
                        title = title = c + ' : ' + rm_ext(d) + ' difference'
                    
                    plt.title(title)
                    plt.plot(x, y, 'r+')
                    
                    plt.savefig(graph_path + '/' + title + ' ' + mode + '.jpg')
                    plt.show()
                    
                else :    
                    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                    for p in range(4) :
                        x = lines[0][p]
                        y = lines[-1]
                        
                        if percentage : 
                            t = read_tensor(gt + '_effective_stiffness.txt')
                            code = c[1:5]
                            coord = indext_to_indexm(code)
                            coeff = float(t[coord[0]][coord[1]])
                            
                            y = [e/coeff * 100 for e in y]
                        
                        h = p // 2
                        if h == 0 :
                            g = p-h
                        elif h == 1 :
                            g = p-h-1
                        
                        axs[h, g].plot(x, y, 'r+')
                        axs[h, g].set_title('Moment of order ' + str(p))
                
                    plt.tight_layout()
                    
                    
                    if percentage :
                        plt.xlabel('Percentage of error')
                        title = title = c + ' : ' + rm_ext(d) + ' percentage'
                    else :
                        plt.xlabel('Difference of error')
                        title = title = c + ' : ' + rm_ext(d) + ' difference'
                    
                    title = c + ' : ' + rm_ext(d)
                    fig.suptitle(title)
                    
                    plt.savefig(graph_path + '/' + title + ' ' + mode + '.jpg')
                    plt.show()  
    


def eliminate_oultiers(x_input, y_input) :
    
    threshold = 95
    
    Q_x = np.percentile(x_input, threshold)
    Q_y = np.percentile(y_input, threshold)
    
    inliers_x = []
    inliers_y = []
    
    if len(x_input) !=  len(y_input) :
        popup('Different lengths when filtering outliers.')
        return

    for i in range(len(x_input)) :
        x = x_input[i]
        y = y_input[i]
        if x <= Q_x and y <= Q_y :
            inliers_x.append(x)
            inliers_y.append(y)

    

    return inliers_x, inliers_y


def correlation(x, y) :
    
    spearman_threshold = 0.85
    
    data = {
        'x' : x,
        'y' : y
    }
    df = pd.DataFrame(data)
    
    spearman = df['y'].corr(df['x'], method='spearman')
    
    if spearman_threshold >= 0.85 :
        return True
    else :
        return False


# Works for groundtruth error propagation only
# Parameters : Ground truth, tensor coefficients, descriptor number, moment
def read_error(gt, m, n, k, p) :
    
    a, b = indexm_to_indext(m,n)
    coefficient = 'C' + str(a) + str(b)
    
    descr_name = descriptors_name[k]
    
    file_name = os.path.join(errorpropag_path, 'groundtruth', gt, coefficient, descr_name + '.txt')
    with open(file_name, 'r') as f :
        lines = f.readlines()
        
        lines0_eval = eval(lines[0])
        
        x = lines0_eval[p]
        y = eval(lines[1])
    
    x_filtered, y_filtered = eliminate_oultiers(x, y)
    
    return x_filtered, y_filtered



# Rank descriptors based on their impact on the stiffness, coefficient by coefficient
# Only works for basic descriptors
def rank_descriptors_v1() :
    
    nb_descriptors = len(descriptors_name)
    
    gt_fullname = filedialog.askopenfilename(title='Select a groundtruth to asses the impact of the descriptors', initialdir=consistent_path)
    if not 'effective_stiffness' in gt_fullname :
        popup('Invalid input !')
        return
    
    gt_basename = os.path.basename(gt_fullname)
    gt = gt_basename.split('_')[0]
    
    # We assume that everyhting has been computed
    tensors = os.listdir(consistent_path)
    if '.DS_Store' in tensors :
        tensors.remove('.DS_Store')
    
    #[row][column][descriptor][moment]
    # To adapt depenging on the number of descriptors
    scores = [
        [[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]],
        [[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]],
        [[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]]
    ]
    
    # For each coefficient, for each descriptor, tells which moment has the most impact
    scores_moments = [
        [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
        [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
        [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    ]
    
    tensors = os.listdir(consistent_path)
    if '.DS_Store' in tensors :
        tensors.remove('.DS_Store')
    
    nb_descriptors = len(descriptors_name)
    
    #[descriptor[moments]]
    # To adapt depending on the number of descriptors        
    descriptors = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
    stiffness = [[[],[],[]],[[],[],[]],[[],[],[]]]
    
    vf = []
      
    for tensor in tensors :
        
        name = tensor.split('_')[0]
        
        t = read_tensor(tensor)
        
        for i in range(3) :
            for j in range(3) :
                stiffness[i][j].append(float(t[i][j]))
        
        distr_name = imgchar_path + '/' + name + '/' + name + '_distributions.txt'
        distr = read_distributions(distr_name)

        vf.append(read_vf(imgchar_path + '/' + name + '/' + name + '_characteristics.txt'))
        
        for k in range(nb_descriptors) :
            descriptors[k][0].append(distr[k][1][1])
            descriptors[k][1].append(distr[k][1][2])
            descriptors[k][2].append(distr[k][1][3])
        
        vf_name = imgchar_path + '/' + name + '/' + name + '_characteristics.txt'
        vf.append(read_vf(vf_name))
    
    
    for k in range(nb_descriptors) :
        
        # To adapt depending on the number of moments calculated
        for p in range(3) :
            x_input = descriptors[k][p]
            
            for m in range(3) :
                for n in range(3) :
                    y_input = stiffness[m][n]
                    
                    x_filtered, y_filtered = eliminate_oultiers(x_input, y_input)
                    
                    if correlation(x_filtered, y_filtered) :
                        x, y = read_error(gt, m, n, k, p)
                        
                        y_impact = (max(y) - min(y))/max(y)
                        x_impact = (max(x) - min(x))

                        ratio = y_impact / x_impact
                        
                        scores[m][n][k][p] = ratio
                        
                        
    for k in range(nb_descriptors) :
        for m in range(3) :
            for n in range(3) :
                max_index = scores[m][n][k].index(max(scores[m][n][k]))
    
    
               

    for m in range(3) :
        for n in range(3) :
            for k in range(nb_descriptors) :
                a, b = indexm_to_indext(m,n)
                coefficient = 'C' + str(a) + str (b)
                max_moment = scores_moments[m][n][k]
                print(coefficient + ' : ' + descriptors_name[k] + ' impact : ' + str(scores[m][n][k][max_moment]) + ' (maximum for moment : ' + str(max_moment + 1) + ')')                
                     

# Ranks descriptors based on the correlation between the difference of stiffness and the difference of pdf
# Only works for basic descriptors
def rank_descriptors_v2() :
    
    consistent_tensors = os.listdir(consistent_path)
    if '.DS_Store' in consistent_tensors :
        consistent_tensors.remove('.DS_Store')
    
    name = errorpropag_path + '/groundtruth'
    
    percentage = False
    

    gt = filedialog.askdirectory(title='Choose a ground truth', initialdir=name)
    if not ('groundtruth' in name) :
        popup('Invalid input')
        return
    
    gt = os.path.basename(gt)
    
    name = name + '/' + gt
    
    coefficients = os.listdir(name)
    if '.DS_Store' in coefficients :
        coefficients.remove('.DS_Store')
    
    for c in coefficients :
        name_c = name + '/' + c
        
        descriptors = descriptors_name
        
        for d in descriptors :
            name_d = name_c + '/' + d + '.txt'
            
            with open(name_d, 'r') as f :
                
                read_lines = f.readlines()
                lines = []
                for l in read_lines :
                    lines.append(eval(l))
    

                for p in range(4) :
                    x = lines[0][p]
                    y = lines[-1]
                    
                    data = {
                        'x' : x,
                        'y' : y
                    }
                    df = pd.DataFrame(data)
                    
                    spearman = df['y'].corr(df['x'], method='spearman')
                    pearson = df['y'].corr(df['x'], method='pearson')
                    
                    print(str(c) + ' : ' + str(d) + ' moment of order : ' + str(p) + ' Pearson correlation coefficient : ' + str(pearson))
                    print(str(c) + ' : ' + str(d) + ' moment of order : ' + str(p)+ ' Spearman correlation coefficient : ' + str(spearman))
                 
    

    
def quit_fen(fen) :
    plt.close('all')
    fen.quit()
    fen.destroy()
    sys.exit()


def overwrite(fen) :
    p = tk.Toplevel()
    
    
    p.wm_title('Overwrite')
    var = tk.BooleanVar(p)
    var.set(False)
    
    def btn(boolean) :
        var.set(boolean)
        label2.config(text='Current value : {}'.format(var.get()))
    
    label1 = tk.Label(p, text='Would you want to overwrite existing results ?')
    label1.grid(row=0, column=0, pady=2, columnspan=2)
    
    label2 = tk.Label(p, text='Current value : {}'.format(var.get()))
    label2.grid(row=1, column=0, pady=2, columnspan=2)
    
    yes_btn = tk.Button(p, text='Yes', command = lambda : btn(True))
    yes_btn.grid(row=2, column=0, pady=2, sticky = 'ew')
    
    no_btn = tk.Button(p, text='No', command = lambda : btn(False))
    no_btn.grid(row=2, column=1, pady=2, sticky = 'ew')
    
    ok_btn = tk.Button(p, text='Ok', command =p.destroy)
    ok_btn.grid(row=3, column=0, sticky='ew', columnspan=2)
    
    
    # The centering does not work
    screen_width = p.winfo_screenwidth()
    screen_height = p.winfo_screenheight()
    
    width = p.winfo_width()
    height = p.winfo_height()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    #p.geometry(f'{width}x{height}+{x}+{y}')
    
    # Does not work
    fen.wait_window(p)
    
    return var.get()


def askPercentage(fen) :
    
    p = tk.Toplevel()
    
    p.wm_title('Percentage')
    var = tk.BooleanVar(p)
    var.set(False)
    
    def btn(boolean) :
        var.set(boolean)
        label2.config(text='Current value : {}'.format(var.get()))
    
    label1 = tk.Label(p, text='Would you want to see the error propagation as a percentage ?')
    label1.grid(row=0, column=0, pady=2, columnspan=2)
    
    label2 = tk.Label(p, text='Current value : {}'.format(var.get()))
    label2.grid(row=1, column=0, pady=2, columnspan=2)
    
    yes_btn = tk.Button(p, text='Yes', command = lambda : btn(True))
    yes_btn.grid(row=2, column=0, pady=2, sticky = 'ew')
    
    no_btn = tk.Button(p, text='No', command = lambda : btn(False))
    no_btn.grid(row=2, column=1, pady=2, sticky = 'ew')
    
    ok_btn = tk.Button(p, text='Ok', command =p.destroy)
    ok_btn.grid(row=3, column=0, sticky='ew', columnspan=2)
    
    fen.wait_window(p)
    
    return var.get()

def popup(text_input, time=0) :
    p = tk.Toplevel()
    
    p.wm_title('Attention')
    label1 = tk.Label(p, text=text_input)
    label1.pack()
    if time != 0 :
        time = format(time, '.2f')
        label2 = tk.Label(p, text='Time elapsed : ' + str(time))
        label2.pack()
    close_btn = tk.Button(p, text='OK', command=p.destroy)
    close_btn.pack()
    
    screen_width = p.winfo_screenwidth()
    screen_height = p.winfo_screenheight()
    
    width = p.winfo_width()
    height = p.winfo_height()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    #p.geometry(f'{width}x{height}+{x}+{y}')
    


# To choose working directory
def choose_wd() :
    global working_dir
    working_dir = filedialog.askdirectory(initialdir=working_dir)
    

def main () :
    fen = tk.Tk()
    
    width = 800
    height = 730
    
    screen_width = fen.winfo_screenwidth()
    screen_height = fen.winfo_screenheight()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    fen.geometry(f'{width}x{height}+{x}+{y}')
    
    initdir_btn = tk.Button(fen, text='Create all the necessary directories', command=init_dirs)
    initdir_btn.pack()
    
    dispdir_btn = tk.Button(fen, text='Display the working directories', command=display_working_dirs)
    dispdir_btn.pack()
    
    choosedir_btn = tk.Button(fen, text='Choose working directory', command=choose_wd)
    choosedir_btn.pack()
    
    imgchar_btn = tk.Button(fen, text='Compute images characteristics', command= lambda : compute_img_char(fen))
    imgchar_btn.pack()
    
    plotdistr_btn = tk.Button(fen, text='Plot the descriptors of a selected image as points', command=plot_distribution)
    plotdistr_btn.pack()
    
    hist_btn = tk.Button(fen, text='Plot the descriptors of a selected image as histograms', command=plot_hist)
    hist_btn.pack()
    
    angle_btn = tk.Button(fen, text='Plot the orientation of a selected image as polar histograms', command=plot_polar)
    angle_btn.pack()
    
    extraction_btn = tk.Button(fen, text='Extract inclusions', command= lambda : extract_inclusion_all(fen))
    extraction_btn.pack()
    
    compute_correlation_btn = tk.Button(fen, text='Compute S2, C, L and CLD', command= lambda : compute_correlation(fen))
    compute_correlation_btn.pack()
    
    plot_correlation_img_btn = tk.Button(fen, text='Plot radially averaged S2 and C as a function of the other descriptors', command=plot_correlation_all)
    plot_correlation_img_btn.pack()
    
    plot_correlation_btn = tk.Button(fen, text='Plot S2, C, L and CLD for a given image', command=plot_correlation_img)
    plot_correlation_btn.pack()
    
    convert_btn = tk.Button(fen, text='Convert images to meshes with Moose', command= lambda : mesh_images_init(fen))
    convert_btn.pack()
    
    run_btn = tk.Button(fen, text='Run simulations on the meshes', command= lambda : run_simulations_init(fen))
    run_btn.pack()
    
    read_btn = tk.Button(fen, text='Compute consistent tensors', command= lambda : compute_results(fen))
    read_btn.pack()
    
    hsw_btn = tk.Button(fen, text='Check if a tensor lies within Hashin-Shtrikman\'s bounds', command=hsw)
    hsw_btn.pack()
    
    plotstiff = tk.Button(fen, text='Plot the stiffness as a function of the moments of the descriptors', command=plot_stiffness)
    plotstiff.pack()
    
    plot_stiffness_s2_norm_btn = tk.Button(fen, text='Plot the stiffness as a function of the norms of S2, C, L and CLD angular', command= lambda : plot_stiffness_correlation('norm'))
    plot_stiffness_s2_norm_btn.pack()
    
    plot_stiffness_s2_dir_btn = tk.Button(fen, text='Plot the stiffness as a function of the moments of S2, C and L in directions 1 and 2', command= lambda : plot_stiffness_correlation('dir'))
    plot_stiffness_s2_dir_btn.pack()
    
    computeground_btn = tk.Button(fen, text='Compute distances between moments of descriptors and associated difference of stiffnesses relative to a ground truth', command= lambda : compute_errorpropag_ref(fen))
    computeground_btn.pack()
    
    graphground_btn = tk.Button(fen, text='Plot ground truth error propagation', command= lambda : plot_errorpropag(fen, 'groundtruth'))
    graphground_btn.pack()
    
    computepairwise_btn = tk.Button(fen, text='Compute distances between moments of descriptors and associated difference of stiffnesses pairwise', command= lambda : compute_pairwise_init(fen))
    computepairwise_btn.pack()
    
    graphpairwise_btn = tk.Button(fen, text='Plot pairwise error propagation', command= lambda : plot_errorpropag(fen, 'pairwise'))
    graphpairwise_btn.pack()
    
    rank1_btn = tk.Button(fen, text='Print the descriptors influence V1', command=rank_descriptors_v1)
    rank1_btn.pack()
    
    rank2_btn = tk.Button(fen, text='Print the descriptors influence V2', command=rank_descriptors_v2)
    rank2_btn.pack()
    
    help_btn = tk.Button(fen, text='Help', command=help)
    help_btn.pack()
    
    quit_btn = tk.Button(fen, text='Quit', command= lambda : quit_fen(fen))
    quit_btn.pack()
    
    fen.mainloop()
  

main()
