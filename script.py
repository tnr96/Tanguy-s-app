import os
import subprocess
import numpy as np
import time
import glob
import csv
import tkinter as tk
from tkinter import filedialog
import cv2
from math import sqrt, radians
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from threading import Thread


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

# Solidity is removed for the moment
descriptors_name = ['Aspect ratio', 'Extent', 'Size', 'Orientation'] # List of descriptors used. Be careful with the order
descriptors_max = {'aspect_ratio' : 1, 'extent' : 1, 'size': 50, 'orientation' : 180, 'solidity' : 1}  # The maximum value of a descriptor. Useful for plotting. Maybe replace it with a case by case max search. 
# Max size to be discussed
descriptor_sampling = 20 # Number of classes in which to divide each descriptor

def display_working_dirs() :
    text = 'Working directory : ' + working_dir + '\n\nImages folder : ' + images_path + '\n\nMOOSE application : ' + moose_dir + '/' + moose_app + '\n\n' \
        'The following files are supposed to be in the Working directory : convert_moose.sh, run_moose.sh.\n\n'\
            'The following directories are used in the Working directory : moose_output, coreform_output, journals, to_run, simulation_results, consistent_tensors, images_characteristics, error_propag.\n\n' \
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
    
    for x, element in enumerate(list1) :
        for k in list2 :
            if rm_ext(element) == rm_ext(k) :
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

prepapre_to_copy()
       
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
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
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
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
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
    print(bins_middle)
    print(counts)
    bins_middle = [radians(e) for e in bins_middle]
    ax.bar(bins_middle, counts, width = 0.1, color='blue')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    title = os.path.basename(name).split('_')[0] + ' : Orientation'
    plt.title(title)
    plt.show()

# Computes the characteristics of one image
def img_char(image_arg) :
    image = cv2.imread(images_path + '/' + image_arg)
    shapeMask = cv2.inRange(image, 0, 0)
    contours_draw, hierarchy = cv2.findContours(shapeMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Find the contours of the inclusions by passing shapeMask as argument
    
    mask = np.zeros(image.shape, np.uint8)
    mask.fill(255)
    thisMask = mask.copy()
    cv2.drawContours(thisMask, contours_draw, -1, (0, 0, 0), cv2.FILLED)
    
    # Satistical descripors we want to compute
    volume_fraction_contours = np.count_nonzero(thisMask == 0) / (thisMask.shape[0]*thisMask.shape[1])
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
        hull =cv2.convexHull(cnt)
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
        #f.write('Solidity :\n')
        #f.write(str(solidity) + '\n')
    
    
    
    
    
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
        #f.write('Solidity :\n')
        #f.write(str(solidity_moments) + '\n')
      

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
    with open(full_name) as f :
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



def plot_stiffness() :
    tensors = os.listdir(consistent_path)
    if '.DS_Store' in tensors :
        tensors.remove('.DS_Store')
    
    nb_descriptors = len(descriptors_name)
    
 
    #[descriptor[moments]]        
    descriptors = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
    stiffness = [[[],[],[]],[[],[],[]],[[],[],[]]]
      
    for tensor in tensors :
        
        name = tensor.split('_')[0]
        
        t = read_tensor(tensor)
        
        for i in range(3) :
            for j in range(3) :
                stiffness[i][j].append(float(t[i][j]))
        
        distr_name = imgchar_path + '/' + name + '/' + name + '_distributions.txt'
        distr = read_distributions(distr_name)
  
        for k in range(nb_descriptors) :
            descriptors[k][0].append(distr[k][1][1])
            descriptors[k][1].append(distr[k][1][2])
            descriptors[k][2].append(distr[k][1][3])
    '''       
    for k in range(nb_descriptors) :
        for moment in range(3) :
            
            for m in range(3) :
                for n in range(3) :
                    x = descriptors[k][moment]
                    y = stiffness[m][n]
                    
                    x_sorted, y_sorted = zip(*sorted(zip(x,y)))
                    
                    
                    a, b = indexm_to_indext(m,n)
                    
                    plt.plot(x_sorted, y_sorted, 'r+')
                    title = 'C' + str(a) + str(b) + ' '+ descriptors_name[k] + ' : Moment of order ' + str(moment+1)
                    plt.title(title)
            
                    plt.show()
    '''
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
           
        
        
# Reads the consistent stiffness tensor from the .txt file
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
    distances = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
   
    
    # For each descriptor
    for i in range(nb_descriptors) :
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
def plot_errorpropag(mode) :
    consistent_tensors = os.listdir(consistent_path)
    if '.DS_Store' in consistent_tensors :
        consistent_tensors.remove('.DS_Store')
    
    name = errorpropag_path + '/' + mode
    
    if mode == 'groundtruth' :
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
                    
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                for p in range(4) :
                    x = lines[0][p]
                    y = lines[-1]
                    
                    
                    h = p // 2
                    if h == 0 :
                        g = p-h
                    elif h == 1 :
                        g = p-h-1
                    
                    axs[h, g].plot(x, y, 'r+')
                    axs[h, g].set_title('Moment of order ' + str(p))
                
                plt.tight_layout()
                title = c + ' : ' + rm_ext(d)
                fig.suptitle(title)
                plt.savefig(graph_path + '/' + title + ' ' + mode + '.jpg')
                plt.show()  
    
    
    
def quit_fen(fen) :
    plt.close('all')
    fen.quit()
    fen.destroy()


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
    height = 480
    
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
    
    convert_btn = tk.Button(fen, text='Convert images to meshes with Moose', command= lambda : mesh_images_init(fen))
    convert_btn.pack()
    
    run_btn = tk.Button(fen, text='Run simulations on the meshes', command= lambda : run_simulations_init(fen))
    run_btn.pack()
    
    read_btn = tk.Button(fen, text='Compute consistent tensors', command= lambda : compute_results(fen))
    read_btn.pack()
    
    plotstiff = tk.Button(fen, text='Plot the stiffness as a function of the moments of the descriptors', command=plot_stiffness)
    plotstiff.pack()
    
    computeground_btn = tk.Button(fen, text='Compute distances between moments of descriptors and associated difference of stiffnesses relative to a ground truth', command= lambda : compute_errorpropag_ref(fen))
    computeground_btn.pack()
    
    graphground_btn = tk.Button(fen, text='Plot ground truth error propagation', command= lambda : plot_errorpropag('groundtruth'))
    graphground_btn.pack()
    
    computepairwise_btn = tk.Button(fen, text='Compute distances between moments of descriptors and associated difference of stiffnesses pairwise', command= lambda : compute_pairwise_init(fen))
    computepairwise_btn.pack()
    
    graphpairwise_btn = tk.Button(fen, text='Plot pairwise error propagation', command= lambda : plot_errorpropag('pairwise'))
    graphpairwise_btn.pack()
    
    help_btn = tk.Button(fen, text='Help', command=help)
    help_btn.pack()
    
    quit_btn = tk.Button(fen, text='Quit', command= lambda : quit_fen(fen))
    quit_btn.pack()
    
    fen.mainloop()
  

main()


