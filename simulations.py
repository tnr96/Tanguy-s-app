import os
import subprocess
import numpy as np
import time
import glob
import csv
import tkinter as tk
from tkinter import filedialog
from threading import Thread

from utilities import *

# Here you should define the paths on your computer 
working_dir = '/Users/tmr96/Documents/Automatic'
moose_dir = '/Users/tmr96/projects/my_files'
simulation_dir = '/Users/tmr96/Documents/Simulations' # Where the different simulations are stored 

# You don't need to change anything here
images_path = moose_dir + '/images' # Images folder
convert_script_path = working_dir + "/convert_moose.sh" # Script .sh
moose_app = 'solid_mechanics' # Moose app to run (assumed to be in moose_dir)
convert_path = images_path + '/convert_moose.i' # Moose input files
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
filtered_path = working_dir + '/filtered_images'


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
    create_dir(journals_path)
    create_dir(errorpropag_path)
    create_dir(errorpropag_path + '/pairwise')
    create_dir(errorpropag_path + '/groundtruth')
    create_dir(graph_path)
    create_dir(graphstiffness_path)
    #create_dir(extracted_path)

  

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


# Creates the .txt file to copy in Cubit command line
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
    tolerance_importance = 0.01  # If the '21' coefficients are too small compared to the others, the symmetry is not checked for them
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
    

# Reads the consistent stiffness tensor from the .txt file
# Please note that this is a string
def read_tensor(t) :
    return read_tensor_fullname(consistent_path + '/' + t)


def read_tensor_fullname(fullname) :
    with open(fullname, 'r') as f :
        lines = f.readlines()
        for i in range(len(lines)) :
            lines[i] = lines[i].split('\t')
            lines[i].remove('\n')
    return lines[0:3]

def read_vf(full_name) :
    vf = 0
    with open(full_name, 'r') as f :
        lines = f.readlines()
        split1 = lines[0].split('\t')
        split2 = split1[1].split(':')[1]
        split2.replace(' ', '')
        vf = float(split2)
    return vf



# Checks if a tensor lies within Hashin-Shtrikman's bounds
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
    


# To choose working directory
def choose_wd() :
    global working_dir
    working_dir = filedialog.askdirectory(initialdir=working_dir)
    

def main () :
    fen = tk.Tk()
    
    width = 800
    height = 600
    
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
    
    convert_btn = tk.Button(fen, text='Convert images to meshes with Moose', command= lambda : mesh_images_init(fen))
    convert_btn.pack()
    
    run_btn = tk.Button(fen, text='Run simulations on the meshes', command= lambda : run_simulations_init(fen))
    run_btn.pack()
    
    read_btn = tk.Button(fen, text='Compute consistent tensors', command= lambda : compute_results(fen))
    read_btn.pack()
    
    hsw_btn = tk.Button(fen, text='Check if a tensor lies within Hashin-Shtrikman\'s bounds', command=hsw)
    hsw_btn.pack()
    
    help_btn = tk.Button(fen, text='Help', command=help)
    help_btn.pack()
    
    quit_btn = tk.Button(fen, text='Quit', command= lambda : quit_fen(fen))
    quit_btn.pack()
    
    fen.mainloop()
  

main()
