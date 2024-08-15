import os
import subprocess
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from scipy.stats import (wasserstein_distance, wasserstein_distance_nd)
from threading import Thread
import pandas as pd

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
descriptors_name = ['Aspect ratio', 'Rectangularity', 'Size', 'Orientation', 'Solidity'] # List of descriptors used. Be careful with the order
descriptors_max = {'aspect_ratio' : 1, 'extent' : 1, 'size': 50, 'orientation' : 180, 'solidity' : 1, 's2' : 1, 'cluster' : 1, 'lp' : 1, 'cld angular' : 1}  # The maximum value of a descriptor. Useful for plotting. Maybe replace it with a case by case max search. 
# Max size to be discussed
descriptor_sampling = 20 # Number of classes in which to divide each descriptor
correlation_descriptors = ['S2', 'Cluster', 'Lineal Path', 'Angular Chord Length Distribution']


samples_names = ['REAL', 'SYNR', 'SYNO']
perturbations_names = ['ANI-20', 'ANI-40', 'ISO-20', 'ISO-40', 'DEL-20', 'DEL-40']
tests_names = perturbations_names.copy()
tests_names.insert(0, 'REF')

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
                S2 = np.load(os.path.join(imgchar_path, img, img + '_s2.npy'))
                norm_s2 = np.linalg.norm(S2, ord=norms_to_compute[k])
                norms_s2.append(norm_s2)
                
                cluster = np.load(os.path.join(imgchar_path, img, img + '_cluster.npy'))
                norm_c = np.linalg.norm(cluster, ord=norms_to_compute[k])
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
                plt.savefig(graphstiffness_path + '/' + title + '.png')
                plt.show() 
    
    
    elif mode == 'dir' :
        
        # Direction, moment
        x_s2 = [[[],[],[]],[[],[],[]]]
        x_c = [[],[],[]],[[],[],[]]
        x_lp = [[],[],[]],[[],[],[]]
        
        for img in to_plot :
            
                S2 = np.load(os.path.join(imgchar_path, img, img + '_s2.npy'))
                cluster = np.load(os.path.join(imgchar_path, img, img + '_cluster.npy'))
                lp = np.load(os.path.join(imgchar_path, img, img + '_lp.npy'))
                
                s2_max = descriptors_max['s2']
                cluster_max = descriptors_max['cluster']
                lp_max = descriptors_max['lp']
                
                s2_distr_1 = distribution_descriptor(S2, s2_max)
                cluster_distr_1 = distribution_descriptor(cluster, cluster_max)
                lp_distr_1 = distribution_descriptor(lp[:,lp.shape[1]//2], lp_max)
                
                s2_moments_1 = compute_moments(s2_distr_1,'s2')
                cluster_moments_1 = compute_moments(cluster_distr_1, 'cluster')
                lp_moments_1 = compute_moments(lp_distr_1, 'lp')
                
                s2_distr_2 = distribution_descriptor(S2, s2_max)
                cluster_distr_2 = distribution_descriptor(cluster, cluster_max)
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
                    plt.savefig(graphstiffness_path + '/' + title + '.png')
                    plt.show()        
        
          
                 
        
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

# Returns the distances between the p-th order moments and the assiocated difference between stiffnesses
# The distances variable is in the form [[descriptor1_0th, descriptor2_1th, ...], [descriptor2_0th, ...], ...]
def compute_error(t1_name, t2_name, m, n) :
    name1 = t1_name.split('_')[0]
    name2 = t2_name.split('_')[0]

    name1 = imgchar_path + '/' + name1 + '/' + name1 + '_distributions.txt'
    name2 = imgchar_path + '/' + name2 + '/' + name2 + '_distributions.txt'
        
    tensor1 = os.path.join(consistent_path, t1_name)
    tensor2 = os.path.join(consistent_path, t2_name)
    
    return compute_error_fullname(tensor1, tensor2, name1, name2, m, n)
    
    
    
def compute_error_fullname(t1, t2, name_distr1, name_distr2, m, n) :
    
    distr1 = read_distributions(name_distr1)
    distr2 = read_distributions(name_distr2)
    
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
            else :
                distances[i][k] = abs(distr1[i][1][k] - distr2[i][1][k])
    
    tensor1 = read_tensor_fullname(t1)
    tensor2 = read_tensor_fullname(t2)
    
    delta = abs(float(tensor1[m][n]) - float(tensor2[m][n]))
    
    
    return [distances,delta]
    
    

def compute_error_vf(t1_name, t2_name, m, n) :
    name1 = t1_name.split('_')[0]
    name2 = t2_name.split('_')[0]

    name1 = imgchar_path + '/' + name1 + '/' + name1 + '_characteristics.txt'
    name2 = imgchar_path + '/' + name2 + '/' + name2 + '_characteristics.txt'
    
    tensor1 = os.path.join(consistent_path, t1_name)
    tensor2 = os.path.join(consistent_path, t2_name)
    
    return compute_error_vf_fullname(tensor1, tensor2, name1, name2, m, n)
    
   

def compute_error_vf_fullname(t1, t2, name_char1, name_char2, m, n) :
    
    v1 = read_vf(name_char1)
    v2 = read_vf(name_char2)

    delta_v = abs(v2-v1)
    
    tensor1 = read_tensor_fullname(t1)
    tensor2 = read_tensor_fullname(t2)


    delta_t = abs(float(tensor1[m][n]) - float(tensor2[m][n]))
    
    return [delta_v, delta_t]


def compute_error_correlation(t1_name, t2_name, m, n) :
    name1 = t1_name.split('_')[0]
    name2 = t2_name.split('_')[0]
    
    dir1 = imgchar_path + '/' + name1 
    dir2 = imgchar_path + '/' + name2
    
    t1 = consistent_path + '/' + t1_name
    t2 = consistent_path + '/' + t2_name
    
    return compute_error_correlation_fullname(t1, t2, dir1, dir2, m, n)


def compute_error_correlation_fullname(t1, t2, dir1, dir2, m, n) :
    
    img_code1 = os.path.basename(t1).split('_')[0]
    img_code2 = os.path.basename(t2).split('_')[0]
    
    path1 = dir1 + '/' + img_code1
    path2 = dir2 + '/' + img_code2
    
    nb_correlation_descriptors = len(correlation_descriptors)
    
    auto_correlation1 = np.load(path1 + '_s2.npy')
    auto_correlation2 = np.load(path2 + '_s2.npy')
    cluster1 = np.load(path1 + '_cluster.npy')
    cluster2 = np.load(path2 + '_cluster.npy')
    lp_1 = np.load(path1 + '_lp.npy')
    lp_2 = np.load(path2+ '_lp.npy')
    cld_angular_1 = np.load(path1 + '_cld_angular.npy', allow_pickle=True)
    cld_angular_2 = np.load(path2 + '_cld_angular.npy', allow_pickle=True)
    
    pdfs_cld_1 = [d.pdf for d in cld_angular_1]
    pdfs_cld_2 = [d.pdf for d in cld_angular_2]
    
    # To adapt depending on the number of correlation descriptors
    emd = nb_correlation_descriptors * [0]
    emd[0] = wasserstein_distance_nd(auto_correlation1[0,:,:,0], auto_correlation2[0,:,:,0])
    emd[1] = wasserstein_distance_nd(cluster1[:,:,0], cluster2[:,:,0])
    emd[2] = wasserstein_distance_nd(lp_1, lp_2)
    emd[3] = wasserstein_distance_nd(pdfs_cld_1, pdfs_cld_2)
    
    tensor1 = read_tensor_fullname(t1)
    tensor2 = read_tensor_fullname(t2)
    
    delta_t = abs(float(tensor1[m][n]) - float(tensor2[m][n]))
    
    return [emd, delta_t]
    
      
        

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
            x = [[], [], [], []]
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
    
    if mode == 'groundtruth' :
        gt = filedialog.askdirectory(title='Choose a ground truth', initialdir=name)
        if not ('groundtruth' in name) :
            popup('Invalid input')
            return
        
        gt = os.path.basename(gt)
        
        name = name + '/' + gt
      
        plot_errorpropag_fullname(fen, name, graph_path, groundtruh=gt)  
    
    else :
        plot_errorpropag_fullname(fen, name, graph_path)
    
def plot_errorpropag_fullname(fen, name, dir_to_save, groundtruth=None) :
    
    percentage = False
    
    if groundtruth != None :
        percentage = askPercentage(fen)
        mode = 'groundtruth'
    else :
        mode = 'pairwise'
    
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
                        t = read_tensor(groundtruth + '_effective_stiffness.txt')
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
                    
                    
                    plt.savefig(dir_to_save + '/' + title + ' ' + mode + '.png')
                    plt.show()
                    
                else :    
                    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                    for p in range(4) :
                        x = lines[0][p]
                        y = lines[-1]
                        
                        if percentage : 
                            t = read_tensor(groundtruth + '_effective_stiffness.txt')
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
                    
                
                    fig.suptitle(title)
                    
                    
                    plt.savefig(dir_to_save + '/' + title + ' ' + mode + '.png')
                    
                    plt.show()  
    
             
def rank_descriptors(fen) :
    consistent_tensors = os.listdir(consistent_path)
    if '.DS_Store' in consistent_tensors :
        consistent_tensors.remove('.DS_Store')
    
    name = errorpropag_path + '/groundtruth'
    
    
    rank_descriptors_file_errorpropag(fen, name, 'b')
    
    plt.show()






def rank_descriptors_file_errorpropag(name, coeff, strings_input=[]) :
    gt = name
    
    if not ('groundtruth' in gt or 'comparison' in gt) :
        popup('Invalid input')
        return
    
    
    correlations_spearman = []
    associated_strings_spearman = []
    
    
    
    name_c = gt + '/' + coeff
    
    for d in descriptors_name :
 
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
                
                correlations_spearman.append(spearman)
                if p == 0 :
                    associated_strings_spearman.append(coeff + ' : ' + str(d) + ' EMD : ' + ' Spearman correlation coefficient : ')
                else : 
                    associated_strings_spearman.append(coeff + ' : ' + str(d) + ' moment of order :' + str(p) + ' Spearman correlation coefficient : ')
                
    descriptors = correlation_descriptors.copy()
    if not 'Volume fraction' in descriptors :
        descriptors.append('Volume fraction')
    
    for d in descriptors :      
        name_d = name_c + '/' + d + '.txt'
    
        with open(name_d, 'r') as f :
            
            read_lines = f.readlines()
            lines = []
            for l in read_lines :
                lines.append(eval(l))

            x = lines[0]
            y = lines[-1]
            
            data = {
                'x' : x,
                'y' : y
            }
            df = pd.DataFrame(data)
            
            spearman = df['y'].corr(df['x'], method='spearman')
            
            correlations_spearman.append(spearman)

            associated_strings_spearman.append(coeff + ' : ' + str(d) + ' EMD : ' + ' Spearman correlation coefficient : ')
              
    tuples_spearman = list(zip(correlations_spearman, associated_strings_spearman))   
    sorted_tuples_abs_spearman = sorted(tuples_spearman, key= lambda x: abs(x[0]))
    
    corr_list_spearman = []
    string_list_spearman = []
    
    
    if strings_input != [] :
        string_input_spearman = strings_input
    

        
    for corr, string in sorted_tuples_abs_spearman :
        string_list_spearman.append(string)
        corr_list_spearman.append(corr)    
        
    if strings_input != [] :
        
        tmp_corr = []
        tmp_str = []
        
        for string_i in string_input_spearman :
            
            index = string_list_spearman.index(string_i)
            tmp_corr.append(corr_list_spearman[index])
            tmp_str.append(string_list_spearman[index])
            
        corr_list_spearman = tmp_corr
        string_list_spearman= tmp_str
            
            
    x = np.arange(1, len(corr_list_spearman)+1, 1)
    
    
    label_spearman = [str(i+1) + ' ' +  "".join(s.split('Spearman')[0].split(':')[1:3])  for i, s in enumerate(reversed(string_list_spearman))]
    
    label_spearman = convert_moment(label_spearman)
    
    
    
    bars = plt.barh(label_spearman[::-1], corr_list_spearman, label=label_spearman, color='skyblue')


    plt.suptitle('Spearman correlation to the stiffness')

    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.02
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')
    
    plt.xticks(np.arange(-1, 1.1, 0.25), labels=['-1', '-0.75', '-0.5', '-0.25', '0', '0.25', '0.5', '0.75', '1'])
    
    
    return string_list_spearman


    
def rank_descriptors_file(name, color, coeff, marker, label, strings_input=[]) :
    
   
    indexm = indext_to_indexm(coeff[1:5])
   
    correlations_spearman = []
    associated_strings_spearman = []
    
    consistent_tensors = os.listdir(name + '/consistent_tensors')
    if '.DS_Store' in consistent_tensors :
        consistent_tensors.remove('.DS_Store')
    consistent_tensors_full = [os.path.join(name, 'consistent_tensors', t) for t in consistent_tensors]
    
    
    stiffness_list = []
    
    
    
    for path in consistent_tensors_full :
  
        
        coeff_value = float(read_tensor_fullname(path)[indexm[0]][indexm[1]])
        stiffness_list.append(coeff_value)
    
    for d, descr in enumerate(descriptors_name) :
        for p in range(1,4) :
            
            x = []
            
            for path in consistent_tensors_full :    
            
                tensor_name = os.path.basename(path).split('_')[0]
                img_char_path = os.path.join(name, 'images_characteristics', tensor_name)
                
                distributions = read_distributions(os.path.join(img_char_path,  tensor_name + '_distributions.txt'))
                
                
                x.append(distributions[d][1][p])
            
            
            data = {
                'x' : x,
                'y' : stiffness_list
            }
            df = pd.DataFrame(data)
            
            spearman = df['y'].corr(df['x'], method='spearman')
            
            
            correlations_spearman.append(spearman)
            
            associated_strings_spearman.append(coeff + ' : ' + descr + ' moment of order : ' + str(p) + ' Spearman correlation coefficient : ')
    
    vfs = []
    for path in consistent_tensors_full :    
            
        tensor_name = os.path.basename(path).split('_')[0]
        path = os.path.join(name, 'images_characteristics', tensor_name, tensor_name + '_characteristics.txt')
        vfs.append(read_vf(path))
        
    data = {
        'x' : vfs,
        'y' : stiffness_list
    }
    df = pd.DataFrame(data)
    
    spearman = df['y'].corr(df['x'], method='spearman')
    
    
    correlations_spearman.append(spearman)
    
    associated_strings_spearman.append(coeff + ' : Volume fraction Spearman correlation coefficient : ')
    
    norms_s2 = []
    norms_c = []
    norms_lp = []
    norms_cld_angular = []
    means_cld_angular = []
        

    for path in consistent_tensors_full :    
        
        tensor_name = os.path.basename(path).split('_')[0]
        img_char_path = os.path.join(name, 'images_characteristics', tensor_name)
        
        auto_correlation = np.load(os.path.join(img_char_path, tensor_name + '_s2.npy'))
        norm_s2 = np.linalg.norm(auto_correlation, ord='fro')
        norms_s2.append(norm_s2)
        
        cluster = np.load(os.path.join(img_char_path, tensor_name + '_cluster.npy'))
        norm_c = np.linalg.norm(cluster, ord='fro')
        norms_c.append(norm_c)
        
        lp = np.load(os.path.join(img_char_path, tensor_name + '_lp.npy'))
        norm_lp = np.linalg.norm(lp, ord='fro')
        norms_lp.append(norm_lp)
    
        cld_angular = np.load(os.path.join(img_char_path, tensor_name + '_cld_angular.npy'), allow_pickle=True)
        pdfs_cld_angular = [d.pdf for d in cld_angular]
        norm_cld_angular = np.linalg.norm(pdfs_cld_angular, ord='fro')
        mean_cld_angular = np.mean(pdfs_cld_angular)
        means_cld_angular.append(mean_cld_angular)
        norms_cld_angular.append(norm_cld_angular)
        
    norms = [[norms_s2, 'S2'], [norms_c, 'Cluster'], [norms_lp, 'Lineal Path'], [norms_cld_angular, 'CLDa']]   
    
    for norm in norms :
        data = {
            'x' : norm[0],
            'y' : stiffness_list
        }
        df = pd.DataFrame(data)
        spearman = df['y'].corr(df['x'], method='spearman')
        
        correlations_spearman.append(spearman)
        associated_strings_spearman.append(coeff + ' : ' + norm[1] + ' Frobenius norm' + ' Spearman correlation coefficient : ')
  
    data = {
        'x' : means_cld_angular,
        'y' : stiffness_list
    }
    df = pd.DataFrame(data)
    
    spearman = df['y'].corr(df['x'], method='spearman')
    
    correlations_spearman.append(spearman)
    associated_strings_spearman.append(coeff + ' : CLDa moment of order : 1 Spearman correlation coefficient : ')
                    
    
    tuples_spearman = list(zip(correlations_spearman, associated_strings_spearman))   
    sorted_tuples_abs_spearman = sorted(tuples_spearman, key= lambda x: abs(x[0]))
    
    
    corr_list_spearman = []
    string_list_spearman = []
    
    
    if strings_input != [] :
        string_input_spearman = strings_input
    
    for corr, string in sorted_tuples_abs_spearman :
        string_list_spearman.append(string)
        corr_list_spearman.append(corr)    
        
    if strings_input != [] :
        
        
        tmp_corr = []
        tmp_str = []
        
        for string_i in string_input_spearman :
            
            index = string_list_spearman.index(string_i)
            tmp_corr.append(corr_list_spearman[index])
            tmp_str.append(string_list_spearman[index])
            
        corr_list_spearman = tmp_corr
        string_list_spearman= tmp_str
            
            
    x = np.arange(1, len(corr_list_spearman)+1, 1)
    

    for i, (x_val, y_val) in enumerate(zip(x, corr_list_spearman), 1) :
        plt.plot(x_val, y_val, marker = marker, color=color, label=label, linestyle='--')
    plt.legend(convert_moment([str(i+1) + ' ' +  "".join(s.split('Spearman')[0].split(':')[1:3])  for i, s in enumerate(reversed(string_list_spearman))]), loc='center left', bbox_to_anchor=(1, 0.5), title='Legend')
    
    
    plt.xlabel('Descriptor index')
    plt.xticks(np.arange(1, len(corr_list_spearman) + 1), labels=np.arange(len(corr_list_spearman), 0, -1))
    plt.ylabel('Spearman correlation coefficient')
   
    return string_list_spearman
    
    
def plot_stiffness_file(working_folder, descr, color) :
    
    consistent = os.path.join(working_folder, 'consistent_tensors')
    tensors = os.listdir(consistent)
    
    if '.DS_Store' in tensors :
        tensors.remove('.DS_Store')
    
    nb_descriptors = len(descriptors_name)
    
    #[descriptor[moments]]
    # To adapt depending on the number of descriptors        
    descriptors = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
    stiffness = [[[],[],[]],[[],[],[]],[[],[],[]]]
    
    vf = []
    
    characteristics = os.path.join(working_folder, 'images_characteristics')
    
    for tensor in tensors :
        
        name = tensor.split('_')[0]
        
        t = read_tensor_fullname(os.path.join(consistent, tensor))
        
        for i in range(3) :
            for j in range(3) :
                stiffness[i][j].append(float(t[i][j]))
        

        distr_name = characteristics + '/' + name + '/' + name + '_distributions.txt'
        distr = read_distributions(distr_name)

        vf.append(read_vf(characteristics + '/' + name + '/' + name + '_characteristics.txt'))
        
        for k in range(nb_descriptors) :
            descriptors[k][0].append(distr[k][1][1])
            descriptors[k][1].append(distr[k][1][2])
            descriptors[k][2].append(distr[k][1][3])
    
    k = descriptors_name.index(descr)
    moment = 0
        
    for m in range(3) :
        for n in range(3) :
            x = descriptors[k][moment]
            y = stiffness[m][n]
            
            fig = plt.gcf()
            axs = fig.get_axes()
            
            a, b = indexm_to_indext(m,n)
            
            axs[3*m + n].plot(x, y, 'r+', color=color)
            axs[3*m + n].set_title('C' + str(a) + str(b))
        
        plt.tight_layout()
   
  
    
def window_descr_rank(fen, reversed_tuples, coeff, save) :    
    p = tk.Toplevel(fen)
    p.title("Correlation of descriptors for error propagation")

    width = 900
    height = 300
    
    screen_width = fen.winfo_screenwidth()
    screen_height = fen.winfo_screenheight()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    p.geometry(f'{width}x{height}+{x}+{y}')
    
    
    frame = ttk.Frame(p, width=800, height=600)
    frame.pack(padx=10, pady=10)

    
    listbox = tk.Listbox(frame, width=800)
    listbox.pack(side=tk.LEFT, fill=tk.X)

    
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.BOTH)
    listbox.config(yscrollcommand=scrollbar.set)

    if save :
        name = '/Users/tmr96/Documents/Other/EMI/' + coeff + '.txt'
        create_file(name)
    
        with open(name, 'a') as f :
            
            for corr, string in reversed_tuples :
                f.write(string + str(corr) + '\n')
                listbox.insert(tk.END, f"{string}: {corr}")
            f.write('\n\n')
    else : 
        for corr, string in reversed_tuples :
            listbox.insert(tk.END, f"{string}: {corr}")

    
    button_ok = ttk.Button(p, text="OK", command=p.destroy)
    button_ok.pack(pady=10)
        
        
    
   

    
def compute_comparison(fen, trial1, trial2) :
    dir_name = trial2
    comparison_dir = dir_name +'/comparison'
    tensors_file = comparison_dir + '/compared_tensors.txt'
    
    
    nb_descriptors = len(descriptors_name)
    
    overwrite_bool = False
    
    if os.path.exists(tensors_file) :
        overwrite_bool = overwrite(fen)
        if overwrite_bool :
            subprocess.run(['rm', tensors_file])
            subprocess.run(['rm', comparison_dir])
            #os.remove(comparison_dir)
            #os.remove(tensors_file)
        
    
    
    time_start = time.time()
    
    create_dir(comparison_dir)
    
    tensors_trial1 = consistent_tensors_folder(trial1)
    tensors_trial2 = consistent_tensors_folder(trial2)
    to_compare = []
    for t in tensors_trial1 :
        if t in tensors_trial2 :
            to_compare.append(t)
            
    create_file(tensors_file)
    
    with open(tensors_file, 'w') as f :
        for t in to_compare :
            f.write(t + '\n')
                
    
    
    for m in range(3) :
        for n in range(3) :
            a, b = indexm_to_indext(m,n)
            coefficient = 'C' + str(a) + str(b)
            name_coeff = comparison_dir + '/'+ coefficient
            if not create_dir(name_coeff) :
                continue
            
            for k in range(nb_descriptors) :
                # I don't know why but x = 4*[[]] does not work (the elements are appened to every list)
                x1 = [[], [], [], []]
                y1 = []
                x2 = [[], [], [], []]
                y2 = []
                
                for t in to_compare :
                    
                    t1 = os.path.join(trial1, 'consistent_tensors', t + '_effective_stiffness.txt')
                    t2 = os.path.join(trial2, 'consistent_tensors', t + '_effective_stiffness.txt')
                    
                
                    distr1 = os.path.join(trial1, 'images_characteristics', t, t + '_distributions.txt')
                    distr2 = os.path.join(trial2, 'images_characteristics', t, t + '_distributions.txt')
            
                    distr3 = os.path.join(trial1, 'images_characteristics', t, t + '_distributions_adim.txt')
                    distr4 = os.path.join(trial2, 'images_characteristics', t, t + '_distributions_adim.txt')
                    
                    point1 = compute_error_fullname(t1, t2, distr1, distr2, m, n)
                    point2 = compute_error_fullname(t1, t2, distr3, distr4, m, n)
                    # Up to third moment
                    for p in range(4) :
                        x1[p].append(point1[0][k][p])
                        x2[p].append(point2[0][k][p])
                    y1.append(point1[-1])
                    y2.append(point2[-1])

               
                name_coeff_descr1 = name_coeff + '/' + descriptors_name[k] + '.txt'
                name_coeff_descr2 = name_coeff + '/' + descriptors_name[k] + '_adim.txt'
                create_file(name_coeff_descr1)
                create_file(name_coeff_descr2)
                
                with open(name_coeff_descr1, 'w') as f :
                    f.write(str(x1) + '\n')
                    f.write(str(y1) + '\n')
                    
                with open(name_coeff_descr2, 'w') as f :
                    f.write(str(x2) + '\n')
                    f.write(str(y2) + '\n')
                        
            x = []
            y = []        
            for t in to_compare :
                
                t1 = os.path.join(trial1, 'consistent_tensors', t + '_effective_stiffness.txt')
                t2 = os.path.join(trial2, 'consistent_tensors', t + '_effective_stiffness.txt')
                
                char1 = os.path.join(trial1, 'images_characteristics', t, t + '_characteristics.txt')
                char2 = os.path.join(trial2, 'images_characteristics', t, t + '_characteristics.txt')
                
                point = compute_error_vf_fullname(t1, t2, char1, char2, m, n)
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
            for t in to_compare :    
                t1 = os.path.join(trial1, 'consistent_tensors', t + '_effective_stiffness.txt')
                t2 = os.path.join(trial2, 'consistent_tensors', t + '_effective_stiffness.txt')
                
                dir1 = os.path.join(trial1, 'images_characteristics', t)
                dir2 = os.path.join(trial2, 'images_characteristics', t)
                
                point = compute_error_correlation_fullname(t1, t2, dir1, dir2, m, n)
                x[0].append(point[0][0])
                x[1].append(point[0][1])
                x[2].append(point[0][2])
                x[3].append(point[0][3])
                y.append(point[1])    
                
                print( (m, n, t) )
                
            for k in range(len(correlation_descriptors)) :
                
                name_coeff_descr = name_coeff + '/' + correlation_descriptors[k] + '.txt'
                create_file(name_coeff_descr)
                
                with open(name_coeff_descr, 'w') as f :
                    f.write(str(x[k]) + '\n')
                    f.write(str(y) + '\n')    
                
        
    time_end = time.time()
    popup('Computation done', time_end - time_start)
   


def compare_trials_init(fen) :
    trial1 = filedialog.askdirectory(initialdir=simulation_dir)
    trial2 = filedialog.askdirectory(initialdir=simulation_dir)
    
    if not trial1.split('/')[-2] in samples_names or not trial2.split('/')[-2] in samples_names :
        popup('Invalid input !')
        return
    
    if not trial1.split('/')[-1] in tests_names or not trial2.split('/')[-1] in tests_names :
        popup('Invalid input !')
        return

    compare_trials_fen(fen, trial1, trial2)
    

def evolution_stiffness(fen, trial1) :
    popup('The following graph shows the evolution of the correlation to the stiffness for each descriptor.\n \
        (Your Trial 2 input is not taken into account.)')
    
    
    
    #adim_bool = overwrite(fen, 'Would you like to see the results as a function of the CoV, std, and skw coeff ?')
    
    again = True
    
    # To adapt depending on the number of perturbations
    colors = ['aquamarine', 'blue', 'coral', 'red', 'orchid', 'fuchsia']
    #['ANI-20', 'ANI-40', 'ISO-20', 'ISO-40', 'DEL-20', 'DEL-40']
    markers = ['.', 'o', 'v', '^', 's', 'p']
    
    labels = ''
    for i in range(len(perturbations_names)) :
        
        labels += perturbations_names[i] + ' : ' + colors[i] + '\n'
        
    popup(labels)
        
    
    while again :
        again = False
        coeff = what_coeff(fen)
        coeff = 'C' + coeff
        
        #plt.subplots(1, 2)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.figure(figsize=(12,9))
        
      
        #string_list_1 = rank_descriptors_v3_file(fen, trial1, 'black', label=os.path.basename(trial1), coeff=coeff, adim_bool=adim_bool)
        
        string_list = rank_descriptors_file(trial1, 'black', coeff, '+', os.path.basename(trial1))
        
        trials = perturbations_names
        
        
        
        for i, trial in enumerate(trials) :
            name = trial1.replace(os.path.basename(trial1), '') + trial
            try :
                #rank_descriptors_v3_file(fen, name, colors[i], label=os.path.basename(name), strings_input=string_list_1, coeff=coeff, adim_bool=adim_bool)
                rank_descriptors_file(name, colors[i], coeff, markers[i], os.path.basename(name), strings_input=string_list)
            except FileNotFoundError as e :
                popup('File not found : ' + e.filename)
            finally :
                continue
        
       
        
   
        
        plt.suptitle('Correlation of each descriptor to the stiffness component ' + coeff)
        #plt.subplots_adjust(wspace=0.36, hspace=0.4, left=0.037, right=0.89)
        
        name = trial1.split('/')[-2] + '-EVOL-' + coeff + '.png'
        path = os.path.join('/Users/tmr96/Documents/Graphs folder', name)
        plt.tight_layout()
        plt.savefig(path)
        
        plt.show()
        
        
        
        
        
        again = overwrite(fen, 'Would you want to see the results for another coefficient ?')
    
def compare_errorpropag(fen, trial1, trial2) :
    
    dir_name = trial2
    comparison_dir = dir_name +'/comparison'
    graphs_dir = dir_name + '/graphs_comparison'
    
    adim_bool = overwrite(fen, 'Would you like to see the results as a function of the CoV, std, and skw coeff ?')
    
    if overwrite(fen, label='Would you want to see all the graphs ?') : 
        create_dir(graphs_dir)
        plot_errorpropag_fullname(fen, comparison_dir, graphs_dir)
    
    again = True
    
    while again :
        again = False
    
        coeff = what_coeff(fen)
        coeff = 'C' + coeff
        
        screen_width = fen.winfo_screenwidth()
        screen_height = fen.winfo_screenheight()
        dpi = 100.0  
        fig_width = screen_width / dpi
        fig_height = screen_height / dpi
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        try : 
            #rank_descriptors_v3_file_errorpropag_spearman(fen, comparison_dir, color='blue', coeff=coeff, save=True, adim_bool=adim_bool)
            rank_descriptors_file_errorpropag(comparison_dir, coeff)
        except FileNotFoundError as e :
            popup('Correlation not computed for this coefficient !')
            print(e.filename)
            return
        
        fig.suptitle('Correlation of each descriptor to the error propagation for ' + coeff)
    
        create_dir(graphs_dir)
        name = trial1.split('/')[-2] + '-' + os.path.basename(trial2) + '-' + coeff
        
        if adim_bool :
            name += '_adim'
        
        name += '.png'
        
        path = os.path.join('/Users/tmr96/Documents/Graphs folder', name)
        
        
        plt.tight_layout()
        plt.savefig(path)
        plt.show()
    
        again = overwrite(fen, 'Do you want to see the results for another coefficient ?')
    
def compare_plot_stiffness(fen, trial1, trial2) :
    
    again = True
    while again :
        again = False
        descr = what_descriptor(fen)

        
        fig, axs = plt.subplots(3, 3, figsize=(14, 12))
        
        plot_stiffness_file(trial1, descr, 'red')
        plot_stiffness_file(trial2, descr, 'blue')
        
        code1 = ' '.join(trial1.split('/')[-2::1])
        code2 = ' '.join(trial2.split('/')[-2::1])
        
        title = f'Stiffness as a function of the {descr} for simulations {code1} and {code2}'
        
       
        fig.suptitle(title)
        
        
        plt.show()

        again = overwrite(fen, 'Do you want to see the results for another descriptor ?')
    

def compare_plot_stiffness_all(fen, trial1) :
    
    popup('The following graph shows the evolution of the correlation to the stiffness for each descriptor.\n \
        (Your Trial 2 input is not taken into account.)')
    
    again = True
    
    
    trials = [trial1.replace('REF', '') + p for p in perturbations_names]
    # To adapt depending on the number of perturbations
    colors = ['aquamarine', 'blue', 'coral', 'red', 'orchid', 'fuchsia']
    #['ANI-20', 'ANI-40', 'ISO-20', 'ISO-40', 'DEL-20', 'DEL-40'] 
    
    while again :
        again = False
        
        descr = what_descriptor(fen)
        
        fig, axs = plt.subplots(3, 3, figsize=(14, 12))
        
        plot_stiffness_file(trial1, descr, 'black')
        
        for x, t in enumerate(trials) :
            try :
                plot_stiffness_file(t, descr, colors[x])
            except FileNotFoundError as e :
                popup('File not found : ' + e.filename)
                continue
        
        code1 = ' '.join(trial1.split('/')[-2:-1:1])
        
        title = f'Stiffness as a function of the {descr} for simulation {code1}'
        
       
        fig.suptitle(title)
        
        
        plt.show()

        again = overwrite(fen, 'Do you want to see the results for another descriptor ?')
    
    
    return

     
  
    
def compare_trials_fen(fen, trial1, trial2) :
    
    code1 = ' '.join(trial1.split('/')[-2::1])
    code2 = ' '.join(trial2.split('/')[-2::1])
    
    p = tk.Toplevel(fen)
    
    p.wm_title(f'Compare {code1} and {code2}')
    
    compute_btn = tk.Button(p, text='Run the computation', command= lambda : compute_comparison(p, trial1, trial2))
    compute_btn.pack()
    
    evolution_btn = tk.Button(p, text='Show the evolution of the correlation to the stiffness', command= lambda : evolution_stiffness(p, trial1))
    evolution_btn.pack()
    
    errorpropag_btn = tk.Button(p, text='Show the correlation to the error propagation', command= lambda : compare_errorpropag(p, trial1, trial2))
    errorpropag_btn.pack()
    
    stiffness_btn = tk.Button(p, text='Plots the stiffness vs the descriptors for one perturbation and the reference', command= lambda : compare_plot_stiffness(p, trial1, trial2))
    stiffness_btn.pack()
    
    stiffness_all_btn = tk.Button(p, text='Plots the stiffness vs the descriptors for all the perturbations and the reference', command= lambda : compare_plot_stiffness_all(p, trial1))
    stiffness_all_btn.pack()
    
    fen.wait_window(p)


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
    
    rank_btn = tk.Button(fen, text='Rank descriptors', command= lambda : rank_descriptors(fen))
    rank_btn.pack()
    
    compare_btn = tk.Button(fen, text='Compare trials', command= lambda : compare_trials_init(fen))
    compare_btn.pack()
    
    help_btn = tk.Button(fen, text='Help', command=help)
    help_btn.pack()
    
    quit_btn = tk.Button(fen, text='Quit', command= lambda : quit_fen(fen))
    quit_btn.pack()
    
    fen.mainloop()
  

main()
