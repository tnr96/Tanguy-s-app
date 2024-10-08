import os
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import cv2
from math import sqrt, radians, radians
import matplotlib.pyplot as plt
import sys
import GooseEYE

import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utilities import *

# Here you should define the paths on your computer 
working_dir = '/Users/tmr96/Documents/Automatic'
moose_dir = '/Users/tmr96/projects/my_files'

# You don't need to change anything here
images_path = moose_dir + '/images' # Images folder
imgchar_path = working_dir + '/images_characteristics' # Where the characteristics of the images are stored
filtered_path = working_dir + '/filtered_images'


cutoff_input = 160

# Solidity is removed for the moment
descriptors_name = ['Aspect ratio', 'Extent', 'Size', 'Orientation', 'Solidity'] # List of descriptors used. Be careful with the order
descriptors_max = {'aspect_ratio' : 1, 'extent' : 1, 'size': 50, 'orientation' : 180, 'solidity' : 1, 's2' : 1, 'cluster' : 1, 'lp' : 1, 'cld angular' : 1}  # The maximum value of a descriptor. Useful for plotting. Maybe replace it with a case by case max search. 
# Max size to be discussed
descriptor_sampling = 20 # Number of classes in which to divide each descriptor
correlation_descriptors = ['S2', 'Cluster', 'Lineal Path', 'Angular Chord Length Distribution']

# It is necessary to specify the cutoff beforehand if we want to calculate the stiffness as a function of S2 in a given direction



def init_dirs() :
    create_dir(imgchar_path)
    images = images_folder(images_path)
    for image in images :
        create_dir(os.path.join(imgchar_path, rm_ext(image)))


# Reomves all the inclusions with a 0 area for every image in the images_path folder
def filter_tiny() :
    images = images_folder(images_path)
    create_dir(filtered_path)
    
    for img in images :
        image = cv2.imread(images_path + '/' + img, cv2.IMREAD_UNCHANGED)
        shapeMask = cv2.inRange(image, 0, 0)
        contours, _ = cv2.findContours(shapeMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        image_filtered = np.zeros(image.shape, np.uint8)
        for number, cnt in enumerate(contours) : 
            area = cv2.contourArea(cnt)
            if area != 0 :
                cv2.drawContours(image_filtered, contours, number, (255,255,255), cv2.FILLED)
                
        name = os.path.join(filtered_path, img)
        
        cv2.imwrite(name, image)
    



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
# To adapt depening on the number of descriptors
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
    solidity_max = descriptors_max['solidity']
    
    # Find a way to improve this (standadize the names as key to dictionnary)
    # [descriptor][moments][order 0]
    x_aspect_ratio, aspect_ration_distr = clean_distribution(distributions[0][1][0], aspect_ratio_max)  
    x_extent, extent_distr = clean_distribution(distributions[1][1][0], extent_max)
    x_size, size_distr = clean_distribution(distributions[2][1][0], size_max)
    x_orientation, orientation_distr = clean_distribution(distributions[3][1][0], orientation_max)
    x_solidity, solidity_distr = clean_distribution(distributions[4][1][0], solidity_max)
    
    distributions_plot = [['Aspect ratio', x_aspect_ratio, aspect_ration_distr],['Extent', x_extent, extent_distr], ['Size', x_size, size_distr], ['Orientation', x_orientation, orientation_distr], ['Solidity', x_solidity, solidity_distr]]
    
    
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
        
        #x,y,w,h = cv2.minAreaRect(cnt)
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


  

def compute_lp(full_name) :
    img = cv2.imread(full_name, cv2.IMREAD_UNCHANGED)
    shapeMask = cv2.inRange(img, 0, 0)
    L = GooseEYE.L((shapeMask.shape[0], shapeMask.shape[1]), shapeMask)
    
    basename = os.path.basename(full_name)
    basename_rm = rm_ext(basename)
    
    np.save(os.path.join(imgchar_path, basename_rm, basename_rm + '_lp.npy'), L)



    

def compute_correlation(fen) :
    
    images = images_folder(images_path)
    if '.DS_Store' in images :
        images.remove('.DS_Store')
    

    images_lp = images.copy()
    
    already_done_lp = []
    
    directories = os.listdir(imgchar_path)
    
    if '.DS_Store' in directories :
        directories.remove('.DS_Store')
    
    for d in directories :
        for f in os.listdir(os.path.join(imgchar_path, d)) :
            if 'lp' in f :
                if not d + '.png' in already_done_lp :
                    already_done_lp.append(d + '.png')
            
            
    overwrite_bool = True
    if len(already_done_lp) > 0:        
        overwrite_bool = overwrite(fen)
    
    if not overwrite_bool :
        images_lp = filter_list(images_lp, already_done_lp)
 
    time_start = time.time() 
    
    for image in images_lp :
        compute_lp(os.path.join(images_path, image))
    
    time_end = time.time()
    popup('Computation done !', time_end - time_start)
    
    


def plot_correlation_img() :
    name = filedialog.askdirectory(title='Choose an image', initialdir=imgchar_path)
    basename = os.path.basename(name)
    
    image = cv2.imread(os.path.join(images_path, basename + '.png'), cv2.IMREAD_UNCHANGED)
    img = cv2.inRange(image, 0, 0)
    phi = 1-(np.count_nonzero(img == 0) / (img.shape[0]*img.shape[1]))
        
    try :
        lp = np.load(os.path.join(name, basename + '_lp.npy'))
    except FileNotFoundError :
        popup('Lineal path not computed for image : ' + basename)

    
    fig, axes = plt.subplots(figsize=(18, 6), nrows=1, ncols=2)

    ax = axes[0]
    im = ax.imshow(image, clim=(0, 1), cmap=mpl.colors.ListedColormap(cm.gray([0, 255])))
    ax.xaxis.set_ticks([0, +50, +100])
    ax.yaxis.set_ticks([0, +50, +100])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$\mathcal{I}$")
    
    ax = axes[1]
    im = ax.imshow(lp, clim=(0, phi), cmap="jet")
    ax.xaxis.set_ticks([0, +50, +100])
    ax.yaxis.set_ticks([0, +50, +100])
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    ax.set_title(r"$Lineal Path$")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, phi])
    cbar.set_ticklabels(["0", r"$\varphi$"])
    
    plt.show()
    
    

    
def quit_fen(fen) :
    plt.close('all')
    fen.quit()
    fen.destroy()
    sys.exit()


def overwrite(fen, label='') :
    p = tk.Toplevel(fen)
    
    
    p.wm_title('Overwrite')
    var = tk.BooleanVar(p)
    var.set(False)
    
    def btn(boolean) :
        var.set(boolean)
        label2.config(text='Current value : {}'.format(var.get()))
    
    if label != '' :
        text = label
    else :
        text = 'Would you want to overwrite existing results ?'
    
    label1 = tk.Label(p, text=text)
    label1.grid(row=0, column=0, pady=2, columnspan=2)
    
    label2 = tk.Label(p, text='Current value : {}'.format(var.get()))
    label2.grid(row=1, column=0, pady=2, columnspan=2)
    
    yes_btn = tk.Button(p, text='Yes', command = lambda : btn(True))
    yes_btn.grid(row=2, column=0, pady=2, sticky = 'ew')
    
    no_btn = tk.Button(p, text='No', command = lambda : btn(False))
    no_btn.grid(row=2, column=1, pady=2, sticky = 'ew')
    
    ok_btn = tk.Button(p, text='Ok', command =p.destroy)
    ok_btn.grid(row=3, column=0, sticky='ew', columnspan=2)
    
    
    # Does not work
    fen.wait_window(p)
    
    return var.get()


def askPercentage(fen) :
    
    p = tk.Toplevel(fen)
    
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
    
    
    
# To choose working directory
def choose_wd() :
    global working_dir
    working_dir = filedialog.askdirectory(initialdir=working_dir)
    

def main () :
    fen = tk.Tk()
    
    width = 800
    height = 320
    
    screen_width = fen.winfo_screenwidth()
    screen_height = fen.winfo_screenheight()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    fen.geometry(f'{width}x{height}+{x}+{y}')
    
    initdir_btn = tk.Button(fen, text='Create all the necessary directories', command=init_dirs)
    initdir_btn.pack()
    
    choosedir_btn = tk.Button(fen, text='Choose working directory', command=choose_wd)
    choosedir_btn.pack()
    
    filter_tiny_btn = tk.Button(fen, text='Filter out tiny inclusions', command=filter_tiny)
    filter_tiny_btn.pack()
    
    imgchar_btn = tk.Button(fen, text='Compute images characteristics', command= lambda : compute_img_char(fen))
    imgchar_btn.pack()
    
    plotdistr_btn = tk.Button(fen, text='Plot the descriptors of a selected image as points', command=plot_distribution)
    plotdistr_btn.pack()
    
    hist_btn = tk.Button(fen, text='Plot the descriptors of a selected image as histograms', command=plot_hist)
    hist_btn.pack()
    
    angle_btn = tk.Button(fen, text='Plot the orientation of a selected image as polar histograms', command=plot_polar)
    angle_btn.pack()
    
    compute_correlation_btn = tk.Button(fen, text='Compute L', command= lambda : compute_correlation(fen))
    compute_correlation_btn.pack()
   
    plot_correlation_btn = tk.Button(fen, text='Plot L for a given image', command=plot_correlation_img)
    plot_correlation_btn.pack()
    
    help_btn = tk.Button(fen, text='Help', command=help)
    help_btn.pack()
    
    quit_btn = tk.Button(fen, text='Quit', command= lambda : quit_fen(fen))
    quit_btn.pack()
    
    fen.mainloop()
  

main()
