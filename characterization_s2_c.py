import os
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')
from pymks import (
    PrimitiveTransformer,
    TwoPointCorrelation,
)

import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image
import dask.array as da

from utilities import *

# Here you should define the paths on your computer 
working_dir = '/Users/tmr96/Documents/Automatic'
moose_dir = '/Users/tmr96/projects/my_files'

# You don't need to change anything here
images_path = moose_dir + '/images' # Images folder
imgchar_path = working_dir + '/images_characteristics' # Where the characteristics of the images are stored
filtered_path = working_dir + '/filtered_images'
extracted_path = working_dir + '/extracted_inclusions'

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
    create_dir(extracted_path)


# Add a periodicity prompt
def compute_s2(full_name, periodic_bool) :
    image = cv2.imread(full_name, cv2.IMREAD_UNCHANGED)
    
    
    basename = os.path.basename(full_name)
    basename_rm = rm_ext(basename)
   
    
    imarray = np.expand_dims(np.array(image), axis=0)
    data = PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0).transform(imarray)
    
    auto_correlation = TwoPointCorrelation(
        periodic_boundary = periodic_bool,
        cutoff = cutoff_input, # cutoff length should depend on the dimensions of my image
        correlations=[(0,0)]
    ).transform(data)   
    
    
    np.save(os.path.join(imgchar_path, basename_rm, basename_rm + '_s2.npy'), auto_correlation[0,:,:,0])
   
   
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
        
               

def compute_cluster(basename, periodic_bool) :
    basename = rm_ext(basename)
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
        periodic_boundary = periodic_bool,
        cutoff = 160, # I can tune the cutoff so that it is just sufficient for the computed two-point cluster function to drop to 0 at large distances
        correlations=[(0, 0)]  
    ).transform(data)
    
    summed_array = da.sum(data_corr, axis = 0)
    
    np.save(os.path.join(imgchar_path, basename, basename + '_cluster.npy'), summed_array)
    

def compute_correlation(fen) :
    
    images = images_folder(images_path)
    if '.DS_Store' in images :
        images.remove('.DS_Store')
    
    images_s2 = images.copy()
    images_c = images.copy()
    
    already_done_s2 = []
    already_done_c =[]
    
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
            
            
    overwrite_bool = True
    if len(already_done_s2) > 0 or len(already_done_c) > 0 :        
        overwrite_bool = overwrite(fen)
    
    if not overwrite_bool :
        images_s2 = filter_list(images_s2, already_done_s2)   
        images_c = filter_list(images_c, already_done_c)
    
    periodic_bool = overwrite(fen, 'Is your structure periodic ?')
    
    time_start = time.time() 
    for image in images_s2 :
        compute_s2(os.path.join(images_path, image), periodic_bool)
    for image in images_c :
        compute_cluster(image, periodic_bool)
    time_end = time.time()
    popup('Computation done !', time_end - time_start)
    
    


def plot_correlation_img() :
    name = filedialog.askdirectory(title='Choose an image', initialdir=imgchar_path)
    basename = os.path.basename(name)
    
    image = cv2.imread(os.path.join(images_path, basename + '.png'), cv2.IMREAD_UNCHANGED)
    img = cv2.inRange(image, 0, 0)
    phi = 1-(np.count_nonzero(img == 0) / (img.shape[0]*img.shape[1]))
    
    
    try :
        S2 = np.load(os.path.join(name, basename + '_s2.npy'))
    except FileNotFoundError :
        popup('S2 not computed for image : ' + basename)
    try :
        cluster = np.load(os.path.join(name, basename + '_cluster.npy'))
    except FileNotFoundError :
        popup('Cluster not computed for image : ' + basename)
        

    
    fig, axes = plt.subplots(figsize=(18, 6), nrows=1, ncols=3)

    ax = axes[0]
    im = ax.imshow(image, clim=(0, 1), cmap=mpl.colors.ListedColormap(cm.gray([0, 255])))
    ax.xaxis.set_ticks([0, +50, +100])
    ax.yaxis.set_ticks([0, +50, +100])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$\mathcal{I}$")

    ax = axes[1]
    im = ax.imshow(S2, clim=(0, phi), cmap="jet")
    ax.xaxis.set_ticks([0, +50, +100])
    ax.yaxis.set_ticks([0, +50, +100])
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    ax.set_title(r"$S_2$")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, phi])
    cbar.set_ticklabels(["0", r"$\varphi$"])
    
    ax = axes[2]
    im = ax.imshow(cluster, clim=(0, phi), cmap="jet")
    ax.xaxis.set_ticks([0, +50, +100])
    ax.yaxis.set_ticks([0, +50, +100])
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    ax.set_title(r"$C_2$")
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
    
    extraction_btn = tk.Button(fen, text='Extract inclusions', command= lambda : extract_inclusion_all(fen))
    extraction_btn.pack()
     
    compute_correlation_btn = tk.Button(fen, text='Compute S2 and C', command= lambda : compute_correlation(fen))
    compute_correlation_btn.pack()
   
    plot_correlation_btn = tk.Button(fen, text='Plot S2 and C for a given image', command=plot_correlation_img)
    plot_correlation_btn.pack()
    
    help_btn = tk.Button(fen, text='Help', command=help)
    help_btn.pack()
    
    quit_btn = tk.Button(fen, text='Quit', command= lambda : quit_fen(fen))
    quit_btn.pack()
    
    fen.mainloop()
  

main()
