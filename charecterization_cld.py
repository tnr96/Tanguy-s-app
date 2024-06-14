import cv2
import porespy as ps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from math import radians, cos, sin, radians
import os
import numpy as np
import time
from tkinter import filedialog

from utilities import *

working_dir = '/Users/tmr96/Documents/Automatic'
moose_dir = '/Users/tmr96/projects/my_files'

images_path = moose_dir + '/images' # Images folder
imgchar_path = working_dir + '/images_characteristics' # Where the characteristics of the images are stored

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

        plotRotMask = mask.copy()
        # ROTATE ALL THE INCLUSIONS BY THE SAME ANGLE
        for i in range(len(contours_draw)):  
            M = cv2.moments(contours_draw[i])
            if M['m00'] != 0:
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
        
    
    images_cld_angular = images.copy()
    images_cld_dir = images.copy()
    
    
    already_done_cld_angular = []
    already_done_cld_dir = []
    
    directories = os.listdir(imgchar_path)
    
    if '.DS_Store' in directories :
        directories.remove('.DS_Store')
    
    
    for d in directories :
        for f in os.listdir(os.path.join(imgchar_path, d)) :
            if 'cld_angular' in f :
                if not d + '.png' in already_done_cld_angular :
                    already_done_cld_angular.append(d + '.png')
            elif 'cld_dir' in f :
                if not d + '.png' in already_done_cld_dir :
                    already_done_cld_dir.append(d +'.png')

    overwrite_bool = True
    if len(already_done_cld_angular) > 0 or len(already_done_cld_dir) > 0 :        
        overwrite_bool = overwrite(fen)
    
    if not overwrite_bool :
        images_cld_angular = filter_list(images_cld_angular, already_done_cld_angular)
        images_cld_dir = filter_list(images_cld_dir, already_done_cld_dir)
        
    
    time_start = time.time() 
    for image in images_cld_angular :
        compute_cld_angular(os.path.join(images_path, image))
    for image in images_cld_dir :
        compute_cld_dir(os.path.join(images_path, image))
        
        
    time_end = time.time()
    popup('Computation done !', time_end - time_start)
   
   
   
def plot_correlation_img() :
    name = filedialog.askdirectory(title='Choose an image', initialdir=imgchar_path)
    basename = os.path.basename(name)
    
    image = cv2.imread(os.path.join(images_path, basename + '.png'), cv2.IMREAD_UNCHANGED)
    img = cv2.inRange(image, 0, 0)
    phi = np.mean(img)    
    
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
        
        

def main () :
    fen = tk.Tk()
    
    width = 800
    height = 320
    
    screen_width = fen.winfo_screenwidth()
    screen_height = fen.winfo_screenheight()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    fen.geometry(f'{width}x{height}+{x}+{y}')
    
    compute_correlation_btn = tk.Button(fen, text='Compute CLD', command= lambda : compute_correlation(fen))
    compute_correlation_btn.pack()
   
    plot_correlation_btn = tk.Button(fen, text='Plot CLD for a given image', command=plot_correlation_img)
    plot_correlation_btn.pack()
    
    help_btn = tk.Button(fen, text='Help', command=help)
    help_btn.pack()
    
    quit_btn = tk.Button(fen, text='Quit', command= lambda : quit_fen(fen))
    quit_btn.pack()
    
    fen.mainloop()
  

main()
