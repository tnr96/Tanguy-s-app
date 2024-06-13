import os
import glob
import matplotlib.pyplot as plt
import sys
import tkinter as tk

# Returns the list of images in a folder
def images_folder(images_path) :
    return [os.path.basename(image) for image in glob.glob(images_path + '/*.png')] # Retrieves all the names of the images in the images folder

# Returns the list of meshes in a folder
def mesh_folder(meshes_path) :
    return [os.path.basename(mesh) for mesh in glob.glob(meshes_path + '/*.e')] 

# Returns the list of consistent tensors for a given trial
def consistent_tensors_folder(trial_path) :
    path = os.path.join(trial_path, 'consistent_tensors')
    return [os.path.basename(tensor).split('_')[0] for tensor in glob.glob(path + '/*.txt')]
    
    
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
    exists = os.path.exists(file)
    if not exists :
        with open(file, 'w') as f :
            f.write('')
    return not exists

# Creates a directory if it does not exist
def create_dir(path) :
    exists = os.path.exists(path)
    if not exists :
        os.makedirs(path)
    return not exists

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
                if not x in index :
                    index.append(x)
    
    index.sort()
    
    for i in reversed(index) :
        list1.pop(i)
      
    return list1

  
  
  
      
def what_coeff(fen) :
    p = tk.Toplevel(fen)
    
    
    p.wm_title('Choose coeff')
    var = tk.StringVar(p)
    var.set('')
    
    def btn(string) :
        var.set(string)
        label2.config(text='Current value : {}'.format(var.get()))
    
    label1 = tk.Label(p, text='For what coefficient do you want to see the results ?')
    label1.grid(row=0, column=0, pady=2, columnspan=3)
    
    label2 = tk.Label(p, text='Current value : {}'.format(var.get()))
    label2.grid(row=1, column=0, pady=2, columnspan=3)
    
    c1111_btn = tk.Button(p, text='1111', command = lambda : btn('1111'))
    c1111_btn.grid(row=2, column=0, pady=2, sticky = 'ew')
    
    c1122_btn = tk.Button(p, text='1122', command = lambda : btn('1122'))
    c1122_btn.grid(row=2, column=1, pady=2, sticky = 'ew')
    
    c1121_btn = tk.Button(p, text='1121', command = lambda : btn('1121'))
    c1121_btn.grid(row=2, column=2, pady=2, sticky = 'ew')
    
    c2211_btn = tk.Button(p, text='2211', command = lambda : btn('2211'))
    c2211_btn.grid(row=3, column=0, pady=2, sticky = 'ew')

    c2222_btn = tk.Button(p, text='2222', command = lambda : btn('2222'))
    c2222_btn.grid(row=3, column=1, pady=2, sticky = 'ew')
    
    c2221_btn = tk.Button(p, text='2221', command = lambda : btn('2221'))
    c2221_btn.grid(row=3, column=2, pady=2, sticky = 'ew')
    
    c2111_btn = tk.Button(p, text='2111', command = lambda : btn('2111'))
    c2111_btn.grid(row=4, column=0, pady=2, sticky = 'ew')

    c2122_btn = tk.Button(p, text='2122', command = lambda : btn('2122'))
    c2122_btn.grid(row=4, column=1, pady=2, sticky = 'ew')

    c2121_btn = tk.Button(p, text='2121', command = lambda : btn('2121'))
    c2121_btn.grid(row=4, column=2, pady=2, sticky = 'ew')
    
    ok_btn = tk.Button(p, text='Ok', command =p.destroy)
    ok_btn.grid(row=5, column=0, sticky='ew', columnspan=3)  
    
    fen.wait_window(p) 
    
    return var.get()

    
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
    
