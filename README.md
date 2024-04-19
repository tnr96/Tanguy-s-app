# Tanguy-s-app
In the application, there is a button called "Display working directories" telling how to put the different files in your working directory. 
You don't even have to create the different folders manually, you can just click the "Create all necessary directories button". However, this button won't overwrite the directories if they already exist.
The six MOOSE scripts running the simulations should be in a folder named "to_run" in the working directory. For some reason I can't import a folder on this Github repository.
The two other MOOSE scripts should be in your images folder, in your projects folder. Otherwise, for some reason, MOOSE can't detect the images while going through the folder.
There is no particular naming convetion for the input images, as all is managed by the app. To test different samples, I just extract the modified folders and store them elsewhere along with a file "Characteristics.rft" that briefly describes the characteristics of the simulation.
I guess the naming of your trials is up to you.
If you encouter any crash during the use of the app, please check the MOOSE input files. They are edited a lot by Python and sometimes the editing is not made properly. When to compute the consistent tensors, if you get a popup telling you that there are not csv files, just click the button "Run simulations" again without overwriting the results.

I also uploaded a script to interpolate a Gaussian distribution to generate images with a certain mean and standard deviation for the orientation. 
The idea to interpolate the distribution is to loop through all inclusions and assign them to a random category if it's not already full.
