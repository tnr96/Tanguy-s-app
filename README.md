# Tanguy-s-app
Follow these steps to set up the application :
1. Download all files and place them in your working directory.
2. Place the six MOOSE input files (sigma_11.i, etc...) in one folder named "to_run" in the working directory.
3. Change the path in run_moose.sh and in the Python files to your working directory (although you can change the working directory in the app).
4. Move convert_moose.i and rename_moose.i in ~/projects/your-files/images. If you don't do this, for some reason, MOOSE won't be able to detect the images.

You're ready to go ! Don't forget to click the "Create all necessary directories" button when your first start the app.

N.B. : 
1. Don't use any "_" to name your images. They are used as a marker to process files and if you do so, the app will just crash.
2. When running the simulations on the meshes ("Run simulations" button), for some reason, the six BC might not be simulated properly on the first go. I would recommend to always run the simulations twice to have complete results. (The second time you should set the overwrite option to false or you will encounter the same problem again).

