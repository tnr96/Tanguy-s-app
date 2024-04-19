[Mesh]
    [mesh_define]
        type = ImageMeshGenerator
        dim = 2
		file = /Users/tmr96/projects/my_files/images/ellipse_perturbed_38.png
    []
    [image]
        input = mesh_define
        type = ImageSubdomainGenerator
		file = /Users/tmr96/projects/my_files/images/ellipse_perturbed_38.png
        threshold = 255 # everything above 255 will be assigned to block 2, everything below to block 1
        upper_value = 2
        lower_value = 1
    []
[]































































































































