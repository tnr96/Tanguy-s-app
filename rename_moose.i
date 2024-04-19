[Mesh]
    [mesh]
        type = FileMeshGenerator
        file = convert_moose_in.e
    []
    [rename]
        type = RenameBoundaryGenerator
        input = mesh
        old_boundary = 'top right bottom left'
        new_boundary = '1 2 3 4'
    []
[]

