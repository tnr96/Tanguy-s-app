# This script: Apply only one non-zero macrostress sig_11, by using Neumann BCs
[GlobalParams]
    displacements = 'disp_x disp_y'
[]
  
[Mesh]
    [base]
      type = FileMeshGenerator
file = '/Users/tmr96/Documents/Automatic/coreform_output/erosion4.e'
    []
[]

# I am also extracting strain_zz and stress_zz to check what type of FE problem I am solving (plane strain or plane stress)
[Physics/SolidMechanics/QuasiStatic]
    [./all]
      add_variables = true # The variables have to be defined in their block beforehand 
      strain = SMALL
      generate_output = 'stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy stress_zx stress_zy stress_zz strain_zx strain_zy strain_zz' # these are auxiliary variables to be used for output
    [../]
[]
  
[Materials]
    [./Elasticity_inclusion]
      type = ComputeIsotropicElasticityTensor
      #block = 'inclusion' # inclusion representing the fiber in the paper
      block = 1 # this is the inclusion in my current mesh
      youngs_modulus = 70000 # MPa
      poissons_ratio  = 0.22
    [../]
    [./Elasticity_matrix] 
      type = ComputeIsotropicElasticityTensor
      #block = 'matrix' # inclusion representing the fiber in the paper
      block = 2
      youngs_modulus = 7000 # MPa
      poissons_ratio  = 0.3
    [../]
    [./stress]
      type = ComputeLinearElasticStress # computes linear elastic stress for a TOTAL and small-strain formulation   
    [../]
[]
  
   
[BCs]
    [macro_sig11_right]
        type = NeumannBC
        variable = disp_x
        value = 1
        boundary = 'right'
    []
    [macro_sig11_left]
      type = NeumannBC
      variable = disp_x
      value = -1
      boundary = 'left'
    []
[]
 

# Then, solver and postprocessors, check average stress and strain theorems hold
[Preconditioning]
    [smp]
      type = SMP
      full = true
    []
[]
  
# USING SOLVER PARAMETERS SAME AS IN https://github.com/idaholab/moose/discussions/25408,
# which helped avoid the divergence issue I had with default steady state solve parameters
[Executioner] 
  type = Steady

  solve_type = 'newton'
  line_search = none

  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'

  l_max_its = 2
  l_tol = 1e-14
  nl_max_its = 30
  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-10
[]

     
[Postprocessors]
    [eps_xy]
      type = ElementAverageValue
      variable = strain_xy
      execute_on = 'timestep_end'
    []
    [eps_xx]
      type = ElementAverageValue
      variable = strain_xx
      execute_on = 'timestep_end'
    []
    [eps_yy]
      type = ElementAverageValue
      variable = strain_yy
      execute_on = 'timestep_end'
    []
    [sig_xy]
      type = ElementAverageValue
      variable = stress_xy
      execute_on = 'timestep_end'
    []
    [sig_xx]
        type = ElementAverageValue
        variable = stress_xx
        execute_on = 'timestep_end'
    []
    [sig_yy]
        type = ElementAverageValue
        variable = stress_yy
        execute_on = 'timestep_end'
    []
    [eps_zx]
      type = ElementAverageValue
      variable = strain_zx
      execute_on = 'timestep_end'
    []
    [eps_zy]
      type = ElementAverageValue
      variable = strain_zy
      execute_on = 'timestep_end'
    []
    [eps_zz]
      type = ElementAverageValue
      variable = strain_zz
      execute_on = 'timestep_end'
    []
   [stress_zx]
     type = ElementAverageValue
     variable = stress_zx
     execute_on = 'timestep_end'
   []
   [stress_zy]
     type = ElementAverageValue
     variable = stress_zy
     execute_on = 'timestep_end'
   []
  [stress_zz]
    type = ElementAverageValue
    variable = stress_zz
    execute_on = 'timestep_end'
  []
[]
  
[Outputs]
    [out]
      type = Exodus
      execute_on = 'TIMESTEP_END'
      enable = false
file_base = /Users/tmr96/Documents/Automatic/simulation_results/erosion4/erosion4_sigma_11
    []
    [csv]
      type = CSV
      execute_on = 'TIMESTEP_END'
      #append_date = True
file_base = /Users/tmr96/Documents/Automatic/simulation_results/erosion4/erosion4_sigma_11
    []
[]
  

  
  
  
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  