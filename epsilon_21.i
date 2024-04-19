# This script: Apply macro-shear strain eps_12
[GlobalParams]
    displacements = 'disp_x disp_y'
[]
  
[Mesh]
    [base]
      type = FileMeshGenerator
file = '/Users/tmr96/Documents/Automatic/coreform_output/erosion4.e'
    []
[]
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
      block = 1 # this is the current inclusion block in my mesh
      youngs_modulus = 70000 # MPa
      poissons_ratio  = 0.22
    [../]
    [./Elasticity_matrix]
      type = ComputeIsotropicElasticityTensor
      block = 2
      youngs_modulus = 7000 # MPa
      poissons_ratio  = 0.3
    [../]
    [./stress]
      type = ComputeLinearElasticStress # computes linear elastic stress for a TOTAL and small-strain formulation   
    [../]
[]
[Functions]
    [linX] 
      type = 'ParsedFunction'
      symbol_names = 'epsilon0'
      symbol_values = '0.01'  #ensure small strain conditions are satisfied
      expression = epsilon0*x  # x must be the x coordinate when this function is being read by the BCs block
    []
    [linY] 
      type = 'ParsedFunction'
      symbol_names = 'epsilon0'
      symbol_values = '0.01'
      expression = epsilon0*y
    []
[] 
[BCs]
    [disp_x_lr]
      type = FunctionDirichletBC
      variable = disp_x
      function = linY
      boundary = 'left right'
    []
    [disp_x_top]
      type = DirichletBC
      variable = disp_x
      boundary = 'top'
      value = 0.01 
    []
    [disp_x_bot]
      type = DirichletBC
      variable = disp_x
      boundary = 'bottom'
      value = 0
    []
    [disp_y_tb]
      type = FunctionDirichletBC
      variable = disp_y
      function = linX
      boundary = 'top bottom'
    []
    [disp_y_l]
      type = DirichletBC
      variable = disp_y
      boundary = 'left'
      value = 0 # half of the target shear strain
    []
    [disp_y_r]
      type = DirichletBC
      variable = disp_y
      boundary = 'right' # half of the target shear strain
      value = 0.01
    []
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
file_base = /Users/tmr96/Documents/Automatic/simulation_results/erosion4/erosion4_epsilon_21
enable = false
  []
  [csv]
    type = CSV
    execute_on = 'TIMESTEP_END'
file_base = /Users/tmr96/Documents/Automatic/simulation_results/erosion4/erosion4_epsilon_21
  []
[]   

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]
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



















































































