## @package src.transient
#  Contains functions to run transients.

from copy import deepcopy

from nonlinearSolve import nonlinearSolve
from utilityFunctions import computeL2RelDiff

## Runs transient for a nonlinear radiation-material problem
#
def runRadiationMaterialTransient(mesh, time_stepper, problem_type,
   psi_left, psi_right, cross_sects, rad_IC, hydro_IC=None,
   dt_option='constant', dt_constant=None, t_start=0.0, t_end=1.0, verbose=False):

   # check input arguments
   if dt_option == 'constant':
      assert dt_constant is not None, "If time step size option is chosen to \
         be 'constant', then a time step size must be provided."
   if problem_type is not 'rad_only':
      assert hydro_IC is not None, 'Hydro IC must be provided for the chosen \
         problem type'

   # initialize time and solutions
   t = t_start
   rad_old   = rad_IC
   hydro_old = hydro_IC
   
   # transient loop
   transient_incomplete = True # boolean flag signalling end of transient
   while transient_incomplete:

       # get time step size
       if dt_option == 'constant':
          dt = dt_constant
       else:
          raise NotImplementedError('Invalid time step size option')
  
       # adjust time step size if it would overshoot the end of the transient
       if t + dt >= t_end:
          dt = t_end - t
          t = t_end
          transient_incomplete = False # signal end of transient
       else:
          t += dt

       # print each time step
       if verbose:
          print("t = %0.3f -> %0.3f:" % (t-dt,t))
  
       # this is where MUSCL Hancock would go
       hydro_star = deepcopy(hydro_old)

       # perform nonlinear solve
       hydro_new, rad_new = nonlinearSolve(
          mesh         = mesh,
          time_stepper = time_stepper,
          problem_type = 'rad_mat',
          dt           = dt,
          psi_left     = psi_left,
          psi_right    = psi_right,
          cx_old       = cross_sects,
          hydro_old    = hydro_old,
          hydro_star   = hydro_star,
          rad_old      = rad_old)
               
       # print the difference between old and new solutions
       internal_energy_diff = computeL2RelDiff(hydro_old, hydro_new,
          aux_func=lambda x: x.e)
       print "Difference in time steps: ", internal_energy_diff, "\n"

       # save old solutions
       hydro_old = deepcopy(hydro_new)
       rad_old   = deepcopy(rad_new)

   # return final solutions
   return rad_new, hydro_new

