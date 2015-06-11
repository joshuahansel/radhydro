## @package src.transient
#  Contains functions to run transients.

from copy import deepcopy

from nonlinearSolve import nonlinearSolve
from utilityFunctions import computeL2RelDiff
from transientSource import computeRadiationExtraneousSource
from takeRadiationStep import takeRadiationStep


## Runs transient for a radiation-only problem
#
def runLinearTransient(mesh, time_stepper,
   psi_left, psi_right, cross_sects, rad_IC, psim_src, psip_src,
   dt_option='constant', dt_constant=None, t_start=0.0, t_end=1.0, verbose=False):

   # check input arguments
   if dt_option == 'constant':
      assert dt_constant is not None, "If time step size option is chosen to \
         be 'constant', then a time step size must be provided."

   # initialize time and solutions
   t = t_start
   Q_older = computeRadiationExtraneousSource(psim_src, psip_src, mesh, t_start)
   Q_old   = deepcopy(Q_older)
   rad_older = rad_IC
   rad_old   = rad_IC
   
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

       # compute new extraneous source
       Q_new = computeRadiationExtraneousSource(psim_src, psip_src, mesh, t)
  
       # take radiation step
       #
       # NOTE: In this case, cross sections are assumed to be constant
       #       with respect to time because cross sections are generally
       #       functions of material properties, and there is no coupling
       #       to material physics in a radiation-only problem.
       #
       rad_new = takeRadiationStep(
          mesh          = mesh,
          time_stepper  = time_stepper,
          problem_type  = 'rad_only',
          dt            = dt,
          psi_left      = psi_left,
          psi_right     = psi_right,
          cx_older      = cross_sects,
          cx_old        = cross_sects,
          cx_new        = cross_sects,
          rad_older     = rad_older,
          rad_old       = rad_old,
          Q_older       = Q_older,
          Q_old         = Q_old,
          Q_new         = Q_new)

       # save older solutions
       Q_older   = deepcopy(Q_old)
       rad_older = deepcopy(rad_old)

       # save old solutions
       Q_old   = deepcopy(Q_new)
       rad_old = deepcopy(rad_new)

   # return final solution
   return rad_new

## Runs transient for a nonlinear radiation-material problem
#
def runNonlinearTransient(mesh, time_stepper, problem_type,
   psi_left, psi_right, cross_sects, rad_IC, hydro_IC,
   dt_option='constant', dt_constant=None, t_start=0.0, t_end=1.0, verbose=False):

   # check input arguments
   if dt_option == 'constant':
      assert dt_constant is not None, "If time step size option is chosen to \
         be 'constant', then a time step size must be provided."

   # initialize time and solutions
   t = t_start
   cx_older = cross_sects
   cx_old   = cross_sects
   cx_new   = cross_sects
   rad_older = rad_IC
   rad_old   = rad_IC
   hydro_older = hydro_IC
   hydro_old   = hydro_IC
   
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
       if verbose:
          internal_energy_diff = computeL2RelDiff(hydro_old, hydro_new,
             aux_func=lambda x: x.e)
          print "Difference with old solution: ", internal_energy_diff, "\n"

       # save older solutions
       cx_older  = deepcopy(cx_old)
       rad_older = deepcopy(rad_old)
       hydro_older = deepcopy(hydro_old)

       # save old solutions
       cx_old  = deepcopy(cx_new) # TODO: have cx_new returned by nonlinearSolve
       rad_old = deepcopy(rad_new)
       hydro_old = deepcopy(hydro_new)

   # return final solutions
   if problem_type == 'rad_only':
      return rad_new
   else:
      return rad_new, hydro_new

