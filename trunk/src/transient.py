## @package src.transient
#  Contains functions to run transients.

from copy import deepcopy
import numpy as np
from math import sqrt

from nonlinearSolve import nonlinearSolve
from utilityFunctions import computeL2RelDiff
from transientSource import computeRadiationExtraneousSource
from takeRadiationStep import takeRadiationStep
from hydroSlopes import HydroSlopes
from musclHancock import hydroPredictor, hydroCorrector

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
   time_index = 0
   transient_incomplete = True # boolean flag signalling end of transient
   while transient_incomplete:

       # increment time index
       time_index += 1

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
          print("Time step %d: t = %f -> %f:" % (time_index,t-dt,t))

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
def runNonlinearTransient(mesh, problem_type,
   psi_left, psi_right, cross_sects, rad_IC, hydro_IC, time_stepper='BE',
   dt_option='constant', dt_constant=None, CFL=0.5, t_start=0.0, t_end=1.0,
   verbose=False):

   # check input arguments
   if dt_option == 'constant':
      assert dt_constant != None, "If time step size option is chosen to \
         be 'constant', then a time step size must be provided."

   # initialize time and solutions
   t = t_start
   cx_older = deepcopy(cross_sects)
   cx_old   = deepcopy(cross_sects)
   cx_new   = deepcopy(cross_sects)
   rad_older = rad_IC
   rad_old   = rad_IC
   hydro_older = hydro_IC
   hydro_old   = hydro_IC
   slopes_older = HydroSlopes(hydro_older)
   e_slopes_old   = np.zeros(mesh.n_elems)
   e_slopes_older = np.zeros(mesh.n_elems)
   
   # transient loop
   time_index = 0
   transient_incomplete = True # boolean flag signalling end of transient
   while transient_incomplete:

       # increment time index
       time_index += 1

       # get time step size
       if dt_option == 'constant':
          # constant time step size
          dt = dt_constant
       elif dt_option == 'CFL':
          # compute time step size according to CFL condition
          sound_speed = [sqrt(i.p * i.gamma / i.rho) + abs(i.u) for i in hydro_old]
          dt_vals = [CFL*(mesh.elements[i].dx)/sound_speed[i]
             for i in range(len(hydro_old))]
          dt = min(dt_vals)
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
          print("Time step %d: t = %f -> %f:" % (time_index,t-dt,t))
  
       # take time step
       if problem_type == 'rad_mat':

          # take time step without MUSCL-Hancock
          hydro_new, rad_new, cx_new, slopes_old, e_slopes_new =\
             takeTimeStepNoMUSCLHancock(
             mesh         = mesh,
             time_stepper = time_stepper,
             dt           = dt,
             psi_left     = psi_left,
             psi_right    = psi_right,
             cx_old       = cx_old,
             cx_older     = cx_older,
             hydro_old    = hydro_old,
             hydro_older  = hydro_older,
             rad_old      = rad_old,
             rad_older    = rad_older,
             slopes_older = slopes_older,
             e_slopes_old = e_slopes_old,
             e_slopes_older = e_slopes_older)

       else:

          # take time step with MUSCL-Hancock
          hydro_new, rad_new, cx_new, slopes_old, e_slopes_new =\
             takeTimeStepMUSCLHancock(
             mesh           = mesh,
             dt             = dt, 
             psi_left       = psi_left,
             psi_right      = psi_right,
             cx_old         = cx_old,
             cx_older       = cx_older,
             hydro_old      = hydro_old,
             hydro_older    = hydro_older,
             rad_old        = rad_old,
             rad_older      = rad_older,
             slopes_older   = slopes_older,
             e_slopes_old   = e_slopes_old,
             e_slopes_older = e_slopes_older,
             time_stepper_predictor='CN',
             time_stepper_corrector='BDF2')
            
       # print the difference between old and new solutions
       if verbose:
          internal_energy_diff = computeL2RelDiff(hydro_old, hydro_new,
             aux_func=lambda x: x.e)
          print "Difference with old solution: ", internal_energy_diff, "\n"

       # save older solutions
       cx_older  = deepcopy(cx_old)
       rad_older = deepcopy(rad_old)
       hydro_older = deepcopy(hydro_old)
       slopes_older = deepcopy(slopes_old)
       e_slopes_older = deepcopy(e_slopes_old)

       # save old solutions
       cx_old  = deepcopy(cx_new)
       rad_old = deepcopy(rad_new)
       hydro_old = deepcopy(hydro_new)
       e_slopes_old = deepcopy(e_slopes_new)

   # return final solutions
   return rad_new, hydro_new


## Takes time step without any MUSCL-Hancock.
#
#  This should only be called if the problem type is 'rad_mat'.
#
def takeTimeStepNoMUSCLHancock(mesh, time_stepper, dt, psi_left, psi_right,
   cx_old, cx_older, hydro_old, hydro_older, rad_old, rad_older,
   slopes_older, e_slopes_old, e_slopes_older, problem_type='rad_mat'):

       # assert that problem type is 'rad_mat'
       assert problem_type == 'rad_mat', "Problem type must be 'rad_mat'\
          for this function to be called"

       # compute slopes
       # NOTE: do we need to prevent limiters from being applied to slopes
       # in this case?
       slopes_old = HydroSlopes(hydro_old)

       # if there is no material motion, then the homogeneous hydro solution
       # should be equal to the old hydro solution
       hydro_star = deepcopy(hydro_old)

       # perform nonlinear solve
       hydro_new, rad_new, cx_new, e_slopes_new = nonlinearSolve(
          mesh         = mesh,
          time_stepper = time_stepper,
          problem_type = problem_type,
          dt           = dt,
          psi_left     = psi_left,
          psi_right    = psi_right,
          cx_old       = cx_old,
          cx_older     = cx_older,
          hydro_old    = hydro_old,
          hydro_older  = hydro_older,
          hydro_star   = hydro_star,
          rad_old      = rad_old,
          rad_older    = rad_older,
          slopes_old   = slopes_old,
          slopes_older = slopes_older,
          e_slopes_old = e_slopes_old,
          e_slopes_older = e_slopes_older)

       return hydro_new, rad_new, cx_new, slopes_old, e_slopes_new


## Takes time step with MUSCL-Hancock.
#
#  This should only be called if the problem type is 'rad_hydro'.
#
def takeTimeStepMUSCLHancock(mesh, dt, psi_left, psi_right,
   cx_old, cx_older, hydro_old, hydro_older, rad_old, rad_older,
   slopes_older, e_slopes_old, e_slopes_older, problem_type='rad_hydro',
   time_stepper_predictor='CN', time_stepper_corrector='BDF2'):

       # assert that BDF2 was not chosen for the predictor time-stepper
       assert time_stepper_predictor != 'BDF2', 'BDF2 cannot be used in\
          the predictor step.'

       # assert that problem type is 'rad_hydro'
       assert problem_type == 'rad_hydro', "Problem type must be 'rad_hydro'\
          for this function to be called"

       # compute slopes
       slopes_old = HydroSlopes(hydro_old)

       # perform predictor step of MUSCL-Hancock
       hydro_star = hydroPredictor(mesh, hydro_old, slopes_old, dt)

       # perform nonlinear solve
       hydro_half, rad_half, cx_half, e_slopes_half = nonlinearSolve(
          mesh         = mesh,
          time_stepper = time_stepper_predictor,
          problem_type = problem_type,
          dt           = 0.5*dt,
          psi_left     = psi_left,
          psi_right    = psi_right,
          cx_old       = cx_old,
          hydro_old    = hydro_old,
          hydro_star   = hydro_star,
          rad_old      = rad_old,
          slopes_old   = slopes_old,
          e_slopes_old = e_slopes_old)

       # perform corrector step of MUSCL-Hancock
       hydro_star = hydroCorrector(mesh, hydro_half, dt)

       # perform nonlinear solve
       hydro_new, rad_new, cx_new, e_slopes_new = nonlinearSolve(
          mesh         = mesh,
          time_stepper = time_stepper_corrector,
          problem_type = problem_type,
          dt           = dt,
          psi_left     = psi_left,
          psi_right    = psi_right,
          cx_old       = cx_old,
          cx_older     = cx_older,
          hydro_old    = hydro_old,
          hydro_older  = hydro_older,
          hydro_star   = hydro_star,
          rad_old      = rad_old,
          rad_older    = rad_older,
          slopes_old   = slopes_old,
          slopes_older = slopes_older,
          e_slopes_old = e_slopes_old,
          e_slopes_older = e_slopes_older)

       return hydro_new, rad_new, cx_new, slopes_old, e_slopes_new


