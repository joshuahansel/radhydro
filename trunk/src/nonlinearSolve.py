## @package src.nonlinearSolve
#  Provides functions for performing nonlinear solves
#

from copy import deepcopy

from takeRadiationStep import takeRadiationStep
from utilityFunctions import computeL2RelDiff, computeEffectiveOpacities,\
   updateCrossSections
from hydroSource import updateVelocity, updateInternalEnergy, QEHandler

## Performs nonlinear solve
#
#  @return new hydro and rad solutions
#
def nonlinearSolve(mesh, time_stepper, problem_type, dt, psi_left, psi_right,
   cx_old, hydro_old, hydro_star, rad_old, slopes_old, e_slopes_old,
   rad_older=None, cx_older=None, hydro_older=None, slopes_older=None,
   e_slopes_older=None, tol=1.0e-9,
   add_ext_src=False, rho_src_func=None, u_src_func=None, E_src_func=None,
   psim_src_func=None, psip_src_func=None):

   # assert that that older arguments were passed if using BDF2
   if time_stepper == 'BDF2':
      assert(rad_older      != None)
      assert(hydro_older    != None)
      assert(cx_older       != None)
      assert(slopes_older   != None)
      assert(e_slopes_older != None)

   # assert source functions provided if extraneous sources indicated
   if add_ext_src:
      assert rho_src_func  is not None, 'Source functions must be provided' 
      assert u_src_func    is not None, 'Source functions must be provided'
      assert E_src_func    is not None, 'Source functions must be provided'
      assert psim_src_func is not None, 'Source functions must be provided'
      assert psip_src_func is not None, 'Source functions must be provided'

   # initialize iterates to the old quantities
   hydro_new  = deepcopy(hydro_old)
   hydro_prev = deepcopy(hydro_old)
   rad_prev   = deepcopy(rad_old)
   cx_prev    = deepcopy(cx_old)

   # initialize convergence flag and iteration counter
   converged = False
   k = 0

   # perform nonlinear iterations:
   while not converged:

       # increment iteration counter
       k += 1

       # update velocity
       if problem_type == 'rad_hydro':
          updateVelocity(mesh         = mesh,
                         time_stepper = time_stepper,
                         dt           = dt, 
                         cx_older     = cx_older,
                         cx_old       = cx_old,
                         cx_prev      = cx_prev,
                         rad_older    = rad_older,
                         rad_old      = rad_old,
                         rad_prev     = rad_prev,
                         hydro_older  = hydro_older,
                         hydro_old    = hydro_old,
                         hydro_star   = hydro_star,
                         hydro_prev   = hydro_prev,
                         hydro_new    = hydro_new)

       # compute QE
       src_handler = QEHandler(mesh, time_stepper)
       QE = src_handler.computeTerm(cx_prev     = cx_prev,
                                    cx_old      = cx_old,
                                    cx_older    = cx_older,
                                    rad_prev    = rad_prev,
                                    rad_old     = rad_old,
                                    rad_older   = rad_older,
                                    hydro_prev  = hydro_prev,
                                    hydro_old   = hydro_old,
                                    hydro_older = hydro_older,
                                    slopes_old  = slopes_old,
                                    slopes_older = slopes_older,
                                    e_slopes_old = e_slopes_old,
                                    e_slopes_older = e_slopes_older)

       # get the modified scattering cross sections
       cx_mod_prev = computeEffectiveOpacities(
          time_stepper = time_stepper,
          dt           = dt,
          cx_prev      = cx_prev,
          hydro_prev   = hydro_prev,
          slopes_old   = slopes_old,
          e_slopes_old = e_slopes_old)

       # evaluate transient source, including linearized planckian
       rad_new = takeRadiationStep(
           mesh          = mesh,
           time_stepper  = time_stepper,
           problem_type  = problem_type,
           dt            = dt,
           psi_left      = psi_left,
           psi_right     = psi_right,
           cx_new        = cx_mod_prev,
           cx_prev       = cx_prev,
           cx_old        = cx_old,
           cx_older      = cx_older,
           rad_prev      = rad_prev,
           rad_old       = rad_old,
           rad_older     = rad_older,
           hydro_prev    = hydro_prev,
           hydro_star    = hydro_star,
           hydro_old     = hydro_old,
           hydro_older   = hydro_older,
           QE            = QE,
           add_ext_source = add_ext_src,
           slopes_old    = slopes_old,
           slopes_older  = slopes_older,
           e_slopes_old  = e_slopes_old,
           e_slopes_older = e_slopes_older)

       # update internal energy
       e_slopes_new = updateInternalEnergy(
          time_stepper = time_stepper,
          dt           = dt,
          QE           = QE,
          cx_prev      = cx_prev,
          rad_new      = rad_new,
          hydro_new    = hydro_new,
          hydro_prev   = hydro_prev,
          hydro_star   = hydro_star,
          slopes_old   = slopes_old,
          e_slopes_old = e_slopes_old)

       # check nonlinear convergence
       # TODO: compute diff of rad solution as well to add to convergence criteria
       rel_diff = computeL2RelDiff(hydro_new, hydro_prev, aux_func=lambda x: x.e)
       print("      Iteration %d: Difference = %7.3e" % (k,rel_diff))
       if rel_diff < tol:
          print("      Nonlinear iteration converged")
          break

       # reset previous iteration quantities if needed
       hydro_prev = deepcopy(hydro_new)
       rad_prev   = deepcopy(rad_new)
       updateCrossSections(cx_prev,hydro_new,slopes_old,e_slopes_old)      

   # return new hydro and radiation
   return hydro_new, rad_new, cx_prev, e_slopes_new

