## @package src.nonlinearSolve
#  Provides functions for performing nonlinear solves
#
#  TODO: generalize to accept all problem types
#

from copy import deepcopy
from newtonStateHandler import NewtonStateHandler
from radiationTimeStepper import takeRadiationStep
from utilityFunctions import computeL2RelDiff
from crossXInterface import updateCrossSections

## Performs nonlinear solve
#
#  @param[in] hydro_guess  guess
#
#  @return new hydro and rad solutions
#
def nonlinearSolve(mesh, time_stepper, problem_type, dt, psi_left, psi_right,
   cx_old, hydro_old, hydro_guess, rad_old, rad_guess, rad_older=None,
   cx_older=None, hydro_older=None, tol=1.0e-9):

   # assert that that older arguments were passed if using BDF2
   if time_stepper == 'BDF2':
      assert(rad_older   is not None)
      assert(hydro_older is not None)
      assert(cx_older    is not None)

   if problem_type != 'rad_mat':
      raise NotImplementedError("Nonlinear solve function currently only solves\
         thermal radiative transfer problems")

   # construct newton state handler
   newton_handler = NewtonStateHandler(mesh,
                       time_stepper=time_stepper,
                       cx_new=cx_old,
                       hydro_guess=hydro_guess)

   # initialize iterates to the provided guesses
   rad_prev   = deepcopy(rad_guess)
   cx_prev    = deepcopy(cx_old)

   # initialize convergence flag and iteration counter
   converged = False
   k = 0

   # perform nonlinear iterations:
   while not converged:

       # increment iteration counter
       k += 1

       # newton_handler returns a deepcopy, not a name copy
       hydro_prev = newton_handler.getNewHydroStates()

       # compute QE
       newton_handler.updateMatCouplingQE(cx_prev=cx_prev,
                                          cx_old=cx_old,
                                          cx_older=cx_older,
                                          hydro_old=hydro_old,
                                          hydro_older=hydro_older,
                                          rad_prev=rad_prev,
                                          rad_old=rad_old,
                                          rad_older=rad_older)

       # get the modified scattering cross sections
       cx_mod_prev = newton_handler.getEffectiveOpacities(dt)

       planckian_new = newton_handler.evalPlanckianImplicit(dt=dt, hydro_star=hydro_old)

       # evaluate transient source, including linearized planckian
       rad_new = takeRadiationStep(
           mesh          = mesh,
           time_stepper  = time_stepper,
           problem_type  = problem_type,
           planckian_new = planckian_new,
           dt            = dt,
           psi_left      = psi_left,
           psi_right     = psi_right,
           cx_new        = cx_mod_prev,
           cx_old        = cx_old,
           cx_older      = cx_older,
           rad_old       = rad_old,
           rad_older     = rad_older,
           hydro_star    = hydro_old,
           hydro_old     = hydro_old,
           hydro_older   = hydro_older)

       # update internal energy
       newton_handler.updateIntEnergy(
          dt=dt,
          rad_new=rad_new,
          hydro_star=hydro_old)

       # check nonlinear convergence
       # TODO: compute diff of rad solution as well to add to convergence criteria
       hydro_new = newton_handler.getNewHydroStates()
       rel_diff = computeL2RelDiff(hydro_new, hydro_prev, aux_func=lambda x: x.e)
       print("  Iteration %d: Difference = %7.3e" % (k,rel_diff))
       if rel_diff < tol:
          print("  Nonlinear iteration converged")
          break

       # reset previous iteration quantities if needed
       hydro_prev = deepcopy(hydro_new)
       rad_prev   = deepcopy(rad_new)

   # return new hydro and radiation
   return hydro_new, rad_new

