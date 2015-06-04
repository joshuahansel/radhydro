## @package src.nonlinearSolve
#  Provides functions for performing nonlinear solves
#
#  TODO: generalize to accept all problem types
#

from newtonStateHandler import NewtonStateHandler
from transientSource import TransientSource
from globalConstants import SPD_OF_LGT as c
from radiationSolveSS import radiationSolveSS
from utilityFunctions import computeL2RelDiff

## Performs nonlinear solve
#
#  @return new hydro and rad solutions
#
def nonlinearSolve(mesh, time_stepper, problem_type, dt, psi_left, psi_right,
   cx_old, hydro_old, hydro_guess, rad_old, tol=1.0e-9):

   if problem_type != 'rad_mat':
      raise NotImplementedError("Nonlinear solve function currently only solves\
         thermal radiative transfer problems")

   # construct newton state handler
   newton_handler = NewtonStateHandler(mesh,
                       time_stepper=time_stepper,
                       cx_new=cx_old,
                       hydro_guess=hydro_guess)

   # initialize hydro guess
   hydro_prev = hydro_guess

   # initialize convergence flag and iteration counter
   converged = False
   k = 0

   # perform nonlinear iterations:
   while not converged:

       # increment iteration counter
       k += 1

       # get the modified scattering cross sections
       cx_prev = newton_handler.getEffectiveOpacities(dt)

       # take radiation step
       transient_source = TransientSource(mesh, time_stepper, problem_type=problem_type,
               newton_handler=newton_handler)
           
       # evaluate transient source, including linearized planckian
       Q_tr = transient_source.evaluate(
           dt            = dt,
           bc_flux_left  = psi_left,
           bc_flux_right = psi_right,
           cx_older      = cx_old,
           cx_old        = cx_old,
           cx_new        = cx_prev,
           rad_old       = rad_old ,
           hydro_star    = hydro_old)

       # update radiation
       alpha = 1./(c*dt)
       beta = {"CN":0.5, "BDF2":2./3., "BE":1.}
       rad_new = radiationSolveSS(mesh, cx_prev, Q_tr,
          bc_psi_left = psi_left,
          bc_psi_right = psi_right,
          diag_add_term = alpha,
          implicit_scale = beta[time_stepper] )

       # update internal energy
       newton_handler.updateIntEnergy(rad_new.E, dt, hydro_star = hydro_old)
       print "Woop", k

       # check nonlinear convergence
       hydro_new = newton_handler.getNewHydroStates()
       rel_diff = computeL2RelDiff(hydro_new, hydro_prev, aux_func=lambda x: x.e)
       print("Iteration %d: Difference = %7.3e" % (k,rel_diff))
       if rel_diff < tol:
          print rel_diff
          print("Nonlinear iteration converged")
          break

       # store new to prev
       hydro_prev = hydro_new

   # return new hydro and radiation
   return hydro_new, rad_new

