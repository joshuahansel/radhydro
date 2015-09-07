## @package src.transient
#  Contains functions to run transients.

from copy import deepcopy
import numpy as np
from math import sqrt

from nonlinearSolve import nonlinearSolve
from utilityFunctions import computeL2RelDiff, computeAnalyticHydroSolution, getIndex
from transientSource import computeRadiationExtraneousSource
from hydroSource import computeMomentumExtraneousSource,\
   computeEnergyExtraneousSource
from takeRadiationStep import takeRadiationStep
from hydroSlopes import HydroSlopes
from musclHancock import hydroPredictor, hydroCorrector
from balanceChecker import BalanceChecker
from plotUtilities import plotHydroSolutions, plotIntErgs
from radUtilities import mu
import globalConstants as GC

## Runs transient for a radiation-only problem.
#
#  @param[in] psim_src  extraneous source function for \f$\Psi^-\f$
#  @param[in] psip_src  extraneous source function for \f$\Psi^+\f$
#
def runLinearTransient(mesh, time_stepper,
   rad_BC, cross_sects, rad_IC, psim_src, psip_src,
   dt_option='constant', dt_constant=None, t_start=0.0, t_end=1.0, verbosity=2):

   # check input arguments
   if dt_option == 'constant':
      assert dt_constant is not None, "If time step size option is chosen to \
         be 'constant', then a time step size must be provided."

   # initialize time and solutions
   t = t_start
   rad_old = rad_IC
   Qpsi_old = computeRadiationExtraneousSource(psim_src, psip_src, mesh, t_start)
   rad_older  = None
   Qpsi_older = None
   
   # transient loop
   time_index = 0
   transient_incomplete = True # boolean flag signalling end of transient
   while transient_incomplete:

       # increment time index
       time_index += 1

       # if first step, then can't use BDF2
       if time_index == 1 and time_stepper == 'BDF2':
          time_stepper_this_step = 'BE'
       else:
          time_stepper_this_step = time_stepper

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
       if verbosity > 0:
          print("Time step %d: t = %f -> %f:" % (time_index,t-dt,t))

       # compute new extraneous source
       Qpsi_new = computeRadiationExtraneousSource(psim_src, psip_src, mesh, t)
  
       # take radiation step
       #
       # NOTE: In this case, cross sections are assumed to be constant
       #       with respect to time because cross sections are generally
       #       functions of material properties, and there is no coupling
       #       to material physics in a radiation-only problem.
       #
       rad_new = takeRadiationStep(
          mesh          = mesh,
          time_stepper  = time_stepper_this_step,
          problem_type  = 'rad_only',
          dt            = dt,
          rad_BC        = rad_BC,
          cx_older      = cross_sects,
          cx_old        = cross_sects,
          cx_new        = cross_sects,
          rad_older     = rad_older,
          rad_old       = rad_old,
          Qpsi_older    = Qpsi_older,
          Qpsi_old      = Qpsi_old,
          Qpsi_new      = Qpsi_new)

       # save older solutions
       Qpsi_older = deepcopy(Qpsi_old)
       rad_older  = deepcopy(rad_old)

       # save old solutions
       Qpsi_old   = deepcopy(Qpsi_new)
       rad_old    = deepcopy(rad_new)

   # return final solution
   return rad_new


## Runs transient for a nonlinear radiation-material problem.
#
#  @param[in] add_ext_src  flag to signal that extraneous sources are to be added
#  @param[in] psim_src  extraneous source function for \f$\Psi^-\f$
#  @param[in] psip_src  extraneous source function for \f$\Psi^+\f$
#  @param[in] mom_src   extraneous source function for the conservation
#                       of momentum equation
#  @param[in] E_src     extraneous source function for the conservation
#                       of total energy equation
#
def runNonlinearTransient(mesh, problem_type,
   rad_BC, cross_sects, rad_IC, hydro_IC, hydro_BC,
   psim_src=None, psip_src=None, mom_src=None, E_src=None, rho_src=None,
   time_stepper='BE', dt_option='constant', dt_constant=None, CFL=0.5,
   slope_limiter="vanleer", t_start=0.0, t_end=1.0, use_2_cycles=False,
   rho_f=None,u_f=None,E_f=None,gamma_value=None,cv_value=None,
   verbosity=2, check_balance=False):

   # check input arguments
   if dt_option == 'constant':
      assert dt_constant != None, "If time step size option is chosen to \
         be 'constant', then a time step size must be provided."

   # initialize old quantities
   t_old = t_start
   cx_old = deepcopy(cross_sects)
   rad_old = deepcopy(rad_IC)
   hydro_old = deepcopy(hydro_IC)
   Qpsi_old, Qmom_old, Qerg_old, Qrho_old = computeExtraneousSources(
      psim_src, psip_src, mom_src, E_src, mesh, t_start, rho_src=rho_src)

   #Just guess e_rad old is hydro initial conditionsstuff
   e_rad_old = np.array([(i.e, i.e) for i in hydro_old])
   
   # set older quantities to nothing; these shouldn't exist yet
   cx_older       = None
   rad_older      = None
   hydro_older    = None
   slopes_older   = None
   e_rad_older    = None
   Qpsi_older     = None
   Qrho_older     = None
   Qmom_older     = None
   Qerg_older     = None
   
   # transient loop
   time_index = 0
   transient_incomplete = True # boolean flag signalling end of transient
   while transient_incomplete:

       # increment time index
       time_index += 1

       # if first step, then can't use BDF2
       if time_index == 1 and time_stepper == 'BDF2':
          time_stepper_this_step = 'BE'
       else:
          time_stepper_this_step = time_stepper

       # get time step size
       if dt_option == 'constant':
          # constant time step size
          dt = dt_constant
       elif dt_option == 'CFL':
          # compute time step size according to CFL condition
          sound_speed = [sqrt(i.p * i.gamma / i.rho) + abs(i.u) for i in hydro_old]
          dt_vals = [CFL*(mesh.elements[i].dx)/sound_speed[i]
             for i in xrange(len(hydro_old))]
          dt = min(dt_vals)

          # if using 2 cycles, then twice the time step size may be taken
          if use_2_cycles:
             dt *= 2.0
       else:
          raise NotImplementedError('Invalid time step size option')
  
       # adjust time step size if it would overshoot the end of the transient
       if t_old + dt >= t_end:
          dt = t_end - t_old
          t_new = t_end
          transient_incomplete = False # signal end of transient
       else:
          t_new = t_old + dt

       # print each time step
       print("Time step %d: t = %f -> %f:" % (time_index,t_old,t_new))
  
       # take time step
       if problem_type == 'rad_mat':

          #Force balance checker vars to zeros
          hydro_F_left = {"rho":0.,"erg":0.,"mom":0.}
          hydro_F_right = {"rho":0.,"erg":0.,"mom":0.}

          if time_stepper == 'TRBDF2':

              # take a half time step with CN
              hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
              Qpsi_new, Qrho_new, Qmom_new, Qerg_new, src_totals_cycle1 =\
                 takeTimeStepRadiationMaterial(
                 mesh         = mesh,
                 time_stepper = 'CN',
                 dt           = 0.5*dt,
                 rad_BC       = rad_BC,
                 hydro_BC     = hydro_BC,
                 cx_old       = cx_old,
                 hydro_old    = hydro_old,
                 rad_old      = rad_old,
                 e_rad_old    = e_rad_old,
                 psim_src     = psim_src,
                 psip_src     = psip_src,
                 rho_src      = rho_src,
                 mom_src      = mom_src,
                 E_src        = E_src,
                 t_old        = t_old,
                 Qpsi_old     = Qpsi_old,
                 Qmom_old     = Qmom_old,
                 slope_limiter= slope_limiter,
                 Qerg_old     = Qerg_old)

              # take a half time step with BDF2
              hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
              Qpsi_new, Qrho_new, Qmom_new, Qerg_new, src_totals_cycle2 =\
                 takeTimeStepRadiationMaterial(
                 mesh         = mesh,
                 time_stepper = 'BDF2',
                 dt           = dt,
                 rad_BC       = rad_BC,
                 hydro_BC     = hydro_BC,
                 cx_old       = cx_new,
                 cx_older     = deepcopy(cx_old),
                 hydro_old    = hydro_new,
                 hydro_older  = deepcopy(hydro_old),
                 rad_old      = rad_new,
                 rad_older    = deepcopy(rad_old),
                 slopes_older = deepcopy(slopes_old),
                 e_rad_old = e_rad_new,
                 e_rad_older = e_rad_old,
                 slope_limiter= slope_limiter,
                 psim_src     = psim_src,
                 psip_src     = psip_src,
                 rho_src      = rho_src,
                 mom_src      = mom_src,
                 E_src        = E_src,
                 t_old        = t_old,
                 Qpsi_old     = Qpsi_old,
                 Qmom_old     = Qmom_old,
                 Qerg_old     = Qerg_old,
                 Qpsi_older   = deepcopy(Qpsi_old),
                 Qmom_older   = deepcopy(Qmom_old),
                 Qerg_older   = deepcopy(Qerg_old),
                 Qrho_older   = deepcopy(Qrho_old))

              raise NotImplementedError("Balance checker is wrong, CN step is just a predictor, dont need the sources")

              # add up source totals for each cycle to total for whole time step
              src_totals = dict()
              for key in src_totals_cycle1:
                 src_totals[key] = src_totals_cycle1[key] + src_totals_cycle2[key]
             
          else: # assume it's a single step method

              # take time step without MUSCL-Hancock
              hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
              Qpsi_new, Qrho_new, Qmom_new, Qerg_new, src_totals =\
                 takeTimeStepRadiationMaterial(
                 mesh         = mesh,
                 time_stepper = time_stepper_this_step,
                 dt           = dt,
                 rad_BC       = rad_BC,
                 hydro_BC     = hydro_BC,
                 cx_old       = cx_old,
                 cx_older     = cx_older,
                 hydro_old    = hydro_old,
                 hydro_older  = hydro_older,
                 rad_old      = rad_old,
                 rad_older    = rad_older,
                 slopes_older = slopes_older,
                 e_rad_old    = e_rad_old,
                 e_rad_older  = e_rad_older,
                 slope_limiter = slope_limiter,
                 psim_src     = psim_src,
                 psip_src     = psip_src,
                 rho_src      = rho_src,
                 mom_src      = mom_src,
                 E_src        = E_src,
                 t_old        = t_old,
                 Qpsi_old     = Qpsi_old,
                 Qrho_old     = Qrho_old,
                 Qmom_old     = Qmom_old,
                 Qerg_old     = Qerg_old,
                 Qpsi_older   = Qpsi_older,
                 Qmom_older   = Qmom_older,
                 Qerg_older   = Qerg_older,
                 Qrho_older   = Qrho_older)

              # compute balance
              if check_balance and (time_stepper != 'BDF2' or time_index>1):
                 bal = BalanceChecker(mesh, problem_type, time_stepper, dt)
                 bal.computeBalance(rad_BC=rad_BC, hydro_old=hydro_old,
                    hydro_new=hydro_new, rad_old=rad_old, rad_new=rad_new,
                    hydro_older=hydro_older, rad_older=rad_older,
                    hydro_F_right=hydro_F_right, hydro_F_left=hydro_F_left, 
                    src_totals=src_totals, cx_new=cx_new,write=True)

       else: # problem_type == 'rad_hydro'

          # if user chose to use the 2-cycle scheme
          if use_2_cycles:

             print("  Cycle 1:")

             # take time step with MUSCL-Hancock
             hydro_half, rad_half, cx_half, slopes_old, e_rad_half,\
             Qpsi_half, Qmom_half, Qerg_half, Qrho_half, hydro_F_left, hydro_F_right,\
             src_totals_cycle1 =\
                takeTimeStepMUSCLHancock(
                mesh           = mesh,
                dt             = 0.5*dt, 
                rad_BC         = rad_BC,
                hydro_BC       = hydro_BC,
                slope_limiter  = slope_limiter,
                cx_old         = cx_old,
                cx_older       = cx_older,
                hydro_old      = hydro_old,
                hydro_older    = hydro_older,
                rad_old        = rad_old,
                rad_older      = None,
                slopes_older   = slopes_older,
                e_rad_old   = e_rad_old,
                e_rad_older = None,
                time_stepper_predictor='CN',
                time_stepper_corrector='CN',
                psim_src     = psim_src,
                psip_src     = psip_src,
                mom_src      = mom_src,
                E_src        = E_src,
                rho_src      = rho_src,
                t_old        = t_old,
                Qpsi_old     = Qpsi_old,
                Qrho_old     = Qrho_old,
                Qmom_old     = Qmom_old,
                Qerg_old     = Qerg_old,
                Qpsi_older   = None,
                Qmom_older   = None,
                Qerg_older   = None,
                Qrho_older   = None,
                verbosity    = verbosity,
                rho_f = rho_f, u_f = u_f, E_f = E_f,
                gamma_value = gamma_value,
                cv_value=cv_value
             )

             #Compute balance over first cycle
             if check_balance:
                print "++++++++++++++++++++++++++++++++++++++++++++++++++++"
                print "    END OF CYCLE 1 "
                bal = BalanceChecker(mesh, problem_type, 'CN', 0.5*dt)
                bal.computeBalance(rad_BC=rad_BC, hydro_old=hydro_old,
                   hydro_new=hydro_half, rad_old=rad_old, rad_new=rad_half,
                   hydro_F_right=hydro_F_right, hydro_F_left=hydro_F_left, 
                   src_totals=src_totals_cycle1, cx_new=cx_half,write=True)

             print("  Cycle 2:")

             # take time step with MUSCL-Hancock
             hydro_new, rad_new, cx_new, slopes_half, e_rad_new,\
             Qpsi_new, Qmom_new, Qerg_new, Qrho_new, hydro_F_left, hydro_F_right,\
             src_totals_cycle2 =\
                takeTimeStepMUSCLHancock(
                mesh           = mesh,
                dt             = 0.5*dt, 
                rad_BC         = rad_BC,
                hydro_BC       = hydro_BC,
                slope_limiter  = slope_limiter,
                cx_old         = cx_half,
                cx_older       = cx_old,
                hydro_old      = hydro_half,
                hydro_older    = hydro_old,
                rad_old        = rad_half,
                rad_older      = rad_old,
                slopes_older   = slopes_old,
                e_rad_old   = e_rad_half,
                e_rad_older = e_rad_old,
                time_stepper_predictor='CN',
                time_stepper_corrector='BDF2',
                psim_src     = psim_src,
                psip_src     = psip_src,
                mom_src      = mom_src,
                E_src        = E_src,
                rho_src      = rho_src,
                t_old        = t_old + 0.5*dt,
                Qpsi_old     = Qpsi_half,
                Qmom_old     = Qmom_half,
                Qerg_old     = Qerg_half,
                Qrho_old     = Qrho_half,
                Qpsi_older   = Qpsi_old,
                Qmom_older   = Qmom_old,
                Qerg_older   = Qerg_old,
                Qrho_older   = Qrho_old,
                verbosity    = verbosity,
                rho_f = rho_f, u_f = u_f, E_f = E_f,
                gamma_value = gamma_value,
                cv_value=cv_value)

             #Compute balance over second cycle
             if check_balance:
                print "++++++++++++++++++++++++++++++++++++++++++++++++++++"
                print "    END OF CYCLE 2 "
                bal = BalanceChecker(mesh, problem_type, 'BDF2', 0.5*dt)
                bal.computeBalance(rad_BC=rad_BC, hydro_old=hydro_half,
                   hydro_new=hydro_new, rad_old=rad_half, rad_new=rad_new,
                   hydro_older=hydro_old, rad_older=rad_old,
                   hydro_F_right=hydro_F_right, hydro_F_left=hydro_F_left, 
                   src_totals=src_totals_cycle2, cx_new=cx_new,write=True)

          else: # use only 1 cycle

             # for first step, can't use BDF2; use CN instead
             time_stepper_corrector = time_stepper

             if time_stepper_corrector == 'BDF2':
                if time_index == 1:
                   time_stepper_corrector = 'CN'
                else:
                   time_stepper_corrector = 'BDF2'

                #Because of bad coding, dirichlet_BC only work here for fixed dt
                if dt_option == 'CFL':
                   if rad_BC.has_mms_func:
                      raise NotImplementedError("For 1 cycle, BDF2 does not work" \
                           " with MMS, time-dependent radiation boundaries")
            
             # take time step with MUSCL-Hancock
             hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
             Qpsi_new, Qmom_new, Qerg_new, Qrho_new, hydro_F_left, hydro_F_right,\
             src_totals =\
                takeTimeStepMUSCLHancock(
                mesh           = mesh,
                dt             = dt, 
                rad_BC         = rad_BC,
                hydro_BC       = hydro_BC,
                slope_limiter  = slope_limiter,
                cx_old         = cx_old,
                cx_older       = cx_older,
                hydro_old      = hydro_old,
                hydro_older    = hydro_older,
                rad_old        = rad_old,
                rad_older      = rad_older,
                slopes_older   = slopes_older,
                e_rad_old   = e_rad_old,
                e_rad_older = e_rad_older,
                time_stepper_predictor='CN',
                time_stepper_corrector=time_stepper_corrector,
                psim_src     = psim_src,
                psip_src     = psip_src,
                mom_src      = mom_src,
                E_src        = E_src,
                rho_src      = rho_src,
                t_old        = t_old,
                Qpsi_old     = Qpsi_old,
                Qmom_old     = Qmom_old,
                Qerg_old     = Qerg_old,
                Qrho_old     = Qrho_old,
                Qpsi_older   = Qpsi_older,
                Qmom_older   = Qmom_older,
                Qerg_older   = Qerg_older,
                Qrho_older   = Qrho_older,
                verbosity    = verbosity,
                rho_f = rho_f, u_f = u_f, E_f = E_f,
                gamma_value = gamma_value,
                cv_value=cv_value)

             # compute balance
             if check_balance and (time_stepper != 'BDF2' or time_index>1):
                bal = BalanceChecker(mesh, problem_type, time_stepper, dt)
                bal.computeBalance(rad_BC=rad_BC, hydro_old=hydro_old,
                   hydro_new=hydro_new, rad_old=rad_old, rad_new=rad_new,
                   hydro_older=hydro_older, rad_older=rad_older,
                   hydro_F_right=hydro_F_right, hydro_F_left=hydro_F_left, 
                   src_totals=src_totals, cx_new=cx_new,write=True)

       # save older solutions
       cx_older  = deepcopy(cx_old)
       rad_older = deepcopy(rad_old)
       hydro_older = deepcopy(hydro_old)
       slopes_older = deepcopy(slopes_old)
       e_rad_older = deepcopy(e_rad_old)
       Qpsi_older = deepcopy(Qpsi_old)
       Qrho_older = deepcopy(Qrho_old)
       Qmom_older = deepcopy(Qmom_old)
       Qerg_older = deepcopy(Qerg_old)

       # save old solutions
       t_old = t_new
       cx_old  = deepcopy(cx_new)
       rad_old = deepcopy(rad_new)
       hydro_old = deepcopy(hydro_new)
       e_rad_old = deepcopy(e_rad_new)
       Qpsi_old = deepcopy(Qpsi_new)
       Qrho_old = deepcopy(Qrho_new)
       Qmom_old = deepcopy(Qmom_new)
       Qerg_old = deepcopy(Qerg_new)

       #Check if in Steady State
       end_at_SS = True
       if end_at_SS:

           trans_change = computeL2RelDiff(hydro_new, hydro_older, aux_func=lambda i:i.E())
           if trans_change < 1.e-7:
               print """Exiting transient (in transient.py) because steady state was detected as total energy only
                        changed by a relative change of %0.4e. Be careful of this over small time steps""" % trans_change
               break

 #  Plots the hydro and rad edge values for internal energy   
 #  plotIntErgs(mesh, e_rad_new, hydro_new, slopes_half)

   # return final solutions
   return rad_new, hydro_new


## Takes time step without any MUSCL-Hancock.
#
#  This should only be called if the problem type is 'rad_mat'.
#
def takeTimeStepRadiationMaterial(mesh, time_stepper, dt, rad_BC,
   cx_old=None, cx_older=None, hydro_old=None, hydro_older=None, rad_old=None, rad_older=None,
   hydro_BC=None, slopes_older=None, e_rad_old=None, e_rad_older=None,
   psim_src=None, psip_src=None, rho_src=None, mom_src=None, E_src=None,
   t_old=None, Qpsi_old=None, Qrho_old=None, Qmom_old=None, Qerg_old=None,
   Qpsi_older=None, Qrho_older=None, Qmom_older=None, Qerg_older=None, slope_limiter=None):

    # compute new extraneous sources
    Qpsi_new, Qmom_new, Qerg_new, Qrho_new = computeExtraneousSources(
       psim_src, psip_src, mom_src, E_src, mesh, t_old+dt, rho_src=rho_src)

    # update hydro BC
    hydro_BC.update(states=hydro_old, t=t_old)

    # update radiation boundary condition if necessary 
    rad_BC.update(t_new=t_old+dt, t_old=t_old, t_older=t_old-dt)

    # compute slopes
    slopes_old = HydroSlopes(hydro_old, bc=hydro_BC, limiter=slope_limiter)

    # if there is no material motion, then the homogeneous hydro solution
    # should be equal to the old hydro solution
    hydro_star = deepcopy(hydro_old)

    # perform nonlinear solve
    hydro_new, rad_new, cx_new, e_rad_new = nonlinearSolve(
       mesh         = mesh,
       time_stepper = time_stepper,
       problem_type = 'rad_mat',
       dt           = dt,
       rad_BC       = rad_BC,
       cx_old       = cx_old,
       cx_older     = cx_older,
       hydro_old    = hydro_old,
       hydro_older  = hydro_older,
       hydro_star   = hydro_star,
       rad_old      = rad_old,
       rad_older    = rad_older,
       slopes_old   = slopes_old,
       slopes_older = slopes_older,
       e_rad_old    = e_rad_old,
       e_rad_older  = e_rad_older,
       Qpsi_new     = Qpsi_new,
       Qrho_new     = Qrho_new,
       Qmom_new     = Qmom_new,
       Qerg_new     = Qerg_new,
       Qpsi_old     = Qpsi_old,
       Qrho_old     = Qrho_old,
       Qmom_old     = Qmom_old,
       Qerg_old     = Qerg_old,
       Qpsi_older   = Qpsi_older,
       Qmom_older   = Qmom_older,
       Qrho_older   = Qrho_older,
       Qerg_older   = Qerg_older)

    # add up sources for entire time step for balance checker
    src_totals =  computeMMSSrcTotal(mesh,dt,time_stepper,
      Qmom_new=Qmom_new,Qmom_old=Qmom_old,Qmom_older=Qmom_older,
      Qpsi_new=Qpsi_new,Qpsi_old=Qpsi_old,Qpsi_older=Qpsi_older,
      Qerg_new=Qerg_new,Qerg_old=Qerg_old,Qerg_older=Qerg_older,
      Qrho_new=Qrho_new,Qrho_old=Qrho_old,Qrho_older=Qrho_older)

    #Store the radiation flux values if necessary
    rad_BC.storeAllIncidentFluxes(rad_new, rad_old=rad_old, rad_older=rad_older)

    return hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
       Qpsi_new, Qrho_new, Qmom_new, Qerg_new, src_totals


## Takes time step with MUSCL-Hancock.
#
#  This should only be called if the problem type is 'rad_hydro'.
#
def takeTimeStepMUSCLHancock(mesh, dt, rad_BC, 
   cx_old, cx_older, hydro_old, hydro_older, rad_old, rad_older,
   hydro_BC, slope_limiter, slopes_older, e_rad_old, e_rad_older,
   psim_src, psip_src, mom_src, E_src, rho_src, t_old,
   Qpsi_old, Qmom_old, Qerg_old, Qpsi_older, Qmom_older, Qerg_older,
   Qrho_old=None, Qrho_older=None,
   time_stepper_predictor='CN', time_stepper_corrector='BDF2',verbosity=2,
   rho_f=None,u_f=None,E_f=None,gamma_value=None,cv_value=None):
    
   #This will print out all results
   debug_mode = False

   # assert that BDF2 was not chosen for the predictor time-stepper
   assert time_stepper_predictor != 'BDF2', 'BDF2 cannot be used in\
      the predictor step.'

   #optionally print out after each solve step
   if debug_mode:

     print "initial conditions"
     if rho_f != None:
         hydro_exact = computeAnalyticHydroSolution(mesh, t=t_old,
            rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)
     else:
         hydro_exact = None 

     plotHydroSolutions(mesh, hydro_old, x_exact=mesh.getCellCenters(),
          exact=hydro_exact)
    
   if verbosity > 1:
      print "    Predictor step:"

   # update hydro BC
   hydro_BC.update(states=hydro_old, t=t_old)

   # compute slopes
   slopes_old = HydroSlopes(hydro_old, bc=hydro_BC, limiter=slope_limiter)

   # perform predictor step of MUSCL-Hancock
   hydro_star = hydroPredictor(mesh, hydro_old, slopes_old, dt)


   if debug_mode:

      print "hydro_star predictor:"
      plotHydroSolutions(mesh, hydro_star, x_exact=mesh.getCellCenters(),
                exact=None)

   # compute new extraneous sources
   Qpsi_half, Qmom_half, Qerg_half, Qrho_half = computeExtraneousSources(
      psim_src, psip_src, mom_src, E_src, mesh, t_old+0.5*dt, rho_src=rho_src)

   #update rad BC to be at t+1/2, keep old at start of time step
   rad_BC.update(t_new=t_old+0.5*dt, t_old=t_old)

   # perform nonlinear solve
   hydro_half, rad_half, cx_half, e_rad_half = nonlinearSolve(
      mesh         = mesh,
      time_stepper = time_stepper_predictor,
      problem_type = 'rad_hydro',
      dt           = 0.5*dt,
      rad_BC       = rad_BC,
      cx_old       = cx_old,
      hydro_old    = hydro_old,
      hydro_star   = hydro_star,
      rad_old      = rad_old,
      slopes_old   = slopes_old,
      e_rad_old    = e_rad_old,
      Qpsi_new     = Qpsi_half,
      Qmom_new     = Qmom_half,
      Qerg_new     = Qerg_half,
      Qrho_new     = Qrho_half,
      Qpsi_old     = Qpsi_old,
      Qmom_old     = Qmom_old,
      Qerg_old     = Qerg_old,
      Qrho_old     = Qrho_old,
      Qpsi_older   = Qpsi_older, # this is a dummy argument
      Qmom_older   = Qmom_older, # this is a dummy argument
      Qerg_older   = Qerg_older, # this is a dummy argument
      verbosity    = verbosity)


   if debug_mode:
       print "hydro_half after nonlinear:"

       if rho_f != None:
           hydro_exact = computeAnalyticHydroSolution(mesh, t=t_old+0.5*dt,
              rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)
       else:
           hydro_exact = None 

       plotHydroSolutions(mesh, hydro_half, x_exact=mesh.getCellCenters(),
           exact=hydro_exact)

   if verbosity > 1:
      print "    Corrector step:"

   # update hydro BC
   hydro_BC.update(states=hydro_half, t=t_old+0.5*dt, slopes=slopes_old, edge_value=True)

   # perform corrector step of MUSCL-Hancock
   hydro_star, hydro_F_left, hydro_F_right = hydroCorrector(
      mesh, hydro_old, hydro_half, slopes_old, dt, bc=hydro_BC)

   if debug_mode:

      print "After hydro corrector step, state *"
      plotHydroSolutions(mesh, hydro_star, x_exact=mesh.getCellCenters(),
               exact=None)

   # compute new extraneous sources
   Qpsi_new, Qmom_new, Qerg_new, Qrho_new = computeExtraneousSources(
      psim_src, psip_src, mom_src, E_src, mesh, t_old+dt, rho_src=rho_src)

   #update rad BC to be at end of time step for implicit terms, old term is
   #kept at beginning of time step. NOTE IF BDF2 in play, here we have assumed
   #That you are using a step back of twice the size of dt. It is better
   #to just use periodic BC 
   rad_BC.update(t_new=t_old+dt, t_old=t_old, t_older=t_old-dt)

   # perform nonlinear solve
   hydro_new, rad_new, cx_new, e_rad_new = nonlinearSolve(
      mesh         = mesh,
      time_stepper = time_stepper_corrector,
      problem_type = 'rad_hydro',
      dt           = dt,
      rad_BC     = rad_BC,
      cx_old       = cx_old,
      cx_older     = cx_older,
      hydro_old    = hydro_old,
      hydro_older  = hydro_older,
      hydro_star   = hydro_star,
      rad_old      = rad_old,
      rad_older    = rad_older,
      slopes_old   = slopes_old,
      slopes_older = slopes_older,
      e_rad_old = e_rad_old,
      e_rad_older = e_rad_older,
      Qpsi_new     = Qpsi_new,
      Qmom_new     = Qmom_new,
      Qerg_new     = Qerg_new,
      Qrho_new     = Qrho_new,
      Qpsi_old     = Qpsi_old,
      Qmom_old     = Qmom_old,
      Qerg_old     = Qerg_old,
      Qrho_old     = Qrho_old,
      Qpsi_older   = Qpsi_older,
      Qrho_older   = Qrho_older,
      Qmom_older   = Qmom_older,
      Qerg_older   = Qerg_older,
      verbosity    = verbosity)

   if debug_mode:

      if rho_f != None:
          hydro_exact = computeAnalyticHydroSolution(mesh, t=t_old+dt,
             rho=rho_f, u=u_f, E=E_f, cv=cv_value, gamma=gamma_value)
      else:
          hydro_exact = None 

    
      plotHydroSolutions(mesh, hydro_new, x_exact=mesh.getCellCenters(),
            exact=hydro_exact)

   # add up sources for entire time step for balance checker
   src_totals =  computeMMSSrcTotal(mesh,dt,time_stepper_corrector,
         Qmom_new=Qmom_new,Qmom_old=Qmom_old,Qmom_older=Qmom_older,
         Qpsi_new=Qpsi_new,Qpsi_old=Qpsi_old,Qpsi_older=Qpsi_older,
         Qerg_new=Qerg_new,Qerg_old=Qerg_old,Qerg_older=Qerg_older,
         Qrho_new=Qrho_new,Qrho_old=Qrho_old,Qrho_older=Qrho_older)

   #Store the incident fluxes on boundary for computing balance
   rad_BC.storeAllIncidentFluxes(rad_new, rad_old=rad_old, rad_older=rad_older)
   

   if verbosity > 1:
      print ""


   return hydro_new, rad_new, cx_new, slopes_old, e_rad_new,\
      Qpsi_new, Qmom_new, Qerg_new, Qrho_new, hydro_F_left, hydro_F_right,\
      src_totals


## Computes all extraneous sources at time \f$t\f$
#
#  @param[in] psim_src
#  @param[in] psip_src
#  @param[in] mom_src
#  @param[in] E_src
#  @param[in] mesh
#  @param[in] t
#
#  @return extraneous source vectors evaluated at \f$t\f$:
#    \f$Q^{ext,\pm}\f$, \f$Q^{ext,\rho u}\f$, \f$Q^{ext,E}\f$
#
def computeExtraneousSources(psim_src, psip_src, mom_src, E_src, mesh, t,
        rho_src=None):

   print "      Computing MMS sources..."

   # compute radiation extraneous source
   if psim_src != None and psip_src != None:
      Qpsi = computeRadiationExtraneousSource(psim_src, psip_src, mesh, t)
   else:
      Qpsi = np.zeros(mesh.n_elems*4)

   # compute momentum extraneous source
   if mom_src != None:
      Qmom = computeMomentumExtraneousSource(mom_src, mesh, t)
   else:
      Qmom = np.zeros(mesh.n_elems)

   # compute energy extraneous source
   if E_src != None:
      Qerg = computeEnergyExtraneousSource(E_src, mesh, t)
   else:
      Qerg = [(0.0,0.0) for i in xrange(mesh.n_elems)]

   # compute rho extraneous source
   if rho_src != None:
      Qrho = computeMomentumExtraneousSource(rho_src, mesh, t)
   else:
      Qrho = np.zeros(mesh.n_elems)

   return Qpsi, Qmom, Qerg, Qrho

#--------------------------------------------------------------------------------
## Function to compute the src totals for MMS sources
#
def computeMMSSrcTotal(mesh, dt, time_stepper, Qpsi_new=None, Qpsi_old=None,
        Qpsi_older=None, Qrho_new=None, Qrho_old=None, Qrho_older=None,Qmom_new=None, Qmom_old=None, Qmom_older=None,
        Qerg_new=None,Qerg_old=None,Qerg_older=None):

   vol = mesh.getElement(0).dx
   # add up sources for each equation, depending on time stepper
   #TODO This will be much easier once MMS sources are accurate with quadrature
   srcs = {}

   if time_stepper == 'BE':

      #Rad source is vector passed to  needs to be integrated over angle and volume
      #If you work out the math, its just the sum *0.5
      vol = mesh.getElement(0).dx
      srcs["rad"] = 0.5*vol*sum(Qpsi_new)*dt
      srcs["mom"] = sum([vol*dt*i for i in Qmom_new])
      srcs["mom"] += sumRadMomQ(vol,dt,Qpsi_new) #add in momentum from radiation
      srcs["erg"] = sum([vol*dt*0.5*(i[0]+i[1]) for i in Qerg_new])
      srcs["rho"] = sum([vol*dt*i for i in Qrho_new])

   elif time_stepper == 'CN':

      #Rad source is vector passed to  needs to be integrated over angle and volume
      #If you work out the math, its just the sum *0.5
      vol = mesh.getElement(0).dx
      srcs["rad"] = 0.5*(0.5*vol*sum(Qpsi_new)*dt)
      srcs["rad"] += 0.5*(0.5*vol*sum(Qpsi_old)*dt)
      srcs["mom"] = 0.5*(sum([vol*dt*i for i in Qmom_new]))
      srcs["mom"] += 0.5*(sum([vol*dt*i for i in Qmom_old]))
      srcs["mom"] += 0.5*(sumRadMomQ(vol,dt,Qpsi_new)+sumRadMomQ(vol,dt,Qpsi_old))
      srcs["erg"] = 0.5*sum([vol*dt*0.5*(i[0]+i[1]) for i in Qerg_new])
      srcs["erg"] += 0.5*sum([vol*dt*0.5*(i[0]+i[1]) for i in Qerg_old])
      srcs["rho"] = 0.5*(sum([vol*dt*i for i in Qrho_new]))
      srcs["rho"] += 0.5*(sum([vol*dt*i for i in Qrho_old]))
      
   elif time_stepper == 'BDF2':

      #Rad source is vector passed to  needs to be integrated over angle and volume
      #If you work out the math, its just the sum *0.5
      vol = mesh.getElement(0).dx
      srcs["rad"] = 2./3.*(0.5*vol*sum(Qpsi_new)*dt)
      srcs["rad"] += 1./6.*(0.5*vol*sum(Qpsi_old)*dt)
      srcs["rad"] += 1./6.*(0.5*vol*sum(Qpsi_older)*dt)
      srcs["mom"] = 2./3*(sum([vol*dt*i for i in Qmom_new]))
      srcs["mom"] += 1/6.*(sum([vol*dt*i for i in Qmom_old]))
      srcs["mom"] += 1/6.*(sum([vol*dt*i for i in Qmom_older]))
      srcs["mom"] += 2./3*sumRadMomQ(vol,dt,Qpsi_new)+1./6.*sumRadMomQ(vol,dt,Qpsi_old) \
                    +1./6.*(sumRadMomQ(vol,dt,Qpsi_older))
      srcs["erg"] = 2./3.*sum([vol*dt*0.5*(i[0]+i[1]) for i in Qerg_new])
      srcs["erg"] += 1./6*sum([vol*dt*0.5*(i[0]+i[1]) for i in Qerg_old])
      srcs["erg"] += 1./6*sum([vol*dt*0.5*(i[0]+i[1]) for i in Qerg_older])
      srcs["rho"] = 2./3*(sum([vol*dt*i for i in Qrho_new]))
      srcs["rho"] += 1/6.*(sum([vol*dt*i for i in Qrho_old]))
      srcs["rho"] += 1/6.*(sum([vol*dt*i for i in Qrho_older]))

   return srcs


def sumRadMomQ(vol, dt,Qpsi):

   #Add in the momentum source from radiation by computing moments
   MomRad = 0.0
   c = GC.SPD_OF_LGT
   for i in range(len(Qpsi)/4):

      # get indices
      Lm = getIndex(i,"L","-") # dof L,-
      Lp = getIndex(i,"L","+") # dof L,+
      Rm = getIndex(i,"R","-") # dof R,-
      Rp = getIndex(i,"R","+") # dof R,+

      QL = 1./c*(mu['+']*Qpsi[Lp] + mu['-']*Qpsi[Lm])
      QR = 1./c*(mu['+']*Qpsi[Rp] + mu['-']*Qpsi[Rm])

      MomRad += 0.5*(QL + QR)*vol*dt

   return MomRad
