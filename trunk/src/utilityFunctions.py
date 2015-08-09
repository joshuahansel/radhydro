## @package src.utilityFunctions
#  Contains helper functions that do not belong in any particular class.

from math import log, sqrt
import numpy as np
import globalConstants as GC
from crossXInterface import CrossXInterface
from hydroState import HydroState

#-----------------------------------------------------------------------------------
## Converge f_L and f_R to f_a and f_x
def EdgToMom(f_l, f_r):
    
    # Based on f(x) = f_a + 2/h*f_x(x - x_i)
    f_a = 0.5*(f_l + f_r)
    f_x = 0.5*(f_r - f_l)

    return f_a, f_x


#-----------------------------------------------------------------------------------
## Converge f_x and f_a to f_l and f_r
def MomToEdg(f_a, f_x):
    
    # Based on f(x) = f_a + 2/h*f_x(x - x_i)
    return (f_a - f_x), (f_a + f_x)

#-----------------------------------------------------------------------------------
## Index function for 1-D LD S-2 DOF handler. This gives the global index in array
#
#  @param[in] i     cell index, from 1 to n-1, where n is number of cells
#  @param[in] side  string, either "L" or "R", corresponding to left or right dof
#  @param[in] dir   string, either "-" or "+", corresponding to - or + direction
#
#  @return    global dof getIndex
#
def getIndex(i, side, dir):

    side_shift = {"L" : 0, "R" : 2}
    dir_shift  = {"-" : 0, "+" : 1}
    return 4*i + side_shift[side] + dir_shift[dir]

#-----------------------------------------------------------------------------------
## Local index function for S-2 DOF. This gives the local index in array of for the 
#  different DOF, essentially no offset

def getLocalIndex(side, dir):

    return getIndex(0, side, dir)

#-----------------------------------------------------------------------------------
## Computes convergence rates
#
#  @param[in] dx        list of mesh sizes or time step sizes for each cycle
#  @param[in] err       list of errors for each cycle
#
#  @return  list of convergence rates. The size of this list will be the number
#           of cycles minus one.
#
def computeConvergenceRates(dx,err):

   # determine number of refinement cycles from length of lists
   n_cycles = len(dx)

   # initialize list of rates
   rates = list()

   # loop over cycles
   for cycle in xrange(n_cycles):
      # compute convergence rate
      if cycle > 0:
         rates.append(log(err[cycle]/err[cycle-1]) / log(dx[cycle]/dx[cycle-1]))

   return rates

#-----------------------------------------------------------------------------------
## Prints convergence table and convergence rates
#
#  @param[in] dx        list of mesh sizes or time step sizes for each cycle
#  @param[in] err       list of errors for each cycle
#  @param[in] rates     list of convergence rates. The size of this list will
#                       be the number of cycles minus one. If this argument
#                       is not provided, rates are computed in this function.
#  @param[in] dx_desc   string description for the size quantity, e.g.,
#                       'dx' or 'dt'
#  @param[in] err_desc  string description for the error, e.g., 'L1' or 'L2'
#
def printConvergenceTable(dx,err,rates=None,dx_desc='size',err_desc='err'):

   # compute rates if they were not provided
   if rates is None:
      rates = computeConvergenceRates(dx,err)

   # determine number of refinement cycles from length of lists
   n_cycles = len(dx)

   # print header
   print('\n%11s %11s    Rate' % (dx_desc,err_desc))
   print('-------------------------------')

   # loop over cycles
   for cycle in xrange(n_cycles):
      # compute convergence rate
      if cycle > 0:
         rate_string = '%7.3f' % rates[cycle-1]
      else:
         rate_string = '-'

      # print line to convergence table
      print('%11.3e %11.3e %7s' % (dx[cycle],err[cycle],rate_string))
   print('\n')

## Function to compute the discrete \f$L^1\f$ norm of an array \f$\mathbf{y}\f$
#  of 2-tuples: \f$\|\mathbf{y}\|_1 = \sum\limits_i |y_{i,L}| + |y_{i,R}|\f$
#
#  @param[in] values  array \f$\mathbf{y}\f$ of 2-tuples:
#                     \f$y_i = (y_{i,L},y_{i,R})\f$
#
#  @return  the discrete \f$L^1\f$ norm \f$\|y\|_1\f$
#
def computeDiscreteL1Norm(values):

   # number of tuples in array
   n = len(values)

   # initialize norm to zero
   norm = 0.0

   # loop over tuples
   for i in xrange(n):
      norm += abs(values[i][0]) + abs(values[i][1])

   return norm

## Function to compute the error for the hydro solution.
#
def computeHydroError(hydro, hydro_exact):

#   # number of elements
#   n = len(hydro)
#
#   # initialize error to zero
#   err = 0.0
#
#   # loop over tuples
#   for i in xrange(n):
#      err += abs(hydro[i].e - hydro_exact[i].e)

   err = computeL2RelDiff(hydro, hydro_exact, aux_func=lambda x: x.e)

   return err

## Function to compute the L2 integral based on relative error of the larger 
#  of the two values. Passed in values are 2-tuples. Average of the L and R values is
#  used to computed average
#
#  @param[in] values_1 array \f$\mathbf{y}\f$ of 2-tuples:
#                     \f$y_i = (y_{i,L},y_{i,R})\f$
#  @param[in] values_2 array \f$\mathbf{y}\f$ of 2-tuples:
#                 \f$y_i = (y_{i,L},y_{i,R})\f$
#  @param[in] aux_func optional function to apply to the values in tuple
#
#  @return  the discrete \f$L^2\f$ norm \f$\|y\|_1\f$
#
def computeL2RelDiffTuples(values1, values2, aux_func=None):

   #apply aux_func, if None default to just the value
   if aux_func == None:
       aux_func = lambda x: x
   f = aux_func

   # get averages
   avg1 = np.array([0.5*(f(i[0])+f(i[1])) for i in values1])
   avg2 = np.array([0.5*(f(i[0])+f(i[1])) for i in values2])

   #compute norms
   norm1 = np.linalg.norm(avg1)
   norm2 = np.linalg.norm(avg2)
   norm_diff = np.linalg.norm(avg1 - avg2)

   return norm_diff/(max(norm1,norm2))

## Function to compute the L2 integral based on relative error of the larger 
#  of the two values. Passed in values are arrays.
#
#  @param[in] values_1 array \f$\mathbf{y}\f$
#  @param[in] values_2 array \f$\mathbf{y}\f$
#  @param[in] aux_func optional function to apply to the values
#
#  @return  the discrete \f$L^2\f$ norm \f$\|y\|_2\f$
#
def computeL2RelDiff(values1, values2, aux_func=None):

   #apply aux_func, if None default to just the value
   if aux_func == None:
       aux_func = lambda x: x
   f = aux_func

   # get values
   vals1 = np.array([f(i) for i in values1])
   vals2 = np.array([f(i) for i in values2])

   #compute norms
   norm1 = np.linalg.norm(vals1)
   norm2 = np.linalg.norm(vals2)
   norm_diff = np.linalg.norm(vals1 - vals2)

   return norm_diff/(max(norm1,norm2))

## Computes effective scattering fraction \f$\nu^k\f$ in linearization
#
#  @param[in] T      Previous iteration temperature \f$T^k\f$
#  @param[in] sig_a  Previous iteration absorption cross section \f$\sigma_a^k\f$
#  @param[in] rho    New density \f$\rho^{n+1}\f$
#  @param[in] spec_heat   Previous iteration specific heat \f$c_v^k\f$
#  @param[in] dt          time step size \f$\Delta t\f$
#  @param[in] scale       coefficient corresponding to time-stepper \f$\gamma\f$
#
def getNu(T, sig_a, rho, spec_heat, dt, scale):

    ## compute \f$c\Delta t\f$
    c_dt = GC.SPD_OF_LGT*dt

    ## compute \f$\beta^k\f$
    beta = 4.*GC.RAD_CONSTANT * T * T * T / spec_heat

    # Evaluate numerator
    num  = scale*sig_a*c_dt*beta/rho

    return num/(1. + num)

#--------------------------------------------------------------------------------
## Computes effective cross sections for linearization. 
#  
#  The effective re-emission source is included as a scattering cross section
#  given by
#  \f[
#      \tilde{\sigma_s} = \sigma_s + \nu\sigma_a(T^k),
#  \f]
#  where \f$\nu\f$ is the effective scattering ratio, which depends on the
#  temporal discretization.
#
#
def computeEffectiveOpacities(time_stepper, dt, cx_prev, hydro_prev,
    slopes_old, e_slopes_old):

    # get coefficient corresponding to time-stepper
    scales = {"CN":0.5, "BE":1., "BDF2":2./3.}
    scale = scales[time_stepper]

    cx_effective = list()

    # loop over cells:
    for i in range(len(cx_prev)):

        cx_i = list()

        # get hydro state and specific heat
        state = hydro_prev[i]
        spec_heat = state.spec_heat

        # get density and temperature at edges
        rho = computeEdgeDensities(i, state, slopes_old)
        T   = computeEdgeTemperatures(state, e_slopes_old[i])

        # loop over edges
        for x in range(2): 

            # get cross sections
            sig_a = cx_prev[i][x].sig_a
            sig_s = cx_prev[i][x].sig_s

            # compute effective scattering ratio
            nu = getNu(T[x], sig_a, rho[x], spec_heat, dt, scale)
        
            #Create new FIXED cross section instance. No need to add scale term
            #here because it will be included in scattering source term
            sig_s_effective = nu*sig_a + sig_s
            cx_i.append( CrossXInterface(sig_s_effective, sig_s+sig_a) )

        cx_effective.append(tuple(cx_i))

    return cx_effective


## Computes edge densities for a cell given hydro state and slopes
#
#  @param[in] i       cell index
#  @param[in] state   average hydro state for cell \f$i\f$
#  @param[in] slopes  HydroSlopes object
#
#  @return \f$(\rho_{i,L},\rho_{i,R})\f$
#
def computeEdgeDensities(i, state, slopes):

   # compute edge velocities
   rhoL = state.rho - 0.5*slopes.rho_slopes[i]
   rhoR = state.rho + 0.5*slopes.rho_slopes[i]
   return (rhoL, rhoR)


## Computes edge velocities for a cell given hydro state and slopes
#
#  @param[in] i       cell index
#  @param[in] state   average hydro state for cell \f$i\f$
#  @param[in] slopes  HydroSlopes object
#
#  @return \f$(u_{i,L},u_{i,R})\f$
#
def computeEdgeVelocities(i, state, slopes):

   # compute edge velocities
   rho = state.rho
   u   = state.u
   mom = rho*u
   rhoL = rho - 0.5*slopes.rho_slopes[i]
   rhoR = rho + 0.5*slopes.rho_slopes[i]
   momL = mom - 0.5*slopes.mom_slopes[i]
   momR = mom + 0.5*slopes.mom_slopes[i]
   return (momL / rhoL, momR / rhoR)


## Computes edge temperatures for a cell given hydro state and internal energy slope
#
#  @param[in] state    average hydro state for cell \f$i\f$
#  @param[in] e_slope  internal energy slope
#
#  @return \f$(T_{i,L},T_{i,R})\f$
#
def computeEdgeTemperatures(state, e_slope):

   # compute internal energies at left and right edges
   eL = state.e - 0.5*e_slope
   eR = state.e + 0.5*e_slope

   # get specific heat
   cv = state.spec_heat

   # compute edge temperature using internal energy slope
   return (eL / cv, eR / cv)

## Computes edge internal energies for a cell, given hydro state and total
#  energy slope.
#
#  This is done in a matter so that 0.5*(E_L+E_R)=E_i, preserving
#  total energy conservation
#
#  @param[in] state    average hydro state for cell \f$i\f$
#  @param[in] slopes    hydro slope object
#
#  @return \f$(e_{i,L},e_{i,R})\f$
#
def computeHydroInternalEnergies(i, state, slopes):

   #Use conserved variables to construct e_L and e_R
   rho, mom, erg = state.getConservativeVariables()

   #Need all values at edges
   rho_L = rho - 0.5*slopes.rho_slopes[i]
   rho_R = rho + 0.5*slopes.rho_slopes[i]
   mom_L = mom - 0.5*slopes.mom_slopes[i]
   mom_R = mom + 0.5*slopes.mom_slopes[i]
   erg_L = erg - 0.5*slopes.erg_slopes[i]
   erg_R = erg + 0.5*slopes.erg_slopes[i]

   # compute internal energies at left and right edges
   u_L = mom_L/rho_L
   u_R = mom_R/rho_R

   e_L = erg_L/rho_L - 0.5*u_L**2
   e_R = erg_R/rho_R - 0.5*u_R**2

   # compute edge temperature using internal energy slope
   return (e_L, e_R)

## Computes edge internal energies for a cell given hydro state and internal energy slope
#
#  @param[in] state    average hydro state for cell \f$i\f$
#  @param[in] e_slope  internal energy slope
#
#  @return \f$(e_{i,L},e_{i,R})\f$
#
def computeEdgeInternalEnergies(state, e_slope):

   # compute internal energies at left and right edges
   eL = state.e - 0.5*e_slope
   eR = state.e + 0.5*e_slope

   # compute edge temperature using internal energy slope
   return (eL, eR)


## Updates all cross sections.
#
def updateCrossSections(cx,hydro,slopes,e_slopes):

   # loop over cells
   for i in range(len(hydro)):

      # get cell average state, specific heat, gamma, and internal energy slope
      state_avg = hydro[i]
      spec_heat = state_avg.spec_heat
      gamma = state_avg.gamma
      de = e_slopes[i]

      # compute edge quantities
      rho = computeEdgeDensities(i, state_avg, slopes)
      u = computeEdgeVelocities(i, state_avg, slopes)
      e = computeEdgeInternalEnergies(state_avg, e_slopes[i])

      # loop over edges and create state for each edge
      for edge in [0,1]:

         # create edge state
         state = HydroState(rho=rho[edge],u=u[edge],e=e[edge],
            spec_heat=spec_heat, gamma=gamma)
 
         # update edge cross section
         cx[i][edge].updateCrossX(state)


## Computes a vector in the ordering of radiation dofs, provided functions
#  of (x,t) for both directions.
#
#  The input function handles are functions of (x,t), and the output source
#  is given in the ordering of radiation dofs.
#
#  @param[in] f_minus  function handle for the minus direction
#  @param[in] f_plus   function handle for the plus direction
#  @param[in] mesh     mesh
#  @param[in] t        time at which to evaluate the functions
#
#  @return vector in the ordering of radiation dofs
#
def computeRadiationVector(f_minus, f_plus, mesh, t):

   # initialize vector
   y = np.zeros(mesh.n_elems*4)

   # loop over elements
   for i in range(mesh.n_elems):

      # get left and right x points on element
      xL = mesh.getElement(i).xl
      xR = mesh.getElement(i).xr

      # get global indices
      iLm = getIndex(i,"L","-") # dof i,L,-
      iLp = getIndex(i,"L","+") # dof i,L,+
      iRm = getIndex(i,"R","-") # dof i,R,-
      iRp = getIndex(i,"R","+") # dof i,R,+

      # compute source
      y[iLm] = f_minus(xL, t)
      y[iLp] = f_plus(xL, t)
      y[iRm] = f_minus(xR, t)
      y[iRp] = f_plus(xR, t)

   return y


## Prints a tuple list.
#
def printTupleList(tuple_list):

    for tuple_i in tuple_list:

        for j in tuple_i:

            print j


## Computes hydro states from analytic functions of (x,t)
#
#  @param[in] mesh  mesh
#  @param[in] t     time value
#  @param[in] rho   analytic function for density, \f$\rho(x,t)\f$
#  @param[in] u     analytic function for velocity, \f$u(x,t)\f$
#  @param[in] E     analytic function for total energy, \f$E(x,t)\f$
#  @param[in] cv    value for specific heat, \f$c_v\f$
#  @param[in] gamma value for gamma constant, \f$\gamma\f$
#
#  @return list of hydro states analytically evaluated at each cell center
#
def computeAnalyticHydroSolution(mesh,t,rho,u,E,cv,gamma):

   hydro = list()
   for i in xrange(mesh.n_elems):

      # get cell center
      x_i = mesh.getElement(i).x_cent

      # evaluate functions at cell center
      rho_i = rho(x=x_i, t=t)
      u_i   =   u(x=x_i, t=t)
      E_i   =   E(x=x_i, t=t)

      # add hydro state for cell
      hydro.append(HydroState(rho=rho_i, u=u_i, E=E_i,
         spec_heat=cv, gamma=gamma))

   return hydro


