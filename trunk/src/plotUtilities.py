## @package src.plotUtilities
#  Provides functions for plotting radiation and hydrodynamics solutions.

import matplotlib.pyplot as plt
import globalConstants as GC
import numpy as np
from numpy import array
import operator           # for adding tuples to each other elementwise
from matplotlib import rc # for rendering tex in plots

## Function to plot angular flux.
#
#  @param[in] mesh      mesh data
#  @param[in] psi_minus S-2 angular flux solution for the minus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^-\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_minus[i]\f$=(\Psi^-_{i,L},
#                       \Psi^-_{i,R})\f$
#  @param[in] psi_plus  S-2 angular flux solution for the plus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^+\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_plus[i]\f$=(\Psi^+_{i,L},
#                       \Psi^+_{i,R})\f$
#  @param[in] save      boolean flag for saving the plot as a .pdf file
#  @param[in] psi_minus_exact  exact angular flux solution for minus direction,
#                              passed as an array of values at each edge.
#  @param[in] psi_plus_exact   exact angular flux solution for minus direction,
#                              passed as an array of values at each edge.
#
def plotAngularFlux(mesh, psi_minus, psi_plus, save=False,
   psi_minus_exact=None, psi_plus_exact=None):

   # create x-points
   x            = makeXPoints(mesh)           # discontinuous x-points
   x_continuous = makeContinuousXPoints(mesh) # continuous    x-points

   # transform arrays of tuples into arrays
   psi_minus_array = makeYPoints(psi_minus)
   psi_plus_array  = makeYPoints(psi_plus)

   # plot
   plt.rc('text', usetex=True)         # use tex to generate text
   plt.rc('font', family='sans-serif') # use sans-serif font family
   plt.plot(x, psi_minus_array, 'r-o', label='$\Psi^-$')
   plt.plot(x, psi_plus_array,  'b-x', label='$\Psi^+$')
    
   # plot exact solutions if there are any
   if psi_minus_exact is not None:
      plt.plot(x_continuous, psi_minus_exact, 'k--', label='$\Psi^-$, exact')
   if psi_plus_exact  is not None:
      plt.plot(x_continuous, psi_plus_exact,  'k-',  label='$\Psi^+$, exact')

   # annotations
   plt.xlabel('$x$')
   plt.ylabel('Angular Flux')
   plt.legend(loc='best')

   # save if requested
   if save:
      plt.savefig('angularflux.pdf')
   else:
      plt.show()

## Function to plot scalar flux.
#
#  @param[in] mesh      mesh data
#  @param[in] psi_minus S-2 angular flux solution for the minus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^-\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_minus[i]\f$=(\Psi^-_{i,L},
#                       \Psi^-_{i,R})\f$
#  @param[in] psi_plus  S-2 angular flux solution for the plus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^+\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_plus[i]\f$=(\Psi^+_{i,L},
#                       \Psi^+_{i,R})\f$
#  @param[in] save      boolean flag for saving the plot as a .pdf file
#  @param[in] scalar_flux_exact  exact scalar flux solution, passed as an
#                                array of values at each edge.
#  @param[in] exact_data_continuous  boolean flag that specifies if the provided
#                                    exact solution data is continuous (True)
#                                    or is given as discontinuous tuples (False).
#
def plotScalarFlux(mesh, psi_minus, psi_plus, save=False, scalar_flux_exact=None,
   exact_data_continuous=True):

   # create new figure
   plotScalarFlux.count += 1
   plt.figure(plotScalarFlux.count)

   # create x-points
   x            = makeXPoints(mesh)           # discontinuous x-points
   x_continuous = makeContinuousXPoints(mesh) # continuous    x-points

   # compute scalar flux from angular fluxes
   scalar_flux = computeScalarFlux(psi_minus, psi_plus)

   # transform array of tuples into array
   scalar_flux_array  = makeYPoints(scalar_flux)

   # plot
   plt.rc('text', usetex=True)         # use tex to generate text
   plt.rc('font', family='sans-serif') # use sans-serif font family
   plt.plot(x, scalar_flux_array, 'r-o', label='$\phi$')
    
   # plot exact solution if there is one
   if scalar_flux_exact is not None:
      if exact_data_continuous:
         plt.plot(x_continuous, scalar_flux_exact, 'k--', label='$\phi$, exact')
      else:
         exact_array = makeYPoints(scalar_flux_exact)
         plt.plot(x,            exact_array,       'k--', label='$\phi$, exact')

   # annotations
   plt.xlabel('$x$')
   plt.ylabel('Scalar Flux')
   plt.legend(loc='best')

   # save if requested
   if save:
      plt.savefig('scalarflux.pdf')
   else:
      plt.show()

plotScalarFlux.count = 0

## Function to compute scalar flux from angular fluxes.
#
#  @param[in] psi_minus S-2 angular flux solution for the minus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^-\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_minus[i]\f$=(\Psi^-_{i,L},
#                       \Psi^-_{i,R})\f$
#  @param[in] psi_plus  S-2 angular flux solution for the plus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^+\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_plus[i]\f$=(\Psi^+_{i,L},
#                       \Psi^+_{i,R})\f$
#  @return  scalar flux solution, \f$\phi\f$, as an array of tuples of left
#           and right values, e.g., scalar_flux[i]\f$=(\phi_{i,L},\phi_{i,R})\f$
#
def computeScalarFlux(psi_minus, psi_plus):
   scalar_flux = [tuple(y for y in tuple(map(operator.add, psi_minus[i], psi_plus[i]))
                       ) for i in xrange(len(psi_minus))]
   return scalar_flux

## Function to compute energy density E from angular fluxes.
#
#  @param[in] psi_minus S-2 angular flux solution for the minus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^-\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_minus[i]\f$=(\Psi^-_{i,L},
#                       \Psi^-_{i,R})\f$
#  @param[in] psi_plus  S-2 angular flux solution for the plus direction
#                       multiplied by \f$2\pi\f$, i.e.,
#                       \f$\Psi^+\f$, passed as an array of tuples of left
#                       and right values, e.g., psi_plus[i]\f$=(\Psi^+_{i,L},
#                       \Psi^+_{i,R})\f$
#  @return  energy density, \f$E\f$, as an array of tuples of left
#           and right values, e.g., E[i]\f$=(E_{i,L},E_{i,R})\f$
#
def computeEnergyDensity(psi_minus, psi_plus):
   E = [tuple((1./GC.SPD_OF_LGT)*y for y in tuple(map(operator.add, psi_minus[i], psi_plus[i]))
                       ) for i in xrange(len(psi_minus))]
   return E

## Function to transform mesh into discontinuous x-points for plotting.
#
#  @param[in] mesh  mesh data
#  @return    array of the discontinuous x-points
#
def makeXPoints(mesh):
   # number of dofs
   n = 2*mesh.n_elems
   # initialize
   x = np.zeros(n)
   # loop over elements
   for i in xrange(mesh.n_elems):
      iL = 2*i
      iR = iL + 1
      x[iL] = mesh.getElement(i).xl # left  end of element
      x[iR] = mesh.getElement(i).xr # right end of element
   # return
   return x

## Function to transform mesh into continuous x-points for plotting.
#
#  @param[in] mesh  mesh data
#  @return    array of the continuous x-points
#
def makeContinuousXPoints(mesh):
   # number of edges
   n = mesh.n_elems + 1
   # initialize
   x = np.zeros(n)
   # loop over elements
   for i in xrange(mesh.n_elems):
      iL = 2*i
      iR = iL + 1
      x[i] = mesh.getElement(i).xl             # left  end of element
   x[n-1] = mesh.getElement(mesh.n_elems-1).xr # right end of element
   # return
   return x

## Function to transform array of tuples into array for plotting.
#
#  @param[in] tuple_sarray  array of tuples
#  @return    array of the discontinuous data
#
def makeYPoints(tuples_array):
   # number of elements
   n_elems = len(tuples_array)
   # number of dofs
   n = 2*n_elems
   # initialize
   y = np.zeros(n)
   # loop over elements
   for i in xrange(n_elems):
      iL = 2*i
      iR = iL + 1
      y[iL] = tuples_array[i][0] # left  value
      y[iR] = tuples_array[i][1] # right value
   # return
   return y

