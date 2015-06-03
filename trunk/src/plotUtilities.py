## @package src.plotUtilities
#  Provides functions for plotting radiation and hydrodynamics solutions.

import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import globalConstants as GC
from matplotlib import rc # for rendering tex in plots
from radUtilities import computeScalarFlux

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
#  @param[in] filename  name of pdf file if plot is saved
#  @param[in] psi_minus_exact  exact angular flux solution for minus direction,
#                              passed as an array of values at each edge.
#  @param[in] psi_plus_exact   exact angular flux solution for minus direction,
#                              passed as an array of values at each edge.
#
def plotAngularFlux(mesh, psi_minus, psi_plus, save=False,
   filename='angularFlux.pdf', psi_minus_exact=None, psi_plus_exact=None):

   # create new figure
   plt.figure()

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
      plt.savefig(filename)
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
#  @param[in] filename  name of pdf file if plot is saved
#  @param[in] scalar_flux_exact  exact scalar flux solution, passed as an
#                                array of values at each edge.
#  @param[in] exact_data_continuous  boolean flag that specifies if the provided
#                                    exact solution data is continuous (True)
#                                    or is given as discontinuous tuples (False).
#
def plotScalarFlux(mesh, psi_minus, psi_plus, save=False, filename='scalarFlux.pdf',
   scalar_flux_exact=None, exact_data_continuous=True):

   # create new figure
   plt.figure()

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
      plt.savefig(filename)
   else:
      plt.show()

def plotTemperatures(mesh, E, save=False, filename='Temperatures.pdf',
        hydro_states=None, print_values=True):

   # create new figure
   plt.figure()

   # create x-points
   x            = makeXPoints(mesh)           # discontinuous x-points
   x_continuous = makeContinuousXPoints(mesh) # continuous    x-points

   # transform array of tuples into array
   rad_T_array  = makeYPoints(E)
   a = GC.RAD_CONSTANT

   rad_T_array = [pow(i/a, 0.25) for i in rad_T_array]

   # plot
   plt.rc('text', usetex=True)         # use tex to generate text
   plt.rc('font', family='sans-serif') # use sans-serif font family
   plt.plot(x, rad_T_array, 'r-', label='$T_r$')

   #if necessary get temperature and plot it
   if hydro_states != None:

       T = [(i[0].getTemperature(), i[1].getTemperature()) for i in hydro_states]
       T = makeYPoints(T)
       plt.plot(x,T,'b-', label='$T_m$')
    
   # annotations
   plt.xlabel('$x$')
   plt.ylabel('$T$ (keV)')
   plt.legend(loc='best')

   #if print requested
   if print_values:
      print "  x   T_r    T_m   "
      print "-------------------"
      for i in range(len(x)):

          print "%.12f" % x[i], "%.12f" % rad_T_array[i], "%.12f" % T[i]



   # save if requested
   if save:
      plt.savefig(filename)
   else:
      plt.show()

   

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

## Useful debugging function that just prints any tuple list
def printTupled(tuple_list):

    for i in tuple_list:

        for j in i:

            print j
