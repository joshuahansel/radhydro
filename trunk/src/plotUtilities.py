## @package src.plotUtilities
#  Provides functions for plotting radiation and hydrodynamics solutions.

import matplotlib.pyplot as plt
import numpy as np
from numpy import array, genfromtxt
import globalConstants as GC
from matplotlib import rc # for rendering tex in plots
from radUtilities import computeScalarFlux
from hydroState import computeVelocity, computeIntEnergy, computePressure
from utilityFunctions import computeHydroInternalEnergies

## Plots a function of (x,t)
#
def plotFunction(f, x, t, legend_label='$f(x,t)$', y_label='$f(x,t)$'):

   plt.figure()
   y = f(x,t)
   plt.plot(x, y, label=legend_label)
   plt.xlabel('$x$')
   plt.ylabel(y_label)
   plt.legend(loc='best')
   plt.show()


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


## Plot the angular flux and exact angular flux
def plotS2Erg(mesh, psim_edge, psip_edge, exact_psim=None, save=False, filename='Radiation.pdf',
        exact_psip=None, edge_values=True, print_values=False):

   # create new figure
   plt.figure()

   # create x-points
   x = mesh.getCellCenters()

   # transform array of tuples into array
   psip = computeAverageValues(psip_edge)
   psim = computeAverageValues(psim_edge)

   x_e = mesh.getCellEdgesDiscontinuous()

   # transform into a list of
   psip_e = computeLRValues(psip_edge)
   psim_e = computeLRValues(psim_edge)

   # plot
   plt.rc('text', usetex=True)         # use tex to generate text
   plt.rc('font', family='sans-serif') # use sans-serif font family
   plt.plot(x, psim, 'r+', label='Numerical $\psi^-$ Avg')
   plt.plot(x, psip, 'b+', label='Numerical $\psi^+$ Avg')
   plt.plot(x_e, psim_e, 'r-', label='Numerical $\psi^-$ LD')
   plt.plot(x_e, psip_e, 'b-', label='Numerical $\psi^+$ LD')

   # annotations
   plt.xlabel('$x$')
   plt.ylabel('$\psi$')

   #plot the exact 
   if exact_psip != None:

       plt.plot(x, exact_psip, "*r", label='Exact $\psi^+$')

   if exact_psim != None:

       plt.plot(x, exact_psim, "*b", label='Exact $\psi^-$')

   plt.legend(loc='best')

   # if print requested
 #  if print_values:
 #     print "  x   E_r    E_r_exact  "
 #     print "-------------------"
 #     for i in range(len(x)):
##
 #         print "%.12f" % x[i], "%.12f" % Er[i], "%.12f" % T[i]


   

   # save if requested
   if save:
      plt.savefig(filename)
   else:
      plt.show()

def plotIntErgs(mesh, e_rad_new, states_new, slopes, save=False, print_values=False):

   # create new figure
   plt.figure()

   # create x-points
   x = mesh.getCellCenters()

   # transform array of tuples into array
   e_rad_avg = computeAverageValues(e_rad_new)

   x_e = mesh.getCellEdgesDiscontinuous()

   e_e = [computeHydroInternalEnergies(i,states_new[i],slopes) for i in
         range(len(states_new))]

   e = [i.e for i in states_new]

   # transform into a list of
   e_rad_e = computeLRValues(e_rad_new)
   e_e = computeLRValues(e_e)

   # plot
   plt.rc('text', usetex=True)         # use tex to generate text
   plt.rc('font', family='sans-serif') # use sans-serif font family
   plt.plot(x, e, 'bx', label='Hydro $e$ avg')
   plt.plot(x_e,e_e,'b--',label='Hydro $e$ edge')
   plt.plot(x, e_rad_avg, 'r+', label='Radiation Avg. $e$')
   plt.plot(x_e, e_rad_e, 'r-', label='Radiation Edge $e$')

   # annotations
   plt.xlabel('$x$')
   plt.ylabel('$e$')

   plt.legend(loc='best')

   # if print requested
 #  if print_values:
 #     print "  x   E_r    E_r_exact  "
 #     print "-------------------"
 #     for i in range(len(x)):
##
 #         print "%.12f" % x[i], "%.12f" % Er[i], "%.12f" % T[i]


   

   # save if requested
   if save:
      plt.savefig(filename)
   else:
      plt.show()

## Plot arbitrary radiation density
#
def plotRadErg(mesh, Er_edge, Fr_edge=None, exact_Fr=None, save=False, 
   filename='Radiation.pdf', exact_Er=None, print_values=False):

   # create new figure
   plt.figure()
   plt.subplot(2,1,1)

   # create x-points
   x = mesh.getCellCenters()

   # transform array of tuples into array
   Er = computeAverageValues(Er_edge)

   # plot
   plt.rc('text', usetex=True)         # use tex to generate text
   plt.rc('font', family='sans-serif') # use sans-serif font family
   plt.plot(x, Er, 'r-', label='Numerical $E_r$')

   # annotations
   plt.xlabel('$x$')
   plt.ylabel('$E_r$')

   # plot exact radiation energy
   if exact_Er != None:
       plt.plot(x, exact_Er, "*--b", label='Exact $E_r$')

   plt.legend(loc='best')

   # plot radiation flux it is supplied
   if Fr_edge != None:
   
      # split plot into 2
      plt.subplot(2,1,2)

      # compute cell average radiation flux
      Fr = computeAverageValues(Fr_edge)

      # divide by speed of light for comparison to E magnitude
      for i in range(len(Fr)):
         Fr[i] /= GC.SPD_OF_LGT

      # plot
      plt.rc('text', usetex=True)         # use tex to generate text
      plt.rc('font', family='sans-serif') # use sans-serif font family
      plt.plot(x, Fr, 'r-', label='Numerical $F_r/c$')

      # annotations
      plt.xlabel('$x$')
      plt.ylabel('$F_r/c$')

      # plot exact radiation flux
      if exact_Fr != None:

         # divide by speed of light for comparison to E magnitude
         for i in range(len(Fr)):
            exact_Fr[i] /= GC.SPD_OF_LGT

         # plot
         plt.plot(x, exact_Fr, "*--b", label='Exact $F_r/c$')

   # legend
   plt.legend(loc='best')

   # if print requested
   if print_values:
      print "  x   E_r  "
      print "-------------------"
      for i in range(len(x)):
         print "%.12f" % x[i], "%.12f" % Er[i]

   # save if requested
   if save:
      plt.savefig(filename)
   else:
      plt.show()

## Plots hydrodynamic and radiation temperatures
#
def plotTemperatures(mesh, Er_edge, save=False, filename='Temperatures.pdf',
        hydro_states=None, exact_Er=None, print_values=False,
        exact_solution_filename=None):

   # create new figure
   plt.figure()

   # create x-points
   x = mesh.getCellCenters()

   # transform array of tuples into array
   Er = computeAverageValues(Er_edge)
   a = GC.RAD_CONSTANT
   Tr = [pow(i/a, 0.25) for i in Er]

   # tex and fonts
   plt.rc('text', usetex=True)         # use tex to generate text
   plt.rc('font', family='sans-serif') # use sans-serif font family

   # plot material temperature
   if hydro_states != None:
       T = [state.getTemperature() for state in hydro_states]
       plt.plot(x,T,'bx', label='$T_m$')
    
   # plot radiation temperature
   plt.plot(x, Tr, 'r+', label='$T_r$')

   # plot the exact temperatures
   if exact_solution_filename != None:
      # read data from csv file
      exact_data = genfromtxt(exact_solution_filename)
      # get x points, 1st column
      x_exact = exact_data[:,0]
      # get exact material temperature, 2nd column
      T_exact = exact_data[:,1]
      # get exact radiation temperature, 3rd column
      Tr_exact = exact_data[:,2]
      # plot exact temperatures
      print T_exact, Tr_exact
      plt.plot(x_exact,T_exact,'b-',label='$T_m$, analytic')
      plt.plot(x_exact,Tr_exact,'r-',label='$T_r$, analytic')

   # annotations
   plt.xlabel('$x$')
   plt.ylabel('$T$ (keV)')
   plt.legend(loc='best')

   # strongly set x-axis limits, keep current y-axis limits
   x_min = mesh.elements[0].xl
   x_max = mesh.elements[-1].xr
   xaxis_min, xaxis_max, yaxis_min, yaxis_max = plt.axis()
   plt.axis((x_min,x_max,yaxis_min,yaxis_max))

   # if print requested
   file_name = "T_results_"+str(mesh.n_elems)+".txt"
   f = open(file_name, "w")
   if print_values:
      print "  x   T_r    T_m   "
      f.write("  x   T_r    T_m   \n")
      print "-------------------"
      f.write("-------------------\n")
      for i in range(len(x)):

          print "%.12f" % x[i], "%.12f" % Tr[i], "%.12f" % T[i]
          f.write("%.12e %.12e %.12e\n" % (x[i],Tr[i],T[i]))

   # save if requested
   if save:
      plt.savefig(filename)
   else:
      plt.show()

   
## Function to transform mesh into discontinuous x-points for plotting.
#
#  @param[in] mesh  mesh data
#
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
#
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
#  @param[in] tuples_sarray  array of tuples
#
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


## Computes the average values for a list of 2-tuples
#
def computeAverageValues(tuple_list):
   return [0.5*i[0] + 0.5*i[1] for i in tuple_list]

## Make a plotting array from L, R values
#
def computeLRValues(tuple_list):
   x = []
   for i in tuple_list:
      x.append(i[0])
      x.append(i[1])

   return x

## Plots hydro solution
#
def plotHydroSolutions(mesh, states, slopes=None, x_exact=None, exact=None,
    save_plot=False, filename='hydro_solution.pdf'):

    # create 11" x 8.5" figure
    plt.figure(figsize=(11,8.5))

    # get cell centers
    x_cent = mesh.getCellCenters()
    if slopes == None:
       x_num = x_cent
    else:
       x_num = mesh.getCellEdgesDiscontinuous()

    # create lists for each quantity to be plotted
    n = mesh.n_elems
    if slopes == None:
       u   = np.zeros(n)
       p   = np.zeros(n)
       rho = np.zeros(n)
       e   = np.zeros(n)
       E   = np.zeros(n)
       mom = np.zeros(n)
       for i in xrange(n):
          u[i]   = states[i].u
          p[i]   = states[i].p
          rho[i] = states[i].rho
          e[i]   = states[i].e
          E[i]   = states[i].E()
          mom[i] = u[i]*rho[i]
    else:
       u   = np.zeros(2*n)
       p   = np.zeros(2*n)
       rho = np.zeros(2*n)
       e   = np.zeros(2*n)
       E   = np.zeros(2*n)
       mom = np.zeros(2*n)
       rho_l, rho_r, mom_l, mom_r, erg_l, erg_r =\
          slopes.createLinearRepresentation(states)
       gam = states[0].gamma
       for i in xrange(n):
          rho[2*i]   = rho_l[i]
          rho[2*i+1] = rho_r[i]
          u[2*i]     = computeVelocity(rho=rho_l[i], mom=mom_l[i])
          u[2*i+1]   = computeVelocity(rho=rho_r[i], mom=mom_r[i])
          p[2*i]     = computePressure(rho=rho_l[i], mom=mom_l[i], erg=erg_l[i], gam=gam)
          p[2*i+1]   = computePressure(rho=rho_r[i], mom=mom_r[i], erg=erg_r[i], gam=gam)
          e[2*i]     = computeIntEnergy(rho=rho_l[i], mom=mom_l[i], erg=erg_l[i])
          e[2*i+1]   = computeIntEnergy(rho=rho_r[i], mom=mom_r[i], erg=erg_r[i])
          E[2*i]     = erg_l[i]
          E[2*i+1]   = erg_r[i]
          mom[2*i]   = u[2*i]*rho[2*i]
          mom[2*i+1] = u[2*i+1]*rho[2*i+1]
 
    # create lists for each exact quantity to be plotted
    if exact == None:
       u_exact = None
       p_exact = None
       rho_exact = None
       e_exact = None
       E_exact = None
       mom_exact = None
    else:
       u_exact   = list()
       p_exact   = list()
       rho_exact = list() 
       e_exact   = list()
       E_exact   = list()
       mom_exact = list()
       for i in exact:
           u_exact.append(i.u)
           p_exact.append(i.p)
           rho_exact.append(i.rho)
           e_exact.append(i.e)
           E_exact.append(i.e*i.rho + 0.5*i.u*i.u*i.rho)
           mom_exact.append(i.u*i.rho)

    # plot each quantity
    plotSingle(x_num=x_num, x_exact=x_exact, y=rho, y_label=r"$\rho$", exact=rho_exact) 
    plotSingle(x_num=x_num, x_exact=x_exact, y=p,   y_label=r"$p$",    exact=p_exact)
    plotSingle(x_num=x_num, x_exact=x_exact, y=mom, y_label=r"$(\rho u)$", exact=mom_exact)
    plotSingle(x_num=x_num, x_exact=x_exact, y=u,   y_label=r"$u$",    exact=u_exact)
    plotSingle(x_num=x_num, x_exact=x_exact, y=E,   y_label=r"$E$",    exact=E_exact)
    plotSingle(x_num=x_num, x_exact=x_exact, y=e,   y_label=r"$e$",    exact=e_exact)


    # save figure
    if save_plot:
       plt.savefig(filename)

    # show figure
    #plt.show(block=False) #show all plots generated to this point
    #raw_input("Press anything to continue...")
    plt.show()
    plotSingle.fig_num=0


## Plots a single plot in a 4x4 subplot array
#
def plotSingle(x_num, x_exact, y, y_label, exact=None):

    # static variable counter
    plotSingle.fig_num += 1
    
    plt.figsize=(15,10)

    plt.subplot(3,2,plotSingle.fig_num)

    plt.xlabel('$x$')
    plt.ylabel(y_label)

    # min and max y for plot
    plot_min = min(y)
    plot_max = max(y)

    plt.plot(x_num, y, "b+-", label="Numeric")
    if exact != None:
       plt.plot(x_exact, exact, "r-x", label="Analytic")
       plot_min = min(plot_min, min(exact))
       plot_max = max(plot_max, max(exact))

    plt.legend(loc='best')

    # if solution profile is approximately constant, then adjust axes
    den = plot_max
    if den == 0.0:
       den = 1.0
    if abs((plot_max - plot_min)/den) < 1.0e-3:
       plt.axis([min(x_num), max(x_num), 0.99*plot_min, 1.01*plot_max])
    
plotSingle.fig_num=0


