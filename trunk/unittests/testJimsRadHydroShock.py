## @package unittests.testRadHydroShock
#  Contains unittest class to test a radiation-hydrodyamics shock problem.

# add source directory to module search path
import sys
sys.path.append('../src')
import numpy as np

# unit test package
import unittest
import pickle
import re

# local packages
from mesh import Mesh
from hydroState import HydroState
from radiation import Radiation
from plotUtilities import plotHydroSolutions, plotTemperatures, plotS2Erg
from utilityFunctions import computeRadiationVector, computeAnalyticHydroSolution
from crossXInterface import ConstantCrossSection
from transient import runNonlinearTransient
from hydroBC import HydroBC
from radBC   import RadBC
import globalConstants as GC
import cProfile
import profile

## Derived unittest class to test the radiation-hydrodynamics shock problem
#
class TestRadHydroShock(unittest.TestCase):
   def setUp(self):
      pass
   def tearDown(self):
      pass
   def test_RadHydroShock(self):

      # create uniform mesh
      n_elems = 1000
      width = 0.04
      x_start = -0.02
      mesh_center = x_start + 0.5*width
      mesh = Mesh(n_elems, width, x_start=x_start)

      # slope limiter
      slope_limiter = "double-minmod"

      # gamma constant
      gam = 5.0/3.0
 
      # material 1 and 2 properties;
      T_ref = 0.1 #keV reference unshocked, upstream, ambient, equilibrium
      
      sig_a1 = 577.35
      sig_s1 = 0.0
      c_v1   = 0.14472799784454
      sig_a2 = sig_a1
      sig_s2 = sig_s1
      c_v2   = c_v1

      #Read in Jim's nondimensional results to set preshock and postshock dimensional
      mach_number = "5.0" #Choices are 2.0, 1.2, 3.0, 5.0
      filename = 'analytic_shock_solutions/data_for_M%s.pickle' % mach_number
      f = open(filename,'r')
      data = pickle.load(f)
      f.close()

      #compute scalings based on an assumed density and reference temperature
      dp =getDimensParams(T_ref=T_ref, rho_ref=1.0, C_v=c_v1, gamma=gam)
      
      print data

      #Scale non-dimensional values into dimensional results
      rho1   = data['Density'][0]*dp['rho']
      u1     = data['Speed'][0]*dp['a']   #velocity times reference sound speed
      Erad1  = data['Er'][0]*dp['Er']
      T1     = data['Tm'][0]*T_ref
      e1     = T1*c_v1
      E1     = rho1*(e1 + 0.5*u1*u1)

      # material 2 IC
      rho2   = data['Density'][-1]*dp['rho']
      u2     = data['Speed'][-1]*dp['a']   #velocity times reference sound speed
      Erad2  = data['Er'][-1]*dp['Er']
      T2     = data['Tm'][-1]*T_ref
      e2     = T2*c_v2
      E2     = rho2*(e2 + 0.5*u2*u2)
 
      print "rho", rho1, rho2
      print "vel", u1, u2
      print "Temperature", T1, T2
      print "momentum", rho1*u1, rho2*u2
      print "E",E1, E2
      print "E_r",Erad1, Erad2
      print sig_a1, sig_s1
      # material 1 IC
 
      # final time
      t_end = 0.01
 
      # temperature plot filename
      test_filename = "radshock_mach_"+re.search("M(\d\.\d)",filename).group(1)+".pdf"
 
      # compute radiation BC; assumes BC is independent of time
      c = GC.SPD_OF_LGT
      psi_left  = 0.5*c*Erad1  
      psi_right = 0.5*c*Erad2

      #Create BC object
      rad_BC = RadBC(mesh, "dirichlet", psi_left=psi_left, psi_right=psi_right)
      
      # construct cross sections and hydro IC
      cross_sects = list()
      hydro_IC = list()
      psi_IC = list()

      for i in range(mesh.n_elems):  
      
         if mesh.getElement(i).x_cent < mesh_center: # material 1
            cross_sects.append( (ConstantCrossSection(sig_s1, sig_s1+sig_a1),
                                 ConstantCrossSection(sig_s1, sig_s1+sig_a1)) )
            hydro_IC.append(
               HydroState(u=u1,rho=rho1,e=e1,spec_heat=c_v1,gamma=gam))

            psi_IC += [psi_left for dof in range(4)]

         else: # material 2
            cross_sects.append((ConstantCrossSection(sig_s2, sig_a2+sig_s2),
                                ConstantCrossSection(sig_s2, sig_a2+sig_s2)))
            hydro_IC.append(
               HydroState(u=u2,rho=rho2,e=e2,spec_heat=c_v2,gamma=gam))

            psi_IC += [psi_right for dof in range(4)]

      #Convert pickle data to dimensional form
      data['Density'] *= dp['rho']
      data['Speed']   *= dp['a']
      data['Er']      *= dp['Er']
      data['Tm']      *= dp['Tm']
      x_anal = data['x']

      #If desired initialize the solutions to analytic result (ish)
      analytic_IC = True
      if analytic_IC:

         hydro_IC = list()
         psi_IC = list()

         for i in range(mesh.n_elems):

            #Determine which analytic x is closest to cell center
            xcent = mesh.getElement(i).x_cent
            min_dist = 9001. 
            idx = -1
            for j in range(len(x_anal)):
               if abs(xcent - x_anal[j]) < min_dist:
                  min_dist = abs(xcent - x_anal[j])
                  idx = j

            #Create state based on these values, noting that the pickle has already
            #been dimensionalized
            rho1   = data['Density'][idx]
            u1     = data['Speed'][idx]   #velocity times reference sound speed
            Erad1  = data['Er'][idx]
            T1     = data['Tm'][idx]
            e1     = T1*c_v1
            E1     = rho1*(e1 + 0.5*u1*u1)
            psi_left =  0.5*c*Erad1

            hydro_IC.append(
               HydroState(u=u1,rho=rho1,e=e1,spec_heat=c_v1,gamma=gam))

            psi_IC += [psi_left for dof in range(4)]
   
      #Smooth out the middle solution optionally, shouldnt need this
      n_smoothed = 0
      state_l = hydro_IC[0]
      state_r = hydro_IC[-1]
        
      rho_l =  state_l.rho
      rho_r =  state_r.rho
      drho  = rho_r - rho_l
      u_l   =  state_l.u
      u_r   =  state_r.u
      du    = u_r - u_l
      e_l   =  state_l.e
      e_r   =  state_r.e
      de    = e_r-e_l
      print "p", state_l.p, state_r.p
      print "mach number", state_l.u/state_l.getSoundSpeed(), state_r.u/state_r.getSoundSpeed()

      #Scale
      idx = 0
      if n_smoothed > 0:
          for i in range(mesh.n_elems/2-n_smoothed/2-1,mesh.n_elems/2+n_smoothed/2):

              rho = rho_l + drho*idx/n_smoothed
              u   = rho_l*u_l/rho
              e   = e_l + de*idx/n_smoothed

              idx+=1

              E   = 0.5*rho*u*u + rho*e
              hydro_IC[i].updateState(rho, rho*u, E)

      # plot hydro initial conditions
      plotHydroSolutions(mesh, hydro_IC)

      rad_IC = Radiation(psi_IC)

      # create hydro BC
      hydro_BC = HydroBC(bc_type='fixed', mesh=mesh, state_L = state_l,
            state_R = state_r)
      #Forcing to reflective? Maybe this is the problem
      #hydro_BC = HydroBC(mesh=mesh,bc_type='reflective')

      # transient options
      t_start  = 0.0

      # if run standalone, then be verbose
      if __name__ == '__main__':
         verbosity = 2
      else:
         verbosity = 0
      
      # run the rad-hydro transient
      rad_new, hydro_new = runNonlinearTransient(
         mesh         = mesh,
         problem_type = 'rad_hydro',
         dt_option    = 'CFL',
         CFL          = 0.6,
         use_2_cycles = True,
         t_start      = t_start,
         t_end        = t_end,
         rad_BC       = rad_BC,
         cross_sects  = cross_sects,
         rad_IC       = rad_IC,
         hydro_IC     = hydro_IC,
         hydro_BC     = hydro_BC,
         verbosity    = verbosity,
         slope_limiter = slope_limiter,
         check_balance=False)

      # plot
      if __name__ == '__main__':

         # compute exact hydro solution
         hydro_exact = None

         # plot hydro solution
         plotHydroSolutions(mesh, hydro_new, exact=hydro_exact, pickle_dic=data)

         # plot material and radiation temperatures
         Tr_exact, Tm_exact = plotTemperatures(mesh, rad_new.E, hydro_states=hydro_new, print_values=True,
            save=True, filename=test_filename,
             pickle_dic=data)

         # plot angular fluxes
         plotS2Erg(mesh, rad_new.psim, rad_new.psip)

         #Make a pickle to save the error tables
         pickname = "results/testJimsShock_M%s_%ielems.pickle" % (mach_number,n_elems)

         #Create dictionary of all the data
         big_dic = { }
         big_dic["hydro"] =  hydro_new
         big_dic["hydro_exact"] = hydro_exact
         big_dic["rad"] = rad_new
         big_dic["Tr_exact"] = Tr_exact
         big_dic["Tm_exact"] = Tm_exact
         pickle.dump( big_dic, open( pickname, "w") )

#-------------------------------------------------------------------------------------------
#
# Takes reference temperature and other parameters, assuming ideal gas, and computes reference
# parameters that can be used to convert the non-dimensional parameters into the
# appropriate values, using an ideal gas equation of state, with a defined C_v, 
# reference temperature and density
#
#  input : nd (nondimensional paramters dictionary)
def getDimensParams(T_ref=None, rho_ref=1.0, C_v=None, gamma=None):

   rho_0 = rho_ref
   T_0 = T_ref

   #Compute the reference a_0 (sound speed), and p_0
   e_0 = C_v*T_0
   a_0 = np.sqrt(e_0*gamma*(gamma-1.))
   p_0 = rho_ref*e_0*(gamma - 1.) 

   dp = {}
   dp['a'] = a_0
   dp['Tm'] = T_0
   dp['rho'] = rho_ref
   dp['Er']  = GC.RAD_CONSTANT * T_0**4.

   return dp

   




   

   
   

   

    



# run main function from unittest module
if __name__ == '__main__':
   unittest.main()

