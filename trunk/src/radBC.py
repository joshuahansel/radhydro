## @package src.radBC
#  Contains class for handling radiation boundary conditions of different types

from copy import deepcopy
from hydroState import HydroState

## Handles radiation boundary conditions
#
class RadBC(object):

   ## Constructor.
   #
   #  @param[in] bc_type  string identifier for boundary condition type, e.g.,
   #                      'periodic', 'vacuum', or 'dirichlet'. Dirichlet indicates exact
   #                      incident radiation flux passed in that is constant in time,
   #                      or a function if an MMS type BC
   #  @param[in] mesh     mesh object
   #  @param[in] psi_left  Optional exact incident flux in positive direction at left edge, default zero
   #  @param[in] psi_right Optional exact indicent flux in negative direction at right edge, default zero
   #  @param[in] psip_BC  For MMS Dirichlet type boundary conditions a function handle for psi^+ as a function of (x,t)
   #  @param[in] psim_BC  For MMS Dirichelt type boundary conditions a function handle for psi^- as a function of (x,t)
   def __init__(self, mesh, bc_type, psi_left=0.0, psi_right=0.0, psip_BC=None,
           psim_BC=None):

      # save the BC type
      self.bc_type = bc_type
      self.has_mms_func = False #whether or not psip_BC and psim_BC are used

      # check the provided rad BC type
      if self.bc_type == 'vacuum':

         self.psi_left = 0.0 
         self.psi_right = 0.0 

      #Set values to None so they are not accidentally used
      elif self.bc_type == 'periodic':

         self.psi_left = None
         self.psi_right = None

      elif self.bc_type == 'dirichlet':

         assert not (psip_BC != None and psi_left != None), "Must provide a function or value"
         assert not (psim_BC != None and psi_right != None), "Must provide a function or value"
         assert (psip_BC == None or psi_left == None), "Cannot specify function and value"
         assert (psim_BC == None or psi_right == None), "Cannot specify function and value"
         assert not (psim_BC != None and psip_BC == None), "Cannot mix value and function"
         assert not (psip_BC != None and psim_BC == None), "Cannot mix value and function"

         #value specified
         if psi_left != None:

            self.psi_left = psi_left
            self.psi_right = psi_right
         
         #else function specified
         else:

            self.has_mms_func = True
            self.psi_left_BC = psip_BC
            self.psi_right_BC = psim_BC

      else:
         raise NotImplementedError("Invalid hydro BC type")

      # compute the edge of boundary
      # boundary cells in case Dirichlet BC are used
      self.x_L = mesh.getElement(0).xl
      self.x_R = mesh.getElement(mesh.n_elems-1).xr

      #Boundary condition class keeps copy of old values
      if self.bc_type == 'dirichlet' and self.has_mms_func:

         self.psi_left_old = None
         self.psi_right_old = None
         self.psi_left_older = None
         self.psi_right_older = None

      else: #all other boundary conditions dont need these values changed

         self.psi_left_old = psi_left
         self.psi_right_old = psi_right
         self.psi_left_older = psi_left
         self.psi_right_older = psi_right
         
   ## Updates the boundary value for a MMS type boundary condition
   #
   #  @param[in] t       time at which to evaluate time-dependent BC
   #
   def update(self, t=None):

      if self.bc_type == 'dirichlet' and self.has_mms_func:

         print "COPYING BC USING TIME STEPS, THIS IS PROBABLY WRONG FOR GENERAL ALGORITHM"
         self.psi_left_older = self.psi_left_old
         self.psi_left_old   = self.psi_left
         self.psi_left = self.psi_left_BC(self.x_L, t)

         self.psi_right_older = self.psi_right_old
         self.psi_right_old   = self.psi_right
         self.psi_right = self.psi_right_BC(self.x_R,t)
         return

      else:
         return


   ## Returns the left and right boundary values for psi
   #
   #  @return left and right boundary values for psi
   #     -# \f$\psi^+(x_L)\f$
   #     -# \f$\psi^-(x_R)\f$
   #
   def getIncidentFluxes(self):

      return self.psi_left, self.psi_right


   ## Get old incident fluxes for psi
   #
   def getOldIncidentFluxes(self):

      return self.psi_left_old, self.psi_right_old

   ## Get older incident fluxes 
   #
   def getOlderIncidentFluxes(self):

      return self.psi_left_older, self.psi_right_older
