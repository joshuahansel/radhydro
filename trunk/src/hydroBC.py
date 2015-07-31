## @package Contains class for handling hydrodynamics boundary conditions.

from copy import deepcopy
from hydroState import HydroState

## Handles hydrodynamics boundary conditions.
#
class HydroBC(object):

   ## Constructor.
   #
   #  @param[in] bc_type  string identifier for boundary condition type, e.g.,
   #                      'reflective' or 'dirichlet'
   #  @param[in] mesh     mesh object
   #  @param[in] rho_BC   function handle for density as a function of (x,t),
   #                      necessary if Dirichlet BC are chosen
   #  @param[in] mom_BC   function handle for momentum as a function of (x,t),
   #                      necessary if Dirichlet BC are chosen
   #  @param[in] erg_BC   function handle for total energy as a function of (x,t),
   #                      necessary if Dirichlet BC are chosen
   #
   def __init__(self, bc_type, mesh, rho_BC=None, mom_BC=None, erg_BC=None):

      # save the BC type
      self.bc_type = bc_type

      # check the provided hydro BC type
      if self.bc_type == 'reflective':
         pass
      elif self.bc_type == 'dirichlet':
         assert rho_BC != None, "density BC function must be provided"
         assert mom_BC != None, "momentum BC function must be provided"
         assert erg_BC != None, "energy BC function must be provided"
         self.rho_BC = rho_BC
         self.mom_BC = mom_BC
         self.erg_BC = erg_BC
      else:
         raise NotImplementedError("Invalid hydro BC type")

      # save number of elements, for indexing
      self.n = mesh.n_elems

      # compute the cell center positions of the left and right ghost
      # boundary cells in case Dirichlet BC are used
      cell_L = mesh.getElement(0)
      cell_R = mesh.getElement(mesh.n_elems-1)
      self.x_L = cell_L.x_cent - cell_L.dx
      self.x_R = cell_R.x_cent + cell_R.dx
      self.dx_L = cell_L.dx
      self.dx_R = cell_R.dx

   ## Updates the boundary values for each conservative variable.
   #
   #  @param[in] states  hydro states for each cell
   #  @param[in] t       time at which to evaluate time-dependent BC
   #
   def update(self, states, t, edge_value=False):

      if self.bc_type == 'reflective':

         # for reflective, gradient is zero at boundaries
         state_L = states[0]
         state_R = states[self.n-1]

         # get conservative variables from states
         self.rho_L, self.mom_L, self.erg_L = state_L.getConservativeVariables()
         self.rho_R, self.mom_R, self.erg_R = state_R.getConservativeVariables()

      elif self.bc_type == 'dirichlet':

         # If updating for use in slopes different than if for use in Riemman solve
         if edge_value:

             # compute conservative variables on edge of boundary to pass to rieman
             # solver
             self.rho_L = self.rho_BC(self.x_L+0.5*self.dx_L, t)
             self.rho_R = self.rho_BC(self.x_R-0.5*self.dx_R, t)
             self.mom_L = self.mom_BC(self.x_L+0.5*self.dx_L, t)
             self.mom_R = self.mom_BC(self.x_R-0.5*self.dx_R, t)
             self.erg_L = self.erg_BC(self.x_L+0.5*self.dx_L, t)
             self.erg_R = self.erg_BC(self.x_R-0.5*self.dx_R, t)

         else: #cell center value

             # compute conservative variables on ghost boundary cells
             self.rho_L = self.rho_BC(self.x_L, t)
             self.rho_R = self.rho_BC(self.x_R, t)
             self.mom_L = self.mom_BC(self.x_L, t)
             self.mom_R = self.mom_BC(self.x_R, t)
             self.erg_L = self.erg_BC(self.x_L, t)
             self.erg_R = self.erg_BC(self.x_R, t)

      else:

         raise NotImplementedError("Invalid hydro BC type")

      # update boundary states
      self.state_L = deepcopy(states[0])
      self.state_R = deepcopy(states[0])
      self.state_L.updateState(rho=self.rho_L, mom=self.mom_L, erg=self.erg_L)
      self.state_R.updateState(rho=self.rho_R, mom=self.mom_R, erg=self.erg_R)

   ## Returns the left and right boundary values for each conservative variable.
   #
   #  @return left and right boundary values for each conservative variable:
   #     -# \f$\rho_0\f$
   #     -# \f$\rho_{N+1}\f$
   #     -# \f$(\rho u)_0\f$
   #     -# \f$(\rho u)_{N+1}\f$i
   #     -# \f$E_0\f$
   #     -# \f$E_{N+1}\f$
   #
   def getBoundaryValues(self):
      return self.rho_L, self.rho_R, self.mom_L, self.mom_R, self.erg_L, self.erg_R


   ## Returns the left and right boundary states
   #
   #  @return left and right boundary states:
   #     -# left state
   #     -# right state
   #
   def getBoundaryStates(self):
      return self.state_L, self.state_R

