## @package src.balanceChecker
#  Contains a class to compute balance

import numpy as np
import radUtilities as RU
import globalConstants as GC
from math import sqrt

## Default dictionary to pass to balance checker
#
empty_srcs = {"rad":0.0, "mass":0.0, "erg":0.0, "mom":0.0}


#================================================================================
## Balance checker class
#
#================================================================================
class BalanceChecker:

    #----------------------------------------------------------------------------
    ## Constructor. 
    #
    #  @param[in] mesh          need spatial mesh for volumes
    #  @param[in] time_stepper  not implemented for all methods necessarily
    #  @param[in] dt            assumed constant essentially
    #----------------------------------------------------------------------------
    def __init__(self, mesh, problem_type, timestepper, dt):

        self.mesh = mesh
        self.time_stepper  = timestepper
        self.dt   = dt
        self.prob = problem_type

    #----------------------------------------------------------------------------
    ## Compute balance on a simple steady state problem
    #  Constructor
    #
    def computeSSRadBalance(self, psi_left, psi_right, rad, sigma_a, Q_iso):

        #assume uniform volume
        vol = self.mesh.getElement(0).dx

        #get Cell invegrated quantities
        sources = sum([Q_iso*2*vol for i in rad.phi])
        absor = sum([0.5*(i[0]+i[1])*vol*sigma_a for i in rad.phi])

        #Get currents
        mu = RU.mu["+"]
        j_in = (psi_left*mu + psi_right*mu)
        j_out= rad.psim[0][0]*mu + rad.psip[-1][1]*mu

        bal = j_in - j_out + sources - absor
        
        print "\n====================================================="
        print "Balance Check"
        print "====================================================="
        print "    Absorption Rate:   %.6e" % absor
        print "            Sources:   %.6e" % sources
        print "Current in:            %.6e" % (j_in)
        print "Current out:           %.6e" % (j_out)
        print "-----------------------------------------------------"
        print "    Absolute Balance:  %.6e" % bal
        print "    Relative Balance:  %.6e" % (bal/(sources))
        print "=====================================================\n"

        return 

    #----------------------------------------------------------------------------
    ## Compute balance for a coupled rad-hydro problem
    #
    def computeBalance(self, psi_left, psi_right, hydro_old,
            hydro_new, rad_old, rad_new, hydro_F_left=None, hydro_F_right=None, 
            src_totals={"rad":0.0,"mass":0.0,"erg":0.0,"mom":0.0}, 
            cx_new=None, write=True):

        #assume uniform volume
        vol = self.mesh.getElement(0).dx
        dt  = self.dt

        # compute mass in domain
        mass_new = sum([i.rho*vol for i in hydro_new])
        mass_old = sum([i.rho*vol for i in hydro_old])

        # compute hydro momentum in domain
        mom_new_hydro = sum([vol*(i.rho*i.u) for i in hydro_new])
        mom_old_hydro = sum([vol*(i.rho*i.u) for i in hydro_old])

        # compute radiation momentum in domain
        c = GC.SPD_OF_LGT
        mom_new_rad = sum([0.5*(i[0]+i[1])*vol/c**2 for i in rad_new.F])
        mom_old_rad = sum([0.5*(i[0]+i[1])*vol/c**2 for i in rad_old.F])

        # compute total momentum in domain
        mom_new = mom_new_hydro + mom_new_rad
        mom_old = mom_old_hydro + mom_old_rad

        # compute hydro energy in domain
        erg_new_hydro = sum([i.E()*vol for i in hydro_new])
        erg_old_hydro = sum([i.E()*vol for i in hydro_old])

        # compute radiation energy in domain
        erg_new_rad = sum([0.5*(i[0]+i[1])*vol for i in rad_new.E])
        erg_old_rad = sum([0.5*(i[0]+i[1])*vol for i in rad_old.E])

        # compute total energy in domain
        erg_new = erg_new_hydro + erg_new_rad
        erg_old = erg_old_hydro + erg_old_rad

        # compute internal energy in domain
        em_new = sum([i.e*vol*i.rho for i in hydro_new])
        em_old = sum([i.e*vol*i.rho for i in hydro_old])

        # compute kinetic energy in domain
        KE_new = sum([vol*(0.5*i.rho*i.u**2) for i in hydro_new])
        KE_old = sum([vol*(0.5*i.rho*i.u**2) for i in hydro_old])

        #Compute momentum deposited to material in a rad_mat only problem,
        #This must still be added, hardcoded as BE for now
        mom_deposition = 0.0
        if self.prob == 'rad_mat':

            if self.time_stepper == 'BE':
                for i in xrange(len(rad_new.F)):
                    mom_l = cx_new[i][0].sig_t*rad_new.F[i][0]*vol/c
                    mom_r = cx_new[i][1].sig_t*rad_new.F[i][1]*vol/c
                    mom_deposition += 0.5*(mom_l + mom_r)
            else:

               print "WARNING: Momentum balance in TRT problems only implemented"\
                    " correctly for BE. All others missing deposition term"



        # compute hydro net inflows
        if (self.prob == 'rad_mat'):
           mass_netflow_hydro = 0.0
           mom_netflow_hydro  = -1.*mom_deposition
           erg_netflow_hydro  = 0.0
        elif (self.prob == 'rad_hydro'):
           mass_netflow_hydro = hydro_F_left["rho"] - hydro_F_right["rho"] 
           mom_netflow_hydro  = hydro_F_left["mom"] - hydro_F_right["mom"] 
           erg_netflow_hydro  = hydro_F_left["erg"] - hydro_F_right["erg"] 

        # compute radiation momentum net inflow
        mom_left_new_rad  = (psi_left + rad_new.psim[0][0])/(3.0*c)
        mom_left_old_rad  = (psi_left + rad_old.psim[0][0])/(3.0*c)
        mom_right_new_rad = (rad_new.psip[-1][1] + psi_right)/(3.0*c)
        mom_right_old_rad = (rad_old.psip[-1][1] + psi_right)/(3.0*c)
        mom_netflow_new_rad = mom_left_new_rad - mom_right_new_rad
        mom_netflow_old_rad = mom_left_old_rad - mom_right_old_rad

        # compute radiation energy net inflow
        mu = RU.mu["+"]
        erg_inflow_new_rad  = psi_left*mu + psi_right*mu
        erg_inflow_old_rad  = psi_left*mu + psi_right*mu
        erg_outflow_new_rad = rad_new.psim[0][0]*mu + rad_new.psip[-1][1]*mu
        erg_outflow_old_rad = rad_old.psim[0][0]*mu + rad_old.psip[-1][1]*mu
        erg_netflow_new_rad = erg_inflow_new_rad - erg_outflow_new_rad
        erg_netflow_old_rad = erg_inflow_old_rad - erg_outflow_old_rad

        # compute balance
        if self.time_stepper == 'BE':

           mass_bal = mass_new - mass_old - dt*(mass_netflow_hydro)
           mom_bal  = mom_new  - mom_old  - dt*(mom_netflow_hydro
              + mom_netflow_new_rad) - src_totals["mom"]
           erg_bal  = erg_new  - erg_old  - dt*(erg_netflow_hydro
              + erg_netflow_new_rad) - src_totals["erg"]
           print erg_new


        elif self.time_stepper == 'CN':

           mass_bal = mass_new - mass_old - dt*(mass_netflow_hydro)
           mom_bal  = mom_new  - mom_old  - dt*(mom_netflow_hydro
              + 0.5*mom_netflow_old_rad + 0.5*mom_netflow_new_rad)
           erg_bal  = erg_new  - erg_old  - dt*(erg_netflow_hydro
              + 0.5*erg_netflow_old_rad + 0.5*erg_netflow_new_rad)

        else:

           raise NotImplementedError("Only BE and CN implemented in balance checker")

        #simple balance
        if (write):

            print "\n====================================================="
            print "Balance Check"
            print "====================================================="
            print "New energy radiation:  %.6e" % erg_new_rad
            print "Old energy radiation:  %.6e" % erg_old_rad
            print "Current in:            %.6e" % (dt*erg_inflow_new_rad)
            print "Current out:           %.6e" % (dt*erg_outflow_new_rad)
            print "-----------------------------------------------------"
            print "New energy material:   %.6e" % em_new
            print "Old energy material:   %.6e" % em_old
            print "New kinetic energy:    %.6e" % (KE_new)
            print "Old kinetic energy:    %.6e" % (KE_old) 
            print "mass     flux left:    %.6e" % (hydro_F_left["rho"]*dt) 
            print "momentum flux left:    %.6e" % (hydro_F_left["mom"]*dt) 
            print "energy   flux left:    %.6e" % (hydro_F_left["erg"]*dt) 
            print "mass     flux right:   %.6e" % (hydro_F_right["rho"]*dt)
            print "momentum flux right:   %.6e" % (hydro_F_right["mom"]*dt)
            print "energy   flux right:   %.6e" % (hydro_F_right["erg"]*dt)
            print "-----------------------------------------------------"
            print "Momentum source total: %.6e" % (src_totals["mom"])
            print "Energy   source total: %.6e" % (src_totals["erg"])
            print "-----------------------------------------------------"
            print "    Mass Excess (Relative):  %.6e (%.6e)" % (mass_bal,
                    mass_bal/max(mass_new,1.E-65))
            print "Momentum Excess (Relative):  %.6e (%.6e)" % (mom_bal, 
                    mom_bal/max(abs(mom_new)+abs(src_totals["mom"]),1.E-65))
            print "  Energy Excess (Relative):  %.6e (%.6e)" % (erg_bal,
                    erg_bal/max(erg_new+abs(src_totals["erg"]),1.E-65))
            print "=====================================================\n"

            


