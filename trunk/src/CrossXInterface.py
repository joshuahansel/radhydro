## @package CrossXInterface
#  contains classes for cross section interface

#================================================================================
"""
Class that handles the cross sections at a particular point in space.  If we want to
make temperature dependent cross sections later, we will make a class that derives
from this class. The derived class will have a function to update the need to change this later.

All Cross sections have units of cm^{-1}.  The basic cross sections included by this
class is \sigma_t, \sigma_s, and \sigma_a, where \sigma_t = \sigma_s+\sigma_a

"""
class CrossXInterface(object):

    """Must fully specify the cross section. It is implied total is sig_a + sig_s."""
    #----------------------------------------------------------------------------
    def __init__(self, sigma_a, sigma_s):

        self.sig_s = sigma_s
        self.sig_a = sigma_a
        self.sig_t = sigma_a+sigma_s
               
    #----------------------------------------------------------------------------
    ## Define how to print this class
    def __str__(self):

        ## Define what happens when 'print <Mesh_object>' is called
        print_str = "\sigma_s : %.4f, \sigma_a : %.4f, \sigma_t : %.4f" % \
                    (self.sig_s, self.sig_a, self.sig_t) 

        return print_str

    #----------------------------------------------------------------------------
    def updateCrossX(self,*args,**kwargs):
        #default behavior is to not update cross sections. If it is necessary to
        #update cross sections, this will be done by a derived class

        return

#===================================================================================
"""
This is an example of a derived cross section.  In this case it is a simple InvCubed
relation where you need to pass in a new density and temperature to update the cross
section. Anything that uses the base class should be able to interact with base class
objects

This crossX has form

sigma_s = micro_sig_s*rho
sigma_a = coeff*rho*T^{-3}
"""
class InvCubedCrossX(CrossXInterface):

    #----------------------------------------------------------------------------
    def __init__(self, sigma_s_micro, rho, temp, scale_coeff=1.0):

        self.coeff = scale_coeff    
        sig_s = sigma_s_micro*rho   ## sig_s is fixed

        ## Must call base class constructor by hand, no default call
        CrossXInterface.__init__(self,0.0,sig_s)

        #Update sigma_a and sigma_t
        self.updateCrossX(rho,temp) ##evaluate absorption cross section

    
    #----------------------------------------------------------------------------
    def updateCrossX(self,rho,temp):
    
        #Set the base class values
        self.sig_a = rho*self.coeff/(temp**3.) 
        self.sig_s = self.sig_s
        self.sig_t = self.sig_s + self.sig_a
