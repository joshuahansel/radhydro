## @package src.crossXInterface
#  Contains cross section classes.

#================================================================================
## Cross section class.
#
#  Class that handles the cross sections at a particular point in space.  If
#  we want to make temperature dependent cross sections later, we will make a
#  class that derives from this class. The derived class will have a function to
#  update the need to change this later.
# 
#  All Cross sections have units of \f$\mbox{cm}^{-1}\f$.  The basic cross
#  sections included by this class are \f$\sigma_t\f$, \f$\sigma_s\f$, and
#  \f$\sigma_a\f$, where \f$\sigma_t = \sigma_s+\sigma_a\f$.
#================================================================================
class CrossXInterface(object):

    #----------------------------------------------------------------------------
    ## Constructor.
    #
    #  Takes \f$\sigma_a\f$ and \f$\sigma_s\f$ and computes \f$\sigma_t\f$.
    #
    #  @param[in] self    self
    #  @param[in] sigma_s \f$\sigma_s\f$, the absorption cross section
    #  @param[in] sigma_t \f$\sigma_t\f$, the scattering cross section
    #----------------------------------------------------------------------------
    def __init__(self, sigma_s, sigma_t):

        ## \f$\sigma_s\f$, the scattering cross section
        self.sig_s = sigma_s
        ## \f$\sigma_t\f$, the absorption cross section
        self.sig_t = sigma_t
        ## \f$\sigma_a\f$, the total cross section
        self.sig_a = sigma_t - sigma_s
               
    #----------------------------------------------------------------------------
    ## Print string definition.
    #
    #  Defines how to print the object. Prints \f$\sigma_s\f$, \f$\sigma_a\f$,
    #  and \f$\sigma_t\f$.
    #
    #  @param[in] self      self
    #  @return    string to be printed with the print command
    #----------------------------------------------------------------------------
    def __str__(self):

        print_str = "\sigma_s : %.12g, \sigma_a : %.12g, \sigma_t : %.12g" % \
                    (self.sig_s, self.sig_a, self.sig_t) 

        return print_str

    #----------------------------------------------------------------------------
    ## Update function cross sections.
    #
    #  In this base class, cross sections are not updated by default; if it is
    #  necessary to update cross sections, this will be done by a derived class.
    #
    #  @param[in] args   arbitrary number of arguments
    #  @param[in] kwargs arbitrary number of keyword arguments
    #----------------------------------------------------------------------------
    def updateCrossX(self,*args,**kwargs):

        return

#===================================================================================
## Inverse cubed cross section class.
#
#  This is an example of a derived cross section.  In this case it is a simple InvCubed
#  relation where you need to pass in a new density and temperature to update the cross
#  section. Anything that uses the base class should be able to interact with base class
#  objects.
#
#  These cross sections are computed as
#
#  \f[
#     \sigma_s = \sigma_s^{'''} \rho
#  \f]
#  \f[
#     \sigma_a = c_a \rho T^{-3}
#  \f]
#
class InvCubedCrossX(CrossXInterface):

    #----------------------------------------------------------------------------
    ## Constructor.
    #
    #  Takes the micro scattering cross section, density, temperature, and
    #  scaling coefficient and computes cross sections.
    #
    #  @param[in] self          self
    #  @param[in] sigma_s_micro \f$\sigma_s^{'''}\f$, the micro scattering
    #             cross section, equal to \f$\frac{\sigma_s}{\rho}\f$
    #  @param[in] rho           \f$\rho\f$, density
    #  @param[in] temp          \f$T\f$, temperature
    #  @param[in] scale_coeff   \f$c_a\f$, the scale coefficient in
    #             \f$\sigma_a = c_a \rho T^{-3}\f$
    #----------------------------------------------------------------------------
    def __init__(self, sigma_s_micro, rho, temp, scale_coeff=1.0):

        ## scale coefficient \f$c_a\f$ in \f$\sigma_a = c_a \rho T^{-3}\f$
        self.coeff = scale_coeff    

        # compute scattering cross section
        sig_s = sigma_s_micro*rho

        # call base class constructor by hand, no default call
        CrossXInterface.__init__(self,sig_s,0.0)

        # update sigma_a and sigma_t
        self.updateCrossX(rho,temp)

    #----------------------------------------------------------------------------
    ## Update function for cross sections.
    #
    #  @param[in] self self
    #  @param[in] rho  \f$\rho\f$, density
    #  @param[in] temp \f$T\f$, temperature
    #----------------------------------------------------------------------------
    def updateCrossX(self,rho,temp):
    
        # set the base class values
        self.sig_a = rho*self.coeff/(temp**3.) 
        self.sig_s = self.sig_s
        self.sig_t = self.sig_s + self.sig_a
