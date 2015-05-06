## @package src.utilityFunctions
#  Contains helper functions that do not belong in any particular class.

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


