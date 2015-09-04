# Reference slopes
slope1 = 1.00
slope2 = 2.00

stats "e_convergence.gpl" using 1 noout
h_min = STATS_min

stats "e_convergence.gpl" using 2 noout
L2_min = STATS_min
L2_max = STATS_max

L2c1 = (1.0/h_min)**slope1 * L2_min
L2c2 = (1.0/h_min)**slope2 * L2_min
L2ref1(x) = L2c1 * x**slope1
L2ref2(x) = L2c2 * x**slope2
L2string1 = sprintf("m = %.2f Reference slope",slope1)
L2string2 = sprintf("m = %.2f Reference slope",slope2)

set xlabel "Mesh Size"
set logscale xy
set key top left
set format y "10^{%L}"
set format x "10^{%L}"

set terminal postscript enhanced color
output_file = "e_convergence.pdf"
set output '| ps2pdf - '.output_file

# Plot L-2 error
if (L2_max > 1) {
  set yrange[*:1]
} else {
  set yrange[*:*]
}
set ylabel "Internal energy (e) L-2 Error"
plot "e_convergence.gpl" using 1:2 with linesp linetype 1 linecolor 1 pointtype 1 title "e error",\
   L2ref1(x) title L2string1 with lines linestyle 2 linecolor 9,\
   L2ref2(x) title L2string2 with lines linestyle 3 linecolor 9
