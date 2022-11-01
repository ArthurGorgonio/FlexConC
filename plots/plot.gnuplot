set terminal png size 1362,666;
set output ARG4
set key outside;
set key right top;
set xlabel "Batch";
set ylabel "Valor"
set grid;
if (ARG5 eq "1") {
    set yrange [0:1];
} else {
    if (ARG5 eq "2") {
        set yrange [-1:1];
    } else {
        set autoscale y;
    }
}
set title ARG3

purple = "#8500bb";
yellow = "#ba8b13";
green = "#00bb7d";
blue = "#0A9EC7";
red = "#FA5025";

data = "< paste ".ARG1." ".ARG2."

if (ARG6 eq "1") {
    plot data using 1:3 title 'DyDaSL - FT' w l lt rgb yellow lw 2, \
    data using 1:($9==1 ? column(3) : NaN) title 'Drift - FT' pt 4 ps 1.5 lt rgb yellow, \
    data using 1:5 title 'DyDaSL - N' w l lt rgb purple lw 2, \
    data using 1:($10==1 ? column(5) : NaN) title 'Drift - N' pt 3 ps 1.5 lt rgb purple, \
    data using 1:7 title 'DyDaSL - S' w l lt rgb green lw 2, \
    data using 1:($11==1 ? column(7) : NaN) title 'Drift - S' pt 6 ps 1.5 lt rgb green
} else {
    if (ARG6 eq "2") {
        plot data using 1:2 title 'DyDaSL - FT' w l lt rgb yellow lw 2, \
        data using 1:4 title 'DyDaSL - N' w l lt rgb purple lw 2, \
        data using 1:6 title 'DyDaSL - S' w l lt rgb green lw 2
    } else {
        plot data using 1:3 title 'DyDaSL - FT' w lp lt rgb yellow lw 2, \
        data using 1:5 title 'DyDaSL - N' w lp lt rgb purple lw 2, \
        data using 1:7 title 'DyDaSL - S' w lp lt rgb green lw 2
    }
}
