set style histogram clustered
set xtics rotate by 0
unset title
set xlabel "Comite de classificadores"
set ylabel "Acuracia dos classificadores (%)"
set boxwidth .5
set style fill solid
set term png
set key outside
set output "comite_resultado.png"
plot filename using 3:xtic(2) title "Naive Bayes" with histograms, \
     "" using 4 title "Tree" with histograms, \
     "" using 5 title "Knn" with histograms, \
     "" using 6 title "Heterogeneo" with histograms
