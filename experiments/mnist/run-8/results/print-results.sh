#!/bin/bash

awk '/n_hidden : 1000/ && /learning_rate : 0.001/ {print $0}' \
    mnist-results.txt > model-1000-0.001-results.txt

awk '/n_hidden : 1000/ && /learning_rate : 0.01/ {print $0}' \
    mnist-results.txt > model-1000-0.01-results.txt

awk '/n_hidden : 500/ && /learning_rate : 0.001/ {print $0}' \
    mnist-results.txt > model-500-0.001-results.txt

awk '/n_hidden : 500/ && /learning_rate : 0.01/ {print $0}' \
    mnist-results.txt > model-500-0.01-results.txt

for fname in `ls model-*.txt`
do
    n_hid=`echo $fname | cut -d'-' -f2`
    l_rate=`echo $fname | cut -d'-' -f3`

    echo "Learning rate: "$l_rate"; Hiddens: "$n_hid
    awk 'BEGIN {sum=0; sq_sum=0; n=0} \
         // {sum+=$23; sq_sum+=$23^2; n++} \
         END {mean= sum/n; std=sqrt((sq_sum - (sum^2)/n)/n); 
         printf("\tMean: %.3f; Std.: %.5f\n", mean*100, std)}' $fname

    rm $fname
done

