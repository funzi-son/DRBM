#!/bin/bash

awk '/n_hidden : 50/ && /learning_rate : 0.001/ {print $0}' \
    20newsgroups-results.txt > model-50-0.001-results.txt

awk '/n_hidden : 50/ && /learning_rate : 0.01/ {print $0}' \
    20newsgroups-results.txt > model-50-0.01-results.txt

awk '/n_hidden : 100/ && /learning_rate : 0.001/ {print $0}' \
    20newsgroups-results.txt > model-100-0.001-results.txt

awk '/n_hidden : 100/ && /learning_rate : 0.01/ {print $0}' \
    20newsgroups-results.txt > model-100-0.01-results.txt

awk '/n_hidden : 500/ && /learning_rate : 0.001/ {print $0}' \
    20newsgroups-results.txt > model-500-0.001-results.txt

awk '/n_hidden : 500/ && /learning_rate : 0.01/ {print $0}' \
    20newsgroups-results.txt > model-500-0.01-results.txt

awk '/n_hidden : 1000/ && /learning_rate : 0.001/ {print $0}' \
    20newsgroups-results.txt > model-1000-0.001-results.txt

awk '/n_hidden : 1000/ && /learning_rate : 0.01/ {print $0}' \
    20newsgroups-results.txt > model-1000-0.01-results.txt

for fname in `ls model-*.txt`
do
    n_hids=`echo $fname | cut -d'-' -f2`
    l_rate=`echo $fname | cut -d'-' -f3`
    n_bins=`echo $fname | cut -d'-' -f4`

    echo "Learning rate: "$l_rate"; Hiddens: "$n_hids"; Bins: "$n_bins
    awk 'BEGIN {sum=0; sq_sum=0; n=0} \
         // {sum+=$23; sq_sum+=$23^2; n++} \
         END {mean= sum/n; std=sqrt((sq_sum - (sum^2)/n)/n); 
         printf("\tMean (test): %.3f; Std. (test): %.5f\n", mean*100, std)}' $fname

    awk 'BEGIN {sum=0; sq_sum=0; n=0} \
         // {sum+=$3; sq_sum+=$3^2; n++} \
         END {mean= sum/n; std=sqrt((sq_sum - (sum^2)/n)/n); 
         printf("\tMean (valid): %.3f; Std. (valid): %.5f\n", mean*100, std)}' $fname

    rm $fname
done

