#!/bin/bash

for amount in $(seq 0.1 0.1 0.6); do
    echo "Testing pruning amount: $amount"
    
    for run in {1..3}; do
        echo "--- Run $run/3 ---"
        python prune.py --data MALAYAGT --amount $amount
    done
    
    echo "---------------------"
done

