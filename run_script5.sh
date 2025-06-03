#!/bin/bash

for run in {1..3}; do
    echo "--- Run $run/3 ---"
    python prune.py --data ISCXVPN2016
done

