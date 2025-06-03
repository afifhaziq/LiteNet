#!/bin/bash


for i in {1..5}
do
    python main.py --data MALAYAGT -s 1 -f 5
    python main.py --data MALAYAGT -s 1 -f 10
    python main.py --data MALAYAGT -s 4 -f 5
    python main.py --data MALAYAGT -s 3 -f 10
    python main.py --data MALAYAGT -s 2 -f 20
    python main.py --data MALAYAGT -s 5 -f 10
    python main.py --data MALAYAGT -s 3 -f 20
    python main.py --data MALAYAGT -s 7 -f 10
    python main.py --data MALAYAGT -s 4 -f 20
    python main.py --data MALAYAGT -s 5 -f 20
    python main.py --data MALAYAGT -s 25 -f 20
    python main.py --data MALAYAGT -s 35 -f 20 
    python main.py --data MALAYAGT -s 37 -f 20
done


