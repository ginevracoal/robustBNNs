#!/bin/bash

INPUTS="60000"
DATASET="mnist"
EPOCHS=20
LR=0.001
INFERENCE="svi"
DEVICE="cuda"

cd ~/robustBNNs/
source venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="tests/$DATE/"
mkdir -p $TESTS
OUT="${TESTS}${TIME}_out.txt"

python3 reducedBNN.py --inputs=$INPUTS --dataset=$DATASET --epochs=$EPOCHS --lr=$LR --inference=$INFERENCE --device=$DEVICE &> $OUT

deactivate