#!/bin/bash

INPUTS="60000"
DATASET="mnist"
EPOCHS=150
LR=0.002
INFERENCE="svi"
DEVICE="cpu"

cd ~/adversarial_examples/src/
source ~/virtualenvs/venv_gpu/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
RESULTS="../results/$DATE/"
mkdir -p $RESULTS
OUT="${RESULTS}${TIME}_out.txt"

python3 reducedBNN.py --inputs=$INPUTS --dataset=$DATASET --epochs=$EPOCHS --lr=$LR --inference=$INFERENCE --device=$DEVICE &> $OUT

deactivate