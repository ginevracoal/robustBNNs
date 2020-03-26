#!/bin/bash

INPUTS="60000"
EPOCHS=150
DATASET="mnist"
INFERENCE="svi"
DEVICE="cpu"

cd ~/adversarial_examples/src/
source ~/virtualenvs/venv_gpu/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
RESULTS="../results/$DATE/"
mkdir -p $RESULTS
OUT="${RESULTS}${TIME}_out.txt"

python3 reducedBNN.py --inputs=$INPUTS --epochs=$EPOCHS --dataset=$DATASET --inference=$INFERENCE --device=$DEVICE &> $OUT

deactivate