#!/bin/bash

# ------ settings -------- #

INPUTS="60000"
DATASET="mnist" # mnist, cifar, fashion_mnist
DEVICE="cuda" # cpu, cuda
INFERENCE="svi" # svi, mcmc

EPOCHS=200
LR=0.0001

MCMC_SAMPLES=10 #100
WARMUP=80 #500

# ------ execution -------- #

cd ~/robustBNNs/
source venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="tests/$DATE/"
mkdir -p $TESTS
OUT="${TESTS}${TIME}_out.txt"

# python3 reducedBNN.py --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --mcmc_samples=$MCMC_SAMPLES --warmup=$WARMUP --device=$DEVICE &> $OUT
python3 lossGradients.py  --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --mcmc_samples=$MCMC_SAMPLES --warmup=$WARMUP --device=$DEVICE &> $OUT
python3 plot.py  --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --mcmc_samples=$MCMC_SAMPLES --warmup=$WARMUP --device=$DEVICE &> $OUT

deactivate