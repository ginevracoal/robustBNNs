#!/bin/bash

# ------ settings -------- #

INPUTS="60000"
DATASET="mnist" # mnist, cifar, fashion_mnist
DEVICE="cuda" # cpu, cuda
INFERENCE="svi" # svi, hmc

EPOCHS=300
LR=0.0001

HMC_SAMPLES=10
WARMUP=100

ATTACK="fgsm" # fgsm, pgd

# ------ execution -------- #

cd ~/robustBNNs/
source venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="tests/$DATE/"
mkdir -p $TESTS
OUT="${TESTS}${TIME}_out.txt"

# python3 reducedBNN.py --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --hmc_samples=$HMC_SAMPLES --warmup=$WARMUP --device=$DEVICE &>> $OUT
# python3 lossGradients.py  --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --hmc_samples=$HMC_SAMPLES --warmup=$WARMUP --device=$DEVICE &>> $OUT
# python3 plot.py  --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --hmc_samples=$HMC_SAMPLES --warmup=$WARMUP --device=$DEVICE &>> $OUT
python3 adversarialAttacks.py --attack=$ATTACK --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --hmc_samples=$HMC_SAMPLES --warmup=$WARMUP --device=$DEVICE &>> $OUT

deactivate