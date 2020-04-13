#!/bin/bash

# ------ settings -------- #

INPUTS="10000"
DATASET="mnist" # mnist, cifar, fashion_mnist
DEVICE="cuda" # cpu, cuda

ARCHITECTURE="conv" # fc, fc2, conv
ACTIVATION="leaky" # leaky, sigm, tanh
HIDDEN_SIZE=32

INFERENCE="hmc" # svi, hmc

EPOCHS=50
LR=0.001

WARMUP=10
N_SAMPLES=20

ATTACK="fgsm" # fgsm, pgd

# ------ execution -------- #

cd ~/robustBNNs/
source venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="tests/$DATE/"
mkdir -p $TESTS
OUT="${TESTS}${TIME}_out.txt"

python3 bnn.py --inputs=$INPUTS --dataset=$DATASET --architecture=$ARCHITECTURE --activation=$ACTIVATION --hidden_size=$HIDDEN_SIZE --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --samples=$N_SAMPLES --warmup=$WARMUP --device=$DEVICE &>> $OUT

# python3 reducedBNN.py --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --samples=$N_SAMPLES --warmup=$WARMUP --device=$DEVICE &>> $OUT
# python3 adversarialAttacks.py --attack=$ATTACK --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --samples=$N_SAMPLES --warmup=$WARMUP --device=$DEVICE &>> $OUT

# python3 lossGradients.py  --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --samples=$N_SAMPLES --warmup=$WARMUP --device=$DEVICE &>> $OUT
# python3 plot.py  --inputs=$INPUTS --dataset=$DATASET --inference=$INFERENCE --epochs=$EPOCHS --lr=$LR --samples=$N_SAMPLES --warmup=$WARMUP --device=$DEVICE &>> $OUT

deactivate