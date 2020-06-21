#!/bin/bash

DEVICE="cuda"
N_INPUTS=10
MODEL_IDX=0
SAVEDIR="TESTS"

source venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
TESTS="tests/$DATE/"
mkdir -p $TESTS
OUT="${TESTS}${TIME}_test.txt"

python3 model_nn.py --n_inputs=$N_INPUTS --model_idx=$MODEL_IDX --train=True --test=True --savedir=$SAVEDIR --device=$DEVICE
python3 model_bnn.py --n_inputs=$N_INPUTS --model_idx=$MODEL_IDX --train=True --test=True --savedir=$SAVEDIR --device=$DEVICE

python3 lossGradients.py --n_inputs=$N_INPUTS --model_idx=$MODEL_IDX --device=$DEVICE
python3 plot_gradients_components.py --heatmaps=False --stripplot=True --compute_grads=True --n_inputs=$N_INPUTS --model_idx=$MODEL_IDX --device=$DEVICE

python3 adversarialAttacks.py --deterministic=True --attack_method="fgsm" --attack=True --n_inputs=$N_INPUTS --model_idx=$MODEL_IDX --train=False --test=False --savedir=$SAVEDIR --device=$DEVICE
python3 adversarialAttacks.py --deterministic=False --attack_method="pgd" --attack=True --n_inputs=$N_INPUTS --model_idx=$MODEL_IDX --train=False --test=False --savedir=$SAVEDIR --device=$DEVICE

deactivate