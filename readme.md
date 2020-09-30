## Robustness of Bayesian Neural Networks to Gradient-Based Attacks

Pyro implementation for paper "Robustness of Byesian Neural Networks to Gradient-Based Attacks", Ginevra Carbone, Matthew Wicker, Luca Laurenti, Andrea Patane, Luca Bortolussi, Guido Sanguinetti, 2020.

### Abstract

Vulnerability to adversarial attacks is one of the principal hurdles to the adoption of deep learning in safety-critical applications. Despite significant efforts, both practical and theoretical, the problem remains open. In this paper, we analyse the geometry of adversarial attacks in the large-data, overparametrized limit for Bayesian Neural Networks (BNNs). We show that, in the limit, vulnerability to gradient-based attacks arises as a result of degeneracy in the data distribution, i.e., when the data lies on a lower-dimensional submanifold of the ambient space. As a direct consequence, we demonstrate that in the limit BNN posteriors are robust to gradient-based adversarial attacks. Experimental results on the MNIST and Fashion MNIST datasets with BNNs trained with Hamiltonian Monte Carlo and Variational Inference support this line of argument, showing that BNNs can display both high accuracy and robustness to gradient based adversarial attacks. 

### Install

```
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Instructions

There are four datasets available for tests: MNIST, Fashion MNIST, CIFAR10, Half Moons.

Code runs on python 3 and pyro 1.3.0.

**Scripts**

`model_nn.py` trains and evaluates a deterministic Neural Network.

`model_bnn.py` trains and evaluates a Bayesian Neural Network.

`model_ensemble.py` trains and evaluates an ensemble of deterministic Neural Networks sharing the same architecture, but with different random initializations and randomly shuffled batches.

`lossGradients.py` loads a trained BNN and computes the expected loss gradients over test points, with an increasing number of posterior samples.

`adversarialAttacks.py` implements FGSM and PGD classic and Bayesian adversarial attacks, and robustness measures. It loads a trained NN, BNN or ensemble network, and then computes the attacks.

`plot_baseline_attacks.py` loads determistic, Bayesian and ensemble versions of the same architecture and attacks them with the chosen method. Then, it plots adversarial accuracy and softmax robustness for an increasing number of samples. 

`plot_eps_attacks.py` loads and attacks a BNN with an increasing attack strenght and an increasing number of samples. The same posterior samples are used when evaluating against the attacks.

`plot_gradients_components.py` loads a BNN, computes and plots gradients components and vanishing gradients heatmaps for an increasing number of posterior samples.

`grid_search_halfMoons.py` runs a grid search on the Half Moons dataset, then computes expected loss gradients and adversarial attacks on test data.

`plot_halfMoons_overparam.py` loads gradients components for the half moons dataset and 
plots their behaviour towards the overparametrized limit.


**Input arguments**

- `--n_inputs` is the number of training points. 
- `--model_idx` is the index of the model chosen from a list of pre-defined settings, that can be found in each model script. 
- `--savedir` is the directory where models and generated data are saved and ready to be loaded. The default input is `TESTS`, which corresponds to `tests/%Y-%m-%d` directory.
- If `--train` is True the model is trained, otherwise it is loaded from the chosen directory. 
- If `--test` is True, the model is evaluated on test data. 
- `--device=="cuda"` runs the code on GPU, while `--device=="cpu"` runs it on CPU.

Additional descriptions of the arguments can be found in parser `--help`.

**Examples**

Reproducing paper figures.

Figure 1
```
python3 grid_search_halfMoons.py --test_points=100 --device=cuda --compute_grads=True --compute_attacks=False
python3 plot_halfMoons_overparam.py --test_points=100 --device=cuda 
```

Figures 2 and 3
```
python3 model_bnn.py --n_inputs=60000 --model_idx=0 --device=cuda --train=True --test=True
python3 plot_gradients_components.py --n_inputs=10000 --model_idx=0 --compute_grads=True  --device=cuda --stripplot=True --heatmaps=True
```




