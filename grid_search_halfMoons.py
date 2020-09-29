"""
Grid search on the Half Moons dataset, compute gradients and attacks.
"""

from utils import *
from savedir import *
from model_bnn import BNN
from lossGradients import loss_gradients, load_loss_gradients
from adversarialAttacks import attack, attack_evaluation, load_attack

import os
import pandas
import argparse
import itertools
import numpy as np
import pyro

class MoonsBNN(BNN):
    def __init__(self, hidden_size, activation, architecture, inference, 
                 epochs, lr, n_samples, warmup, n_inputs, input_shape, output_size):
        super(MoonsBNN, self).__init__("half_moons", hidden_size, activation, architecture, 
                inference, epochs, lr, n_samples, warmup, input_shape, output_size,
                step_size=0.001)
        self.name = self.get_name(n_inputs)

#####################
# training networks #
#####################

def _train(hidden_size, activation, architecture, inference, 
           epochs, lr, n_samples, warmup, n_inputs, posterior_samples, device):

    batch_size = 64 if inference=="svi" else 1024

    train_loader, _, inp_shape, out_size = \
            data_loaders(dataset_name="half_moons", batch_size=batch_size, 
                         n_inputs=n_inputs, shuffle=False)

    bnn = MoonsBNN(hidden_size, activation, architecture, inference, 
                   epochs, lr, n_samples, warmup, n_inputs, inp_shape, out_size)
    bnn.train(train_loader=train_loader, device=device)

def serial_train(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples):

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    for init in combinations:
        _train(*init, "cuda")

def parallel_train(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples):
    from joblib import Parallel, delayed

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    Parallel(n_jobs=10)(
        delayed(_train)(*init, "cpu") for init in combinations)

#######################
# computing gradients #
#######################

def _compute_grads(hidden_size, activation, architecture, inference, 
           epochs, lr, n_samples, warmup, n_inputs, posterior_samples, rel_path, test_points,
           device):

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name="half_moons", batch_size=32, n_inputs=test_points, shuffle=True)

    bnn = MoonsBNN(hidden_size, activation, architecture, inference, 
                   epochs, lr, n_samples, warmup, n_inputs, inp_shape, out_size)
    bnn.load(device=device, rel_path=rel_path)
    loss_gradients(net=bnn, n_samples=posterior_samples, savedir=bnn.name+"/", 
                    data_loader=test_loader, device=device, filename=bnn.name)

def parallel_compute_grads(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples, 
                          rel_path, test_points):
    from joblib import Parallel, delayed

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    Parallel(n_jobs=10)(
        delayed(_compute_grads)(*init, rel_path, test_points, "cpu") for init in combinations)

def serial_compute_grads(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples, 
                          rel_path, test_points):

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    for init in combinations:
        _compute_grads(*init, rel_path, test_points, "cuda")

#####################
# computing attacks #
#####################

def _compute_attacks(method, hidden_size, activation, architecture, inference, 
           epochs, lr, n_samples, warmup, n_inputs, posterior_samples, rel_path, test_points):

    _, _, x_test, y_test, inp_shape, out_size = \
            load_dataset(dataset_name="half_moons", n_inputs=test_points, channels="first") 

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    bnn = MoonsBNN(hidden_size, activation, architecture, inference, 
                   epochs, lr, n_samples, warmup, n_inputs, inp_shape, out_size)
    bnn.load(device="cpu", rel_path=rel_path)
        
    x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, dataset_name="half_moons", 
                      device="cpu", method=method, filename=bnn.name, 
                      n_samples=posterior_samples)

def parallel_grid_attack(method, hidden_size, activation, architecture, inference, epochs, lr, 
                         n_samples, warmup, n_inputs, posterior_samples, rel_path, test_points):
    from joblib import Parallel, delayed

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                        epochs, lr, n_samples, warmup, n_inputs, posterior_samples))

    Parallel(n_jobs=10)(
        delayed(_compute_attacks)(method, *init, rel_path, test_points) 
                for init in combinations)

def grid_attack(method, hidden_size, activation, architecture, inference, epochs, lr, 
                  n_samples, warmup, n_inputs, posterior_samples, test_points, device="cuda", 
                  rel_path=TESTS):
   
    _, _, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name="half_moons", n_inputs=test_points, channels="first") 

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                        epochs, lr, n_samples, warmup, n_inputs))
    for init in combinations:

        bnn = MoonsBNN(*init, inp_shape, out_size)
        bnn.load(device=device, rel_path=rel_path)
        
        for p_samp in posterior_samples:
            x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, dataset_name="half_moons", 
                              device=device, method=method, filename=bnn.name, 
                              n_samples=p_samp)

def main(args):

    # === settings ===

    inference = ["hmc"]
    n_samples = [250]
    warmup = [100] #, 200, 500]
    n_inputs = [5000]#, 10000, 15000]
    epochs = [None]
    lr = [None]
    hidden_size = [32, 128]#, 256, 512] 
    activation = ["leaky"]
    architecture = ["fc2"]
    attack = "fgsm"
    posterior_samples = [10,20,50]
    test_points = 100

    # === grid search ===

    init = (hidden_size, activation, architecture, inference, 
            epochs, lr, n_samples, warmup, n_inputs, posterior_samples)
    
    rel_path=DATA if args.savedir=="DATA" else TESTS

    if args.device=="cuda":

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # serial_train(*init)
        # serial_compute_grads(*init, rel_path=rel_path, test_points=test_points)
        grid_attack(attack, *init, test_points=test_points, device=args.device, 
                    rel_path=rel_path) 

    else:

        torch.set_default_tensor_type('torch.FloatTensor')
        parallel_train(*init)
        parallel_compute_grads(*init, rel_path=rel_path, test_points=test_points)
        parallel_grid_attack(attack, *init, rel_path=rel_path, test_points=test_points) 


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="Grid search BNN model")
    parser.add_argument("--savedir", default='TESTS', type=str, help="choose dir for loading the BNN: DATA, TESTS")  
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
    main(args=parser.parse_args())