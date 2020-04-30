import argparse
import os
from directories import *
from utils import *
import pyro
import torch
from torch import nn
import torch.nn.functional as nnf
import numpy as np
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
import torch.optim as torchopt
from pyro import poutine
import pyro.optim as pyroopt
import torch.nn.functional as F
from utils import plot_loss_accuracy
import torch.distributions.constraints as constraints
softplus = torch.nn.Softplus()
from pyro.nn import PyroModule
from bnn import BNN


class MoonsBNN(BNN):
    def __init__(self, dataset_name, hidden_size, activation, architecture, inference, 
                 epochs, lr, n_samples, warmup, input_shape, output_size, n_inputs):
        super(MoonsBNN, self).__init__(dataset_name, hidden_size, activation, architecture, 
                inference, epochs, lr, n_samples, warmup, input_shape, output_size)
        self.name = self.get_name(epochs, lr, n_samples, warmup, n_inputs)


def main(args):

    dataset = "half_moons"

    init = (args.hidden_size, args.activation, args.architecture, 
            args.inference, args.epochs, args.lr, args.samples, args.warmup)
    
    train_loader, test_loader, inp_shape, out_size = \
                            data_loaders(dataset_name=dataset, batch_size=64, 
                                         n_inputs=args.inputs, shuffle=True)

    bnn = MoonsBNN(dataset, *init, inp_shape, out_size, args.inputs)
   
    bnn.train(train_loader=train_loader, device=args.device)
    # bnn.load(device=args.device, rel_path=DATA)

    bnn.evaluate(test_loader=test_loader, device=args.device)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="Toy example on half moons")

    parser.add_argument("--inputs", default=1000, type=int)
    parser.add_argument("--hidden_size", default=128, type=int, help="power of 2 >= 16")
    parser.add_argument("--activation", default="leaky", type=str, 
                        help="relu, leaky, sigm, tanh")
    parser.add_argument("--architecture", default="fc2", type=str, help="fc, fc2")
    parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--samples", default=10, type=int)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
   
    main(args=parser.parse_args())