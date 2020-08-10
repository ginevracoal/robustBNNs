"""
Deterministic Neural Network model
"""
from savedir import *
from utils import *
import os
import argparse
import math 
import numpy as np

import torch
from torch import nn
import torch.nn.functional as nnf
import torch.optim as torchopt

DEBUG = False

saved_NNs = {"model_0":{"dataset":"mnist", "hidden_size":512, "activation":"leaky",
                        "architecture":"conv", "epochs":5, "lr":0.01},
             "model_1":{"dataset":"mnist", "hidden_size":512, "activation":"leaky",
                        "architecture":"fc2", "epochs":10, "lr":0.01}}


class NN(nn.Module):

    def __init__(self, dataset_name, input_shape, output_size, hidden_size, activation, 
                 architecture, lr, epochs):

        if math.log(hidden_size, 2).is_integer() is False or hidden_size<16:
            raise ValueError("\nhidden size should be a power of 2 greater than 16.")

        super(NN, self).__init__()
        self.dataset_name = dataset_name
        self.loss_func = nn.CrossEntropyLoss()
        self.architecture = architecture
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.lr, self.epochs = lr, epochs

        self.name = self.get_name(dataset_name, hidden_size, activation, architecture, lr, epochs)
        self.set_model(architecture, activation, input_shape, output_size, hidden_size)
        # print("\nTotal number of weights =", sum(p.numel() for p in self.parameters()))

    def get_name(self, dataset_name, hidden_size, activation, architecture, lr, epochs):
        return str(dataset_name)+"_nn_hid="+str(hidden_size)+"_act="+str(activation)+\
               "_arch="+str(architecture)+"_ep="+str(epochs)+"_lr="+str(lr)

    def set_model(self, architecture, activation, input_shape, output_size, hidden_size):

        input_size = input_shape[0]*input_shape[1]*input_shape[2]
        in_channels = input_shape[0]
        n_classes = output_size

        if activation == "relu":
            activ = nn.ReLU
        elif activation == "leaky":
            activ = nn.LeakyReLU
        elif activation == "sigm":
            activ = nn.Sigmoid
        elif activation == "tanh":
            activ = nn.Tanh
        else: 
            raise AssertionError("\nWrong activation name.")

        if architecture == "fc":
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, output_size))

        elif architecture == "fc2":
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, output_size))

        elif architecture == "conv":

            if self.dataset_name not in ["mnist","fashion_mnist"]:
                raise NotImplementedError()

            self.model = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=5),
                activ(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, hidden_size, kernel_size=5),
                activ(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Flatten(),
                nn.Linear(int(hidden_size/(4*4))*input_size, output_size))
        else:
            raise NotImplementedError()

    def forward(self, inputs, *args, **kwargs):
        x = self.model(inputs)
        return x

    def save(self, savedir=None, seed=None):
        name = self.name 
        directory = name if savedir is None else savedir
        filename = name+"_weights.pt" if seed is None else name+"_weights_"+str(seed)+".pt"

        os.makedirs(os.path.dirname(TESTS+directory+"/"), exist_ok=True)
        print("\nSaving: ", TESTS+directory+"/"+filename)
        torch.save(self.state_dict(), TESTS+directory+"/"+filename)

        if DEBUG:
            print("\nCheck saved weights:")
            print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
            print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])

    def load(self, device, savedir=None, seed=None, rel_path=TESTS):
        name = self.name
        directory = name if savedir is None else savedir
        filename = name+"_weights.pt" if seed is None else name+"_weights_"+str(seed)+".pt"

        print("\nLoading: ", rel_path+directory+"/"+filename)
        self.load_state_dict(torch.load(rel_path+directory+"/"+filename))
        print("\n", list(self.state_dict().keys()), "\n")
        self.to(device)

        if DEBUG:
            print("\nCheck loaded weights:")    
            print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
            print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])

    def train(self, train_loader, device, seed=0):
        print("\n == NN training ==")
        random.seed(seed)
        self.to(device)

        optimizer = torchopt.Adam(params=self.parameters(), lr=self.lr)

        start = time.time()
        for epoch in range(self.epochs):
            total_loss = 0.0
            correct_predictions = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).argmax(-1)
                
                optimizer.zero_grad()
                outputs = self.forward(x_batch)
                loss = self.loss_func(outputs, y_batch)
                loss.backward()
                optimizer.step()

                predictions = outputs.argmax(-1)
                correct_predictions += (predictions == y_batch).sum()
                total_loss += loss.item()

            total_loss = total_loss / len(train_loader.dataset)
            accuracy = 100 * correct_predictions / len(train_loader.dataset)
            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
                  end="\t")

        execution_time(start=start, end=time.time())
        self.save()

    def evaluate(self, test_loader, device, *args, **kwargs):
        self.to(device)
        # random.seed(0)

        with torch.no_grad():

            correct_predictions = 0.0

            for x_batch, y_batch in test_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).argmax(-1)
                outputs = self(x_batch)
                predictions = outputs.argmax(-1)
                correct_predictions += (predictions == y_batch).sum()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("\nAccuracy: %.2f%%" % (accuracy))
            return accuracy

def main(args):

    rel_path=DATA if args.savedir=="DATA" else TESTS
    train_inputs = 100 if DEBUG else None

    dataset, hid, activ, arch, ep, lr = saved_NNs["model_"+str(args.model_idx)].values()

    train_loader, test_loader, inp_shape, out_size = \
                            data_loaders(dataset_name=dataset, batch_size=64, 
                                         n_inputs=args.n_inputs, shuffle=True)

    nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size, 
            hidden_size=hid, activation=activ, architecture=arch, epochs=ep, lr=lr)

    if args.train:
        nn.train(train_loader=train_loader, device=args.device)
    else:
        nn.load(device=args.device, rel_path=rel_path)
    
    if args.test:
        nn.evaluate(test_loader=test_loader, device=args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Base NN")
    parser.add_argument("--n_inputs", default=60000, type=int, help="number of input points")
    parser.add_argument("--model_idx", default=0, type=int, help="choose idx from saved_NNs")
    parser.add_argument("--train", default=True, type=eval)
    parser.add_argument("--test", default=True, type=eval)
    parser.add_argument("--savedir", default='DATA', type=str, help="DATA, TESTS")  
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
    main(args=parser.parse_args())
