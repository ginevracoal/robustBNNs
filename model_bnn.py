"""
Bayesian Neural Network model.
"""

import argparse
import os
from savedir import *
from utils import *
import numpy as np
import pandas as pd 
import copy

import torch
from torch import nn
import torch.nn.functional as nnf
import torch.optim as torchopt
import torch.distributions.constraints as constraints
softplus = torch.nn.Softplus()

import pyro
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
import pyro.optim as pyroopt
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.distributions import OneHotCategorical, Normal, Categorical, Uniform
from pyro.nn import PyroModule

from utils import plot_loss_accuracy
from collections import OrderedDict
from model_nn import NN


DEBUG=False


saved_BNNs = {"model_0":["mnist", {"hidden_size":512, "activation":"leaky",
                         "architecture":"conv", "inference":"svi", "epochs":5, 
                         "lr":0.01, "n_samples":None, "warmup":None}],
              "model_1":["mnist", {"hidden_size":512, "activation":"leaky",
                         "architecture":"fc2", "inference":"hmc", "epochs":None,
                         "lr":None, "n_samples":100, "warmup":50}],
              "model_2":["fashion_mnist", {"hidden_size":1024, "activation":"leaky",
                         "architecture":"conv", "inference":"svi", "epochs":10,
                         "lr":0.001, "n_samples":None, "warmup":None}],
              "model_3":["fashion_mnist", {"hidden_size":1024, "activation":"leaky",
                         "architecture":"fc2", "inference":"hmc", "epochs":None,
                         "lr":None, "n_samples":100, "warmup":50}],
              "model_4":["fashion_mnist", {"hidden_size":1024, "activation":"leaky",
                         "architecture":"conv", "inference":"svi", "epochs":5,
                         "lr":0.01, "n_samples":None, "warmup":None}],
              "model_5":["mnist", {"hidden_size":512, "activation":"leaky",
                         "architecture":"fc2", "inference":"svi", "epochs":10, 
                         "lr":0.01, "n_samples":None, "warmup":None}],
              "model_6":["mnist", {"hidden_size":256, "activation":"leaky",
                         "architecture":"conv", "inference":"svi", "epochs":10, 
                         "lr":0.05, "n_samples":None, "warmup":None}],
              "model_7":["mnist", {"hidden_size":1024, "activation":"leaky",
                         "architecture":"fc2", "inference":"svi", "epochs":5, 
                         "lr":0.02, "n_samples":None, "warmup":None}],
              "model_8":["mnist", {"hidden_size":1024, "activation":"leaky",
                         "architecture":"conv", "inference":"svi", "epochs":10, 
                         "lr":0.02, "n_samples":None, "warmup":None}],
              "model_9":["fashion_mnist", {"hidden_size":512, "activation":"leaky",
                         "architecture":"fc", "inference":"hmc", "epochs":None,
                         "lr":None, "n_samples":100, "warmup":100}],
                         }


class BNN(PyroModule):

    def __init__(self, dataset_name, hidden_size, activation, architecture, inference, 
                 epochs, lr, n_samples, warmup, input_shape, output_size, 
                 step_size=0.005, num_steps=10):
        super(BNN, self).__init__()
        self.dataset_name = dataset_name
        self.inference = inference
        self.architecture = architecture
        self.epochs = epochs
        self.lr = lr
        self.n_samples = n_samples
        self.warmup = warmup
        self.step_size = step_size
        self.num_steps = num_steps
        self.basenet = NN(dataset_name=dataset_name, input_shape=input_shape, output_size=output_size, 
                        hidden_size=hidden_size, activation=activation, architecture=architecture, 
                        epochs=epochs, lr=lr)
        # print(self.basenet)
        self.name = self.get_name()

    def get_name(self, n_inputs=None):
        
        name = str(self.dataset_name)+"_bnn_"+str(self.inference)+"_hid="+\
               str(self.basenet.hidden_size)+"_act="+str(self.basenet.activation)+\
               "_arch="+str(self.basenet.architecture)

        if n_inputs:
            name = name+"_inp="+str(n_inputs)

        if self.inference == "svi":
            return name+"_ep="+str(self.epochs)+"_lr="+str(self.lr)
        elif self.inference == "hmc":
            return name+"_samp="+str(self.n_samples)+"_warm="+str(self.warmup)+\
                   "_stepsize="+str(self.step_size)+"_numsteps="+str(self.num_steps)

    def model(self, x_data, y_data):

        priors = {}
        for key, value in self.basenet.state_dict().items():
            loc = torch.zeros_like(value)
            scale = torch.ones_like(value)
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key):prior})

        lifted_module = pyro.random_module("module", self.basenet, priors)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            lhat = nnf.log_softmax(logits, dim=-1)
            obs = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

    def guide(self, x_data, y_data=None):

        dists = {}
        for key, value in self.basenet.state_dict().items():
            loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value)) 
            scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))
            distr = Normal(loc=loc, scale=softplus(scale))
            dists.update({str(key):distr})

        lifted_module = pyro.random_module("module", self.basenet, dists)()
        
        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            preds = nnf.softmax(logits, dim=-1)

        return preds

    def save(self, rel_path=TESTS, filename=None):

        if filename is None:
            filename = self.name+"_weights"

        path = rel_path + self.name +"/"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.inference == "svi":
            self.basenet.to("cpu")
            self.to("cpu")

            param_store = pyro.get_param_store()
            print("\nSaving: ", path + filename +".pt")
            print(f"\nlearned params = {param_store.get_all_param_names()}")
            param_store.save(path + filename +".pt")

        elif self.inference == "hmc":
            self.basenet.to("cpu")
            self.to("cpu")

            for key, value in self.posterior_predictive.items():
                torch.save(value.state_dict(), path+filename+"_"+str(key)+".pt")

                if DEBUG:
                    print(value.state_dict()["model.5.bias"])

    def load(self, device, rel_path=TESTS, filename=None):

        if filename is None:
            filename = self.name+"_weights"

        path = rel_path + self.name +"/"

        self.device=device
        self.basenet.device=device

        if self.inference == "svi":
            param_store = pyro.get_param_store()
            param_store.load(path + filename + ".pt")
            for key, value in param_store.items():
                param_store.replace_param(key, value.to(device), value)
            print("\nLoading ", path + filename + ".pt\n")

        elif self.inference == "hmc":

            self.posterior_predictive={}
            for model_idx in range(self.n_samples):
                net_copy = copy.deepcopy(self.basenet)
                net_copy.load_state_dict(torch.load(path+filename+"_"+str(model_idx)+".pt"))
                self.posterior_predictive.update({model_idx:net_copy})      

            if len(self.posterior_predictive)!=self.n_samples:
                raise AttributeError("wrong number of posterior models")

        self.to(device)
        self.basenet.to(device)

    def forward(self, inputs, n_samples=10, avg_posterior=False, seeds=None):

        if seeds:
            if len(seeds) != n_samples:
                raise ValueError("Number of seeds should match number of samples.")

        if self.inference == "svi":

            if avg_posterior is True:

                guide_trace = poutine.trace(self.guide).get_trace(inputs)  

                avg_state_dict = {}
                for key in self.basenet.state_dict().keys():
                    avg_weights = guide_trace.nodes[str(key)+"_loc"]['value']
                    avg_state_dict.update({str(key):avg_weights})

                self.basenet.load_state_dict(avg_state_dict)
                preds = [self.basenet.model(inputs)]

            else:

                preds = []  

                if seeds:
                    for seed in seeds:
                        pyro.set_rng_seed(seed)
                        guide_trace = poutine.trace(self.guide).get_trace(inputs)   
                        preds.append(guide_trace.nodes['_RETURN']['value'])

                else:

                    for _ in range(n_samples):
                        guide_trace = poutine.trace(self.guide).get_trace(inputs)   
                        preds.append(guide_trace.nodes['_RETURN']['value'])

                if DEBUG:
                    print("\nlearned variational params:\n")
                    print(pyro.get_param_store().get_all_param_names())
                    print(list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))
                    print("\n", pyro.get_param_store()["model.0.weight_loc"][0][:5])
                    print(guide_trace.nodes['module$$$model.0.weight']["fn"].loc[0][:5])
                    print("posterior sample: ", 
                      guide_trace.nodes['module$$$model.0.weight']['value'][5][0][0])

        elif self.inference == "hmc":

            preds = []
            posterior_predictive = list(self.posterior_predictive.values())

            if seeds is None:
                seeds = range(n_samples)

            for seed in seeds:
                net = posterior_predictive[seed]
                out = net.forward(inputs)
                out = nnf.softmax(out, dim=-1)
                preds.append(out)
    
        output_probs = torch.stack(preds).mean(0)
        return output_probs 

    def _train_hmc(self, train_loader, n_samples, warmup, step_size, num_steps, device, rel_path, filename):

        print("\n == HMC training ==")
        pyro.clear_param_store()

        num_batches = int(len(train_loader.dataset)/train_loader.batch_size)
        batch_samples = int(n_samples/num_batches)+1
        print("\nn_batches=",num_batches,"\tbatch_samples =", batch_samples)

        kernel = HMC(self.model, step_size=step_size, num_steps=num_steps)
        mcmc = MCMC(kernel=kernel, num_samples=batch_samples, warmup_steps=warmup, num_chains=1)

        start = time.time()

        # for x_batch, y_batch in train_loader:
        #     x_batch = x_batch.to(device)
        #     labels = y_batch.to(device).argmax(-1)
        #     mcmc.run(x_batch, labels)

        # self.posterior_predictive={}
        # posterior_samples = mcmc.get_samples(n_samples)
        # state_dict_keys = list(self.basenet.state_dict().keys())

        # if DEBUG:
        #     print("\n", list(posterior_samples.values())[-1])

        # for model_idx in range(n_samples):
        #     net_copy = copy.deepcopy(self.basenet)

        #     model_dict=OrderedDict({})
        #     for weight_idx, weights in enumerate(posterior_samples.values()):
        #         model_dict.update({state_dict_keys[weight_idx]:weights[model_idx]})
            
        #     net_copy.load_state_dict(model_dict)
        #     self.posterior_predictive.update({model_idx:net_copy})

        self.posterior_predictive={}
        state_dict_keys = list(self.basenet.state_dict().keys())

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            labels = y_batch.to(device).argmax(-1)
            mcmc.run(x_batch, labels)

            posterior_samples = mcmc.get_samples(batch_samples)

            for model_idx in range(batch_samples):
                net_copy = copy.deepcopy(self.basenet)

                model_dict=OrderedDict({})
                for weight_idx, weights in enumerate(posterior_samples.values()):
                    model_dict.update({state_dict_keys[weight_idx]:weights[model_idx]})
                
                net_copy.load_state_dict(model_dict)
                self.posterior_predictive.update({model_idx:net_copy})

        execution_time(start=start, end=time.time())     

        if DEBUG:
            print("\n", weights[model_idx]) 

        self.save(rel_path=rel_path, filename=filename)

    def _train_svi(self, train_loader, epochs, lr, device, rel_path, filename):
        self.device=device

        print("\n == SVI training ==")

        optimizer = pyro.optim.Adam({"lr":lr})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        loss_list = []
        accuracy_list = []

        start = time.time()
        for epoch in range(epochs):
            loss = 0.0
            correct_predictions = 0.0

            for x_batch, y_batch in train_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                labels = y_batch.argmax(-1)
                loss += svi.step(x_data=x_batch, y_data=labels)

                outputs = self.forward(x_batch, n_samples=10)
                predictions = outputs.argmax(dim=-1)
                correct_predictions += (predictions == labels).sum().item()
            
            if DEBUG:
                print("\n", pyro.get_param_store()["model.0.weight_loc"][0][:5])
                print("\n", predictions[:10], "\n", labels[:10])

            total_loss = loss / len(train_loader.dataset)
            accuracy = 100 * correct_predictions / len(train_loader.dataset)

            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}", 
                  end="\t")

            loss_list.append(loss)
            accuracy_list.append(accuracy)

        execution_time(start=start, end=time.time())
        self.save(rel_path=rel_path, filename=filename)

        plot_loss_accuracy(dict={'loss':loss_list, 'accuracy':accuracy_list},
                           path=rel_path+self.name+"/"+self.name+"_training.png")

    def train(self, train_loader, device, rel_path, filename=None):
        self.device=device
        self.basenet.device=device

        self.to(device)
        self.basenet.to(device)

        random.seed(0)
        pyro.set_rng_seed(0)

        if self.inference == "svi":
            self._train_svi(train_loader, self.epochs, self.lr, device, rel_path=rel_path, filename=filename)

        elif self.inference == "hmc":
            self._train_hmc(train_loader, self.n_samples, self.warmup,
                            self.step_size, self.num_steps, device=device, rel_path=rel_path, filename=filename)

    def evaluate(self, test_loader, device, n_samples=10, seeds_list=None):
        self.device=device
        self.basenet.device=device
        self.to(device)
        self.basenet.to(device)

        random.seed(0)
        pyro.set_rng_seed(0)

        bnn_seeds=list(range(n_samples)) if seeds_list is None else seeds_list

        with torch.no_grad():

            correct_predictions = 0.0
            for x_batch, y_batch in test_loader:

                x_batch = x_batch.to(device)
                outputs = self.forward(x_batch, n_samples=n_samples, seeds=bnn_seeds)
                predictions = outputs.argmax(-1)
                labels = y_batch.to(device).argmax(-1)
                correct_predictions += (predictions == labels).sum().item()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("Accuracy: %.2f%%" % (accuracy))
            return accuracy

def main(args):

    rel_path=DATA if args.savedir=="DATA" else TESTS

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    dataset, model = saved_BNNs["model_"+str(args.model_idx)]
    batch_size = 5000 if model["inference"] == "hmc" else 128

    train_loader, test_loader, inp_shape, out_size = \
                            data_loaders(dataset_name=dataset, batch_size=batch_size, 
                                         n_inputs=args.n_inputs, shuffle=True)
                        
    bnn = BNN(dataset, *list(model.values()), inp_shape, out_size)
   
    if args.train:
        bnn.train(train_loader=train_loader, device=args.device, rel_path=rel_path)
    else:
        bnn.load(device=args.device, rel_path=rel_path)

    if args.test:
        test_samples = 10

        print("\n== Evaluate on test data ==\n")
        bnn.evaluate(test_loader=test_loader, device=args.device, n_samples=test_samples)

        print(f"\n== Evaluate the first {test_samples} posterior samples ==\n")
        for seed in range(test_samples):
            bnn.evaluate(test_loader=test_loader, device=args.device, 
                n_samples=1, seeds_list=[seed])


if __name__ == "__main__":
    # assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", default=60000, type=int, help="number of input points")
    parser.add_argument("--model_idx", default=0, type=int, help="choose idx from saved_BNNs")
    parser.add_argument("--train", default=True, type=eval, help="train or load saved model")
    parser.add_argument("--test", default=True, type=eval, help="evaluate on test data")
    parser.add_argument("--savedir", default='DATA', type=str, help="choose dir for loading the BNN: DATA, TESTS")  
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
    main(args=parser.parse_args())
