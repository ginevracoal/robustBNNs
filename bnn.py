import argparse
import os
from directories import *
from utils import *
import pyro
import torch
from torch import nn
import torch.nn.functional as nnf
import numpy as np
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
import torch.optim as torchopt
from pyro import poutine
import pyro.optim as pyroopt
import torch.nn.functional as F
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.infer.mcmc.util import predictive
from pyro.infer.abstract_infer import TracePredictive
from pyro.distributions import OneHotCategorical, Normal, Categorical
from nn import NN
softplus = torch.nn.Softplus()

DEBUG = False


class BNN(nn.Module):

    def __init__(self, dataset_name, input_shape, output_size, hidden_size, activation, 
                 architecture, inference):
        super(BNN, self).__init__()
        self.dataset_name = dataset_name
        self.inference = inference
        self.architecture = architecture
        self.net = NN(dataset_name=dataset_name, input_shape=input_shape, output_size=output_size, 
                      hidden_size=hidden_size, activation=activation, architecture=architecture)

    def get_hyperparams(self, args):

        if self.inference == "svi":
            return {"epochs":args.epochs, "lr":args.lr}

        elif self.inference == "hmc":
            return {"hmc_samples":args.hmc_samples, "warmup":args.warmup}

    def get_name(self, hyperparams):
        
        name = str(self.dataset_name)+"_bnn_"+str(self.inference)+"_hid="+\
               str(self.net.hidden_size)+"_act="+str(self.net.activation)+\
               "_arch="+str(self.net.architecture)

        if self.inference == "svi":
            return name+"_ep="+str(hyperparams["epochs"])+"_lr="+str(hyperparams["lr"])

        elif self.inference == "hmc":
            return name+"_samp="+str(hyperparams["hmc_samples"])+"_warm="+str(hyperparams["warmup"])

    def model(self, x_data, y_data):

        state_dict = self.net.state_dict()

        priors = {}
        for key, value in state_dict.items():
            prior = Normal(loc=torch.zeros_like(value), scale=torch.ones_like(value))
            priors.update({str(key):prior})

        if DEBUG:
            print(state_dict.keys())
            print("\n", priors)

        # if self.architecture == "fc":
        #     l1w_prior = Normal(loc=torch.zeros_like(net_dict["l1.1.weight"]), 
        #                         scale=torch.ones_like(net.l1.weight))
        #     l1b_prior = Normal(loc=torch.zeros_like(net.l1.bias), 
        #                         scale=torch.ones_like(net.l1.bias))
        #     outw_prior = Normal(loc=torch.zeros_like(net.out.weight), 
        #                         scale=torch.ones_like(net.out.weight))
        #     outb_prior = Normal(loc=torch.zeros_like(net.out.bias), 
        #                         scale=torch.ones_like(net.out.bias))
        #     priors = {'l1': l1w_prior, 'out.bias': l1b_prior,
        #               'out.weight': outw_prior, 'out.bias': outb_prior}

        # if self.architecture == "fc2":
        #     x = self.l1(inputs)
        #     x = self.l2(inputs)
        #     return self.out(x)

        # if self.architecture == "conv":
        #     x = self.conv1(inputs)
        #     x = self.conv2(inputs)
        #     return self.out(x)
        
        lifted_module = pyro.random_module("module", self.net, priors)()
        lhat = F.log_softmax(lifted_module(x_data), dim=-1)
        cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
        return cond_model

    def guide(self, x_data, y_data=None):

        state_dict = self.net.state_dict()

        priors = {}
        for key, value in state_dict.items():
            prior = Normal(loc=torch.zeros_like(value), scale=torch.ones_like(value))
            priors.update({str(key):prior})

        if DEBUG:
            print(state_dict.keys())
            print("\n", priors.keys())

        # if self.architecture == "fc":

            # l1w_mu = torch.randn_like(net_dict["l1.1.weight"])
            # l1w_sigma = torch.randn_like(net_dict["l1.1.weight"])
            # l1w_mu_param = pyro.param("l1w_mu", l1w_mu)
            # l1w_sigma_param = softplus(pyro.param("l1w_sigma", l1w_sigma))
            # l1w_prior = Normal(loc=l1w_mu_param, scale=l1w_sigma_param).independent(1)

            # l1b_mu = torch.randn_like(net_dict["l1.1.bias"])
            # l1b_sigma = torch.randn_like(net_dict["l1.1.bias"])
            # l1b_mu_param = pyro.param("l1b_mu", l1b_mu)
            # l1b_sigma_param = softplus(pyro.param("l1b_sigma", l1b_sigma))
            # l1b_prior = Normal(loc=l1b_mu_param, scale=l1b_sigma_param).independent(1)

            # outw_mu = torch.randn_like(net_dict["out.0.weight"])
            # outw_sigma = torch.randn_like(net_dict["out.0.weight"])
            # outw_mu_param = pyro.param("outw_mu", outw_mu)
            # outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
            # outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)

            # outb_mu = torch.randn_like(net_dict["out.0.bias"])
            # outb_sigma = torch.randn_like(net_dict["out.0.bias"])
            # outb_mu_param = pyro.param("outb_mu", outb_mu)
            # outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
            # outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param).independent(1)

            # priors = {'l1': l1w_prior, 'out.bias': l1b_prior,
            #           'out.weight': outw_prior, 'out.bias': outb_prior}

        lifted_module = pyro.random_module("module", self.net, priors)()
        logits = F.log_softmax(lifted_module(x_data), dim=-1)
        return logits
 
    def save(self, hyperparams):

        name = self.get_name(hyperparams)
        path = TESTS + name +"/"
        filename = name+"_weights"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.inference == "svi":
            self.net.to("cpu")
            self.to("cpu")
            param_store = pyro.get_param_store()
            print("\nSaving: ", path + filename +".pt")
            print(f"\nlearned params = {param_store.get_all_param_names()}")
            param_store.save(path + filename +".pt")

        elif self.inference == "hmc":
            self.net.to("cpu")
            self.to("cpu")
            save_to_pickle(data=self.posterior_samples, path=path, filename=filename+".pkl")

    def load(self, n_inputs, hyperparams, device, rel_path=TESTS):

        name = self.get_name(hyperparams)
        path = rel_path + name +"/"
        filename = name+"_weights"

        if self.inference == "svi":
            param_store = pyro.get_param_store()
            param_store.load(path + filename + ".pt")
            print("\nLoading ", path + filename + ".pt\n")

        elif self.inference == "hmc":
            posterior_samples = load_from_pickle(path + filename + ".pkl")
            self.posterior_samples = posterior_samples
            print("\nLoading ", path + filename + ".pkl\n")

        self.to(device)
        self.net.to(device)

    def forward(self, inputs, n_samples=100):

        if self.inference == "svi":

            preds = []
            for i in range(n_samples):
                # pyro.set_rng_seed(i)
                guide_trace = poutine.trace(self.guide).get_trace(inputs)           
                preds.append(guide_trace.nodes['_RETURN']['value'])

            if DEBUG:
                print(list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))
                print(guide_trace.nodes['module$$$l1.1.weight']['value'][0,0:5])

        elif self.inference == "hmc":

            if DEBUG:
                print("\nself.net.state_dict() keys = ", self.net.state_dict().keys())

            for key, value in state_dict.items():
                prior = Normal(loc=torch.zeros_like(value), scale=torch.ones_like(value))
                priors.update({str(key):prior})

                preds = []
                n_samples = min(n_samples, len(self.posterior_samples[0]))
                for i in range(n_samples):

                    for key, value in self.net.state_dict().items():
                        out = self.posterior_samples[key][i]
                        self.net.state_dict().update({str(key):out})

                    preds.append(self.net.forward(inputs))
        
        print(f"{n_samples} post samp", end="\t")
        pred = torch.stack(preds, dim=0)
        return pred.mean(0) 

    def _train_hmc(self, train_loader, hyperparams, device):
        print("\n == redBNN HMC training ==")

        num_samples, warmup_steps = (hyperparams["hmc_samples"], hyperparams["warmup"])

        # kernel = HMC(self.model, step_size=0.0855, num_steps=4)
        kernel = NUTS(self.model)
        batch_samples = int(num_samples*train_loader.batch_size/len(train_loader.dataset))+1
        print("\nSamples per batch =", batch_samples)
        hmc = MCMC(kernel=kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)

        start = time.time()

        out_weight, out_bias = ([],[])
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).argmax(-1)
            hmc.run(x_batch, y_batch)

            # out_weight.append(hmc.get_samples()["module$$$out.weight"])
            # out_bias.append(hmc.get_samples()["module$$$out.bias"])

        execution_time(start=start, end=time.time())

        # out_weight, out_bias = (torch.cat(out_weight), torch.cat(out_bias))
        # self.posterior_samples = {"module$$$out.weight":out_weight, "module$$$out.bias":out_bias}

        self.save(hyperparams=hyperparams)
        return self.posterior_samples

    def _train_svi(self, train_loader, hyperparams, device):
        print("\n == redBNN SVI training ==")

        epochs, lr = hyperparams["epochs"], hyperparams["lr"]

        optimizer = pyro.optim.Adam({"lr":lr})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        start = time.time()
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0.0
            accuracy = 0.0
            total = 0.0

            n_inputs = 0
            for x_batch, y_batch in train_loader:
                n_inputs += len(x_batch)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).argmax(-1)
                total += y_batch.size(0)

                outputs = self.forward(x_batch).to(device)
                loss = svi.step(x_data=x_batch, y_data=y_batch)

                predictions = outputs.argmax(dim=1)
                total_loss += loss / len(train_loader.dataset)
                correct_predictions += (predictions == y_batch).sum()
                accuracy = 100 * correct_predictions / len(train_loader.dataset)

            # if DEBUG:
            #   print("\nmodule$$$l2.0.weight should be fixed:\n",
            #         pyro.get_param_store()["module$$$l2.0.weight"][0][0][:3])
            #   print("\noutw_mu should change:\n", pyro.get_param_store()["outw_mu"][:3])

            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
                  end="\t")

        execution_time(start=start, end=time.time())

        hyperparams = {"epochs":epochs, "lr":lr}    
        self.save(hyperparams=hyperparams)

    def train(self, train_loader, hyperparams, device):
        self.to(device)
        self.net.to(device)
        random.seed(0)
        pyro.set_rng_seed(0)

        if self.inference == "svi":
            self._train_svi(train_loader, hyperparams, device)

        elif self.inference == "hmc":
            self._train_hmc(train_loader, hyperparams, device)

    def evaluate(self, test_loader, device):
        self.to(device)
        self.net.to(device)
        random.seed(0)
        pyro.set_rng_seed(0)

        with torch.no_grad():

            correct_predictions = 0.0

            for x_batch, y_batch in test_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).argmax(-1)
                outputs = self.forward(x_batch)
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == y_batch).sum()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("\nAccuracy: %.2f%%" % (accuracy))
            return accuracy


def main(args):

    # === load dataset ===
    train_loader, test_loader, inp_shape, out_size = \
                            data_loaders(dataset_name=args.dataset, batch_size=128, 
                                         n_inputs=args.inputs, shuffle=True)

    data, ep, lr, hid, act, arc, inf = (args.dataset, args.epochs, args.lr, args.hidden_size,
                                        args.activation, args.architecture, args.inference)       
    bnn = BNN(dataset_name=data, input_shape=inp_shape, output_size=out_size, 
              hidden_size=hid, activation=act, architecture=arc, inference=inf)

    hyperparams = bnn.get_hyperparams(args)
    bnn.train(train_loader=train_loader, hyperparams=hyperparams, device=args.device)
    # bnn.load(n_inputs=args.inputs, hyperparams=hyperparams, device=args.device, rel_path=TESTS)
    bnn.evaluate(test_loader=test_loader, device=args.device)
    

if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="BNN")

    parser.add_argument("--inputs", default=100, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist, fashion_mnist, cifar")
    parser.add_argument("--hidden_size", default=64, type=int, help="power of 2")
    parser.add_argument("--activation", default="leaky", type=str, help="leaky, sigm, tanh")
    parser.add_argument("--architecture", default="conv", type=str, help="conv, fc, fc2")
    parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--hmc_samples", default=5, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
   
    main(args=parser.parse_args())