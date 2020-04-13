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
from utils import plot_loss_accuracy
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
            return {"n_samples":args.samples, "warmup":args.warmup}

    def get_name(self, hyperparams):
        
        name = str(self.dataset_name)+"_bnn_"+str(self.inference)+"_hid="+\
               str(self.net.hidden_size)+"_act="+str(self.net.activation)+\
               "_arch="+str(self.net.architecture)

        if self.inference == "svi":
            name = name+"_ep="+str(hyperparams["epochs"])+"_lr="+str(hyperparams["lr"])

        elif self.inference == "hmc":
            name = name+"_samp="+str(hyperparams["n_samples"])+"_warm="+str(hyperparams["warmup"])

        self.name = name
        return name

    def model(self, x_data, y_data):

        state_dict = self.net.state_dict()
        priors = {}
        for key, value in state_dict.items():
            prior = Normal(loc=torch.zeros_like(value), scale=torch.ones_like(value))
            priors.update({str(key):prior})

        # if DEBUG:
        #     print(state_dict.keys())
        #     print("\n", priors)
        
        lifted_module = pyro.random_module("module", self.net, priors)()
        lhat = F.log_softmax(lifted_module(x_data), dim=-1)
        cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

        return cond_model

    def guide(self, x_data, y_data=None):

        state_dict = self.net.state_dict()

        priors = {}
        for key, value in state_dict.items():
            loc = pyro.param(str(f"{key}_loc"), 0.01*torch.randn_like(value))
            scale = softplus(pyro.param(str(f"{key}_scale"), 0.1*torch.randn_like(value)))
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key):prior})

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

    def load(self, hyperparams, device, rel_path=TESTS):

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

    # def predict(self, inputs, n_samples=10):
    #     sampled_models = [self.guide(None, None) for _ in range(n_samples)]
    #     outputs = [self.model(inputs).data for model in sampled_models]
    #     mean = outputs.mean(0)
    #     return mean.argmax(-1)

    def forward(self, inputs):
        return self.net.forward(inputs)

    def predict(self, inputs, n_samples=10):
        # random.seed(0)
        # pyro.set_rng_seed(0)

        if self.inference == "svi":

            preds = []
            for i in range(n_samples):
                guide_trace = poutine.trace(self.guide).get_trace(inputs)   
                preds.append(guide_trace.nodes['_RETURN']['value'])

            if DEBUG:
                print(list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))

        elif self.inference == "hmc":

            preds = []
            for key, value in self.net.state_dict().items():
                for i in range(n_samples):

                    weights = self.posterior_samples[str(f"module$$${key}_{i}")]
                    self.net.state_dict().update({str(key):weights})

                    preds.append(self.net.forward(inputs))
    
        pred = torch.stack(preds, dim=0)
        return pred.mean(0) 

    def _train_hmc(self, train_loader, hyperparams, device):
        print("\n == HMC training ==")

        n_samples, warmup_steps = (hyperparams["n_samples"], hyperparams["warmup"])

        kernel = HMC(self.model, step_size=0.0855, num_steps=4)
        batch_samples = int(n_samples*train_loader.batch_size/len(train_loader.dataset))+1
        print("\nSamples per batch =", batch_samples, ", Total samples =", n_samples)
        mcmc = MCMC(kernel=kernel, num_samples=n_samples, warmup_steps=warmup_steps, num_chains=1)

        start = time.time()
        sample_idx = 0
        posterior_samples = {}

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).argmax(-1)
            mcmc.run(x_batch, y_batch)
            
            for idx in range(batch_samples):
                for k, v in mcmc.get_samples().items():
                    posterior_samples.update({f"{k}_{sample_idx}":v[idx]})
                
                sample_idx += 1
                       
        execution_time(start=start, end=time.time())

        if DEBUG:
            print(posterior_samples.keys())

        self.posterior_samples = posterior_samples
        self.save(hyperparams=hyperparams)

    def _train_svi(self, train_loader, hyperparams, device):
        print("\n == SVI training ==")

        epochs, lr = hyperparams["epochs"], hyperparams["lr"]

        optimizer = pyro.optim.Adam({"lr":lr})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        loss_list = []
        accuracy_list = []

        start = time.time()
        for epoch in range(epochs):
            loss = 0.0
            correct_predictions = 0.0
            accuracy = 0.0

            n_inputs = 0
            for x_batch, y_batch in train_loader:
                n_inputs += len(x_batch)
                x_batch = x_batch.to(device)
                labels = y_batch.to(device).argmax(-1)
                loss += svi.step(x_data=x_batch, y_data=labels)

                outputs = self.predict(x_batch).to(device)
                predictions = outputs.argmax(dim=-1)
                correct_predictions += (predictions == labels).sum()
            
            loss = loss / len(train_loader.dataset)
            accuracy = 100 * correct_predictions / len(train_loader.dataset)

            if DEBUG:
                print(pyro.get_param_store().get_all_param_names())
                print(outputs[:3][:5])
                print(pyro.get_param_store()["model.0.weight_loc"][0][:5])

            print(f"\n[Epoch {epoch + 1}]\t loss: {loss:.8f} \t accuracy: {accuracy:.2f}", 
                  end="\t")

            loss_list.append(loss)
            accuracy_list.append(accuracy)

        execution_time(start=start, end=time.time())

        hyperparams = {"epochs":epochs, "lr":lr}    
        self.save(hyperparams=hyperparams)

        plot_loss_accuracy(dict={'loss':loss_list, 'accuracy':accuracy_list},
                           path=TESTS+self.name+"/"+self.name+"_training.png")

    def train(self, train_loader, hyperparams, device):
        self.to(device)
        self.net.to(device)
        random.seed(0)
        pyro.set_rng_seed(0)

        if self.inference == "svi":
            self._train_svi(train_loader, hyperparams, device)

        elif self.inference == "hmc":
            self._train_hmc(train_loader, hyperparams, device)

    def evaluate(self, test_loader, device, n_samples=100):
        self.to(device)
        self.net.to(device)
        random.seed(0)
        pyro.set_rng_seed(0)

        with torch.no_grad():

            correct_predictions = 0.0

            for x_batch, y_batch in test_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).argmax(-1)
                outputs = self.predict(x_batch, n_samples=n_samples)
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == y_batch).sum()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("\nAccuracy: %.2f%%" % (accuracy))
            return accuracy


def main(args):

    train_loader, test_loader, inp_shape, out_size = \
                            data_loaders(dataset_name=args.dataset, batch_size=64, 
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
    parser.add_argument("--hidden_size", default=32, type=int, help="power of 2")
    parser.add_argument("--activation", default="leaky", type=str, help="leaky, sigm, tanh")
    parser.add_argument("--architecture", default="conv", type=str, help="conv, fc, fc2")
    parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--samples", default=10, type=int)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
   
    main(args=parser.parse_args())