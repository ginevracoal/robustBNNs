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
from pyro.distributions import Normal, HalfCauchy
from reducedBNN import redBNN, NN
from pyro.contrib.autoguide import AutoLaplaceApproximation
from pyro.nn import PyroModule

saved_LaplaceRedBNNs = {}


class LaplaceRedBNN(redBNN):

    def __init__(self, dataset_name, input_shape, output_size, base_net):
        super(LaplaceRedBNN, self).__init__(dataset_name, input_shape, output_size, 
                                            "svi", base_net)
        self.dataset_name = dataset_name
        self.net = base_net

    def model(self, x_data, y_data):
        net = self.net

        outw_prior = Normal(loc=torch.zeros_like(net.out.weight), 
                            scale=torch.ones_like(net.out.weight))
        outb_prior = Normal(loc=torch.zeros_like(net.out.bias), 
                            scale=torch.ones_like(net.out.bias))

        outw = pyro.sample("w", outw_prior)
        outb = pyro.sample("b", outb_prior)

        yhat = torch.matmul(x_data, outw.t()) + outb 
        lhat = F.log_softmax(yhat, dim=-1)
        # print(yhat.shape, lhat.shape, y_data.shape)
        # exit()
        sigma = pyro.sample("sigma", HalfCauchy(2))
        obs = pyro.sample("obs", Normal(lhat, sigma), obs=y_data)

    def guide(self, x_data, y_data=None):
        delta_guide = AutoLaplaceApproximation(self.model)
        return delta_guide

    def _train_svi(self, train_loader, hyperparams, device):

        epochs, lr = hyperparams["epochs"], hyperparams["lr"]

        optimizer = pyro.optim.Adam({"lr":lr})
        elbo = Trace_ELBO()
        delta_guide = AutoLaplaceApproximation(self.model).to(device)
        svi = SVI(self.model, delta_guide, optimizer, loss=elbo)

        start = time.time()
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0.0
            accuracy = 0.0
            total = 0.0

            n_inputs = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                out_batch = self.net.l2(self.net.l1(x_batch))
                loss = svi.step(x_data=out_batch, y_data=y_batch)

                self.guide = delta_guide
                # self.guide = delta_guide.laplace_approximation(out_batch, y_batch)
                
                n_inputs += len(x_batch)
                outputs = self.forward(x_batch)
                predictions = outputs.argmax(dim=1)
                labels = y_batch.argmax(-1)
                total += y_batch.size(0)
                total_loss += loss / len(train_loader.dataset)
                correct_predictions += (predictions == labels).sum()
                accuracy = 100 * correct_predictions / len(train_loader.dataset)

            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
                  end="\t")

        execution_time(start=start, end=time.time())

        hyperparams = {"epochs":epochs, "lr":lr}    
        self.save(n_inputs=len(train_loader.dataset), hyperparams=hyperparams)
 
    def train(self, train_loader, epochs, lr, device):
        hyperparams = {"epochs":epochs, "lr":lr}
        super(LaplaceRedBNN, self).train(train_loader=train_loader, 
                                         hyperparams=hyperparams, device=device)

    def forward(self, inputs, n_samples=10):
        out_batch = self.net.l2(self.net.l1(inputs))
        predictive = Predictive(model=self.model, guide=self.guide, num_samples=n_samples,
                                return_sites=("w","b","obs"))
        preds = predictive(out_batch, None)["obs"]
        return preds.mean(0)


def main(args):
    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # === load dataset ===
    train_loader, test_loader, inp_shape, out_size = \
                            data_loaders(dataset_name=args.dataset, batch_size=64, 
                                         n_inputs=args.inputs, shuffle=True)
    # === base NN ===

    dataset, epochs, lr, rel_path = ("mnist", 20, 0.001, DATA)  

    nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size)
    # nn.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, device=args.device)
    nn.load(epochs=epochs, lr=lr, device=args.device, rel_path=rel_path)
    # nn.evaluate(test_loader=test_loader, device=args.device)

    # === LaplaceRedBNN ===

    # init = saved_LaplaceRedBNN[args.dataset]
                            
    bnn = LaplaceRedBNN(dataset_name=args.dataset, input_shape=inp_shape, 
                        output_size=out_size, base_net=nn)

    bnn.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, device=args.device)
    bnn.evaluate(test_loader=test_loader, device=args.device)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--inputs", default=1000, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, 
                        help="mnist, fashion_mnist, cifar")
    parser.add_argument("--activation", default="leaky", type=str, 
                        help="relu, leaky, sigm, tanh")
    parser.add_argument("--architecture", default="conv", type=str, help="conv, fc, fc2")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
   
    main(args=parser.parse_args())