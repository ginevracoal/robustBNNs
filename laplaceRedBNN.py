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

saved_baseNNs = {"mnist": (20, 0.001)}
saved_LaplaceRedBNNs = {"mnist": (60000, 23, 0.001)}


class LaplaceRedBNN(redBNN):

    def __init__(self, dataset_name, input_shape, output_size, base_net):
        super(LaplaceRedBNN, self).__init__(dataset_name, input_shape, output_size, 
                                            "svi", base_net)
        self.dataset_name = dataset_name
        self.net = base_net

    def get_filename(self, n_inputs, epochs, lr):
        return str(self.dataset_name)+"_LaplaceRedBNN_inp="+str(n_inputs)+"_ep="+\
               str(epochs)+"_lr="+str(lr)

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

    def _train_svi(self, train_loader, hyperparams, device):

        epochs, lr = hyperparams["epochs"], hyperparams["lr"]

        optimizer = pyro.optim.Adam({"lr":lr})
        elbo = Trace_ELBO()
        self.delta_guide = AutoLaplaceApproximation(self.model)
        svi = SVI(self.model, self.delta_guide, optimizer, loss=elbo)

        start = time.time()
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                out_batch = self.net.l2(self.net.l1(x_batch))
                loss = svi.step(x_data=out_batch, y_data=y_batch)
                outputs = self.forward(x_batch)
                predictions = outputs.argmax(dim=1)
                labels = y_batch.argmax(-1)

                correct_predictions += (predictions == labels).sum()
                total_loss += loss

            total_loss = total_loss / len(train_loader.dataset)
            accuracy = 100 * correct_predictions / len(train_loader.dataset)
            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
                  end="\t")

        execution_time(start=start, end=time.time())
        self.save(n_inputs=len(train_loader.dataset), epochs=epochs, lr=lr)
        # hyperparams = {"epochs":epochs, "lr":lr}    
        # self.save(n_inputs=len(train_loader.dataset), hyperparams=hyperparams)
 
    def train(self, train_loader, epochs, lr, device):
        hyperparams = {"epochs":epochs, "lr":lr}
        super(LaplaceRedBNN, self).train(train_loader=train_loader, 
                                         hyperparams=hyperparams, device=device)

    def forward(self, inputs, n_samples=20):
        out_batch = self.net.l2(self.net.l1(inputs))
        predictive = Predictive(model=self.model, guide=self.delta_guide, 
                                num_samples=n_samples, return_sites=("w","b","obs"))
        w = predictive(out_batch, None)["w"].mean(0)
        b = predictive(out_batch, None)["b"].mean(0)
        yhat = torch.matmul(out_batch, w.t()) + b 
        preds = F.log_softmax(yhat, dim=-1)
        return preds

    def save(self, n_inputs, epochs, lr):
        name = self.get_filename(n_inputs=n_inputs, epochs=epochs, lr=lr)
        path = TESTS+name+"/"
        filename = name+"_guide.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("\nSaving: ", path+filename)
        save_to_pickle(self.delta_guide, path, filename)

    def load(self, n_inputs, epochs, lr, device, rel_path):
        # hyperparams = {"epochs":epochs, "lr":lr}
        # super(LaplaceRedBNN, self).load(n_inputs=n_inputs, hyperparams=hyperparams, 
        #                                 device=device, rel_path=rel_path)
        name = self.get_filename(n_inputs=n_inputs, epochs=epochs, lr=lr)
        self.delta_guide = load_from_pickle(TESTS+name+"/"+name+"_guide.pkl")

def main(args):
    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # === load dataset ===
    train_loader, test_loader, inp_shape, out_size = \
                            data_loaders(dataset_name=args.dataset, batch_size=128, 
                                         n_inputs=args.inputs, shuffle=True)
    # === base NN ===
    epochs, lr = saved_baseNNs[args.dataset]

    nn = NN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size)
    # nn.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, device=args.device)
    nn.load(epochs=epochs, lr=lr, device=args.device, rel_path=DATA)
    # nn.evaluate(test_loader=test_loader, device=args.device)

    # === LaplaceRedBNN ===
    # inputs, epochs, lr = (args.inputs, args.epochs, args.lr)
    inputs, epochs, lr = saved_LaplaceRedBNNs[args.dataset]
                            
    bnn = LaplaceRedBNN(dataset_name=args.dataset, input_shape=inp_shape, 
                        output_size=out_size, base_net=nn)

    # bnn.train(train_loader=train_loader, epochs=epochs, lr=lr, device=args.device)
    bnn.load(n_inputs=inputs, epochs=epochs, lr=lr, device=args.device, rel_path=DATA)
    bnn.evaluate(test_loader=test_loader, device=args.device)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--inputs", default=1000, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, 
                        help="mnist, fashion_mnist, cifar")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
   
    main(args=parser.parse_args())