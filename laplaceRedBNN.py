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
from pyro.distributions import Normal, Categorical, HalfCauchy
from reducedBNN import redBNN, NN
from pyro.contrib.autoguide import AutoLaplaceApproximation
from pyro.nn import PyroModule
from lossGradients import loss_gradients
from plot.gradients_components import stripplot_gradients_components

saved_baseNNs = {"mnist": (20, 0.001)}
saved_LaplaceRedBNNs = {"mnist": (60000, 23, 0.001)}


class LaplaceRedBNN(redBNN):
    # pyro implementation

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

        # sigma = pyro.sample("sigma", HalfCauchy(2))
        # obs = pyro.sample("obs", Normal(lhat, sigma), obs=y_data)

        cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
        return cond_model

    def _train_svi(self, train_loader, hyperparams, device):

        epochs, lr = hyperparams["epochs"], hyperparams["lr"]

        optimizer = pyro.optim.Adam({"lr":lr})
        elbo = Trace_ELBO()
        delta_guide = AutoLaplaceApproximation(self.model)
        svi = SVI(self.model, delta_guide, optimizer, loss=elbo)

        start = time.time()


        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                labels = y_batch.argmax(-1)

                out_batch = self.net.l2(self.net.l1(x_batch))
                loss = svi.step(x_data=out_batch, y_data=labels)
                total_loss += loss

                self.delta_guide = delta_guide.laplace_approximation(out_batch, labels)
                
            total_loss = total_loss / len(train_loader.dataset)
            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f}", end="\t")
            # outputs = self.forward(x_batch)
            # predictions = outputs.argmax(dim=1)

            # correct_predictions += (predictions == labels).sum()
            # accuracy = 100 * correct_predictions / len(train_loader.dataset)
    
            # only on the last batch

        execution_time(start=start, end=time.time())
        self.save(n_inputs=len(train_loader.dataset), epochs=epochs, lr=lr)

    def train(self, train_loader, epochs, lr, device):
        hyperparams = {"epochs":epochs, "lr":lr}
        super(LaplaceRedBNN, self).train(train_loader=train_loader, 
                                         hyperparams=hyperparams, device=device)

    def forward(self, inputs, n_samples=10):

        out_batch = self.net.l2(self.net.l1(inputs))
        predictive = Predictive(model=self.model, guide=self.delta_guide, 
                                num_samples=n_samples, return_sites=("w","b"))
        out_w = predictive(out_batch, None)["w"].mean(0)
        out_b = predictive(out_batch, None)["b"].mean(0)

        # print(predictive(out_batch, None))
        # print("sampled_bias =", predictive(out_batch, None)["b"])
        yhat = torch.matmul(out_batch, out_w.t()) + out_b
        preds = F.softmax(yhat, dim=-1)
        return preds

    def save(self, n_inputs, epochs, lr):
        name = self.get_filename(n_inputs=n_inputs, epochs=epochs, lr=lr)
        path = TESTS+name+"/"
        filename = name+"_guide.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("\nSaving: ", path+filename)
        save_to_pickle(self.delta_guide, path, filename)
        self.name = name

    def load(self, n_inputs, epochs, lr, device, rel_path):
        name = self.get_filename(n_inputs=n_inputs, epochs=epochs, lr=lr)
        self.delta_guide = load_from_pickle(rel_path+name+"/"+name+"_guide.pkl")
        self.name = name


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
    inputs, epochs, lr = (args.inputs, args.epochs, args.lr)
    # inputs, epochs, lr = saved_LaplaceRedBNNs[args.dataset]
                            
    bnn = LaplaceRedBNN(dataset_name=args.dataset, input_shape=inp_shape, 
                        output_size=out_size, base_net=nn)

    bnn.train(train_loader=train_loader, epochs=epochs, lr=lr, device=args.device)
    # bnn.load(n_inputs=inputs, epochs=epochs, lr=lr, device=args.device, rel_path=DATA)

    n_samples_list=[1, 50, 100, 500]
    loss_gradients_list=[]
    for posterior_samples in n_samples_list:
        bnn.evaluate(test_loader=test_loader, device=args.device, n_samples=posterior_samples)

        loss_gradients(net=bnn, n_samples=posterior_samples, savedir=bnn.name+"/", 
                        dataseta_loader=test_loader, device=args.device, filename=bnn.name)
        # loss_gradients = load_loss_gradients(n_samples=posterior_samples, filename=bnn.name, 
        #                                      relpath=TESTS, savedir=bnn.name+"/")

        loss_gradients_list.append(loss_gradients)

    stripplot_gradients_components(loss_gradients_list=loss_gradients_list, 
        n_samples_list=n_samples_list, dataset_name=args.dataset, filename=bnn.name)



if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--inputs", default=1000, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, 
                        help="mnist, fashion_mnist, cifar")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
   
    main(args=parser.parse_args())