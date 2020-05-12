import argparse
import os
from directories import *
from utils import *
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as nnf
import torch.optim as torchopt
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from nn import NN
from lossGradients import loss_gradients
from plot.gradients_components import stripplot_gradients_components


class GPbaseNN(NN):
    def __init__(self, dataset_name, input_shape, output_size, hidden_size):
        super(GPbaseNN, self).__init__(dataset_name, input_shape, output_size, hidden_size, 
                                       activation="leaky", architecture="fc2")

    def get_name(self, dataset_name, hidden_size, activation="leaky", architecture="fc2"):
        return str(dataset_name)+"_gp_base_nn_hid="+str(hidden_size)

    def set_model(self, architecture, activation, input_shape, output_size, hidden_size):

        input_size = input_shape[0]*input_shape[1]*input_shape[2]
        in_channels = input_shape[0]
        n_classes = output_size

        self.model = nn.Sequential(nn.Flatten(),
                                    nn.Linear(input_size, hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.LeakyReLU())
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = self.out(self.model(inputs))
        return nn.LogSoftmax(dim=-1)(x)


class GPRedBNN:

    def __init__(self, dataset_name, base_net):
        self.dataset_name = dataset_name
        self.net = base_net

    def get_filename(self, n_inputs):
        return str(self.dataset_name)+"_GPRedBNN_inp="+str(n_inputs)

    def train(self, x_train, y_train, device):
        
        x_train = torch.from_numpy(x_train).float().to(device)
        out_train = self.net.model(x_train)
        out_train=out_train.cpu().detach().numpy()
        start = time.time()
        gp = GaussianProcessClassifier(kernel=1.0 * RBF([1.0]), random_state=0,
                                       multi_class='one_vs_rest', n_jobs=10, copy_X_train=False)
        labels = y_train.argmax(-1)
        gp.fit(out_train, labels)
        execution_time(start=start, end=time.time())

        self.gp = gp
        print(out_train, self.gp.predict(out_train))
        self.save(gp=gp, n_inputs=len(x_train))

        # print(self.gp.get_params())
        return gp

    def evaluate(self, x_test, y_test, device):

        x_test = torch.from_numpy(x_test).float().to(device)
        out_test = self.net.model(x_test)
        out_test = out_test.cpu().detach().numpy()
        predictions = self.gp.predict(out_test)

        labels = y_test.argmax(-1)
        # print(self.gp.predict_proba(out_test)[:10])
        # print(predictions[:10], labels[:10])
        correct_predictions = np.sum((predictions == labels))

        accuracy = 100 * correct_predictions / len(x_test)
        print("\nTest accuracy = %.2f%%" % (accuracy))
        return accuracy

    def save(self, gp, n_inputs):
        name = self.get_filename(n_inputs=n_inputs)
        path = TESTS+self.net.name+"/"
        filename = name+".pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_to_pickle(gp, path, filename)
        self.name = name

    def load(self, n_inputs, rel_path):
        name = self.get_filename(n_inputs=n_inputs)
        self.gp = load_from_pickle(rel_path+self.net.name+"/"+name+".pkl")
        self.name = name


def main(args):

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    saved_models = {"half_moons":{"base_inputs":10000, "epochs":10, "lr":0.001, 
                                  "hidden_size":32, "gp_inputs":1000}}

    base_inputs, epochs, lr, hidden, gp_inputs = saved_models[args.dataset].values()
    
    # === base NN ===
    train_loader, test_loader, inp_shape, out_size = \
                            data_loaders(dataset_name=args.dataset, batch_size=64, 
                                         n_inputs=base_inputs, shuffle=True)

    nn = GPbaseNN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size, 
                  hidden_size=hidden)
    # nn.train(train_loader=train_loader, epochs=epochs, lr=lr, device=args.device)
    nn.load(epochs=epochs, lr=lr, device=args.device, rel_path=TESTS)
    nn.evaluate(test_loader=test_loader, device=args.device)

    # === LaplaceRedBNN ===

    x_train, y_train, x_test, y_test, _, _ = \
                            load_dataset(args.dataset, n_inputs=gp_inputs, shuffle=True)

    gp = GPRedBNN(dataset_name=args.dataset, base_net=nn)

    # gp.train(x_train=x_train, y_train=y_train, device=args.device)
    gp.load(n_inputs=args.inputs, rel_path=TESTS)
    gp.evaluate(x_test=x_test, y_test=y_test, device=args.device)
    exit()
    n_samples_list=[1, 50, 100, 500]
    loss_gradients_list=[]
    for posterior_samples in n_samples_list:

        loss_gradients(net=gp, n_samples=posterior_samples, savedir=gp.name+"/", 
                        dataseta_loader=test_loader, device=args.device, filename=gp.name)
        # loss_gradients = load_loss_gradients(n_samples=posterior_samples, filename=gp.name, 
        #                                      relpath=TESTS, savedir=gp.name+"/")

        loss_gradients_list.append(loss_gradients)

    stripplot_gradients_components(loss_gradients_list=loss_gradients_list, 
        n_samples_list=n_samples_list, dataset_name=args.dataset, filename=gp.name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--inputs", default=1000, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, 
                        help="mnist, fashion_mnist, cifar, half_moons")   
    parser.add_argument("--hidden_size", default=16, type=int, help="power of 2")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
    main(args=parser.parse_args())