import argparse
import os
from directories import *
from utils import *
import numpy as np
import pandas 
import random
import torch
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as nnf
import torch.optim as torchopt
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from nn import NN
from lossGradients import loss_gradients
from plot.gradients_components import stripplot_gradients_components
from adversarialAttacks import attack, load_attack, attack_evaluation


saved_GPbaseNNs = {"half_moons":{"base_inputs":10000, "epochs":10, "lr":0.001, "hidden_size":32, 
                                 "gp_inputs":2000}, 
                   "mnist":{"base_inputs":30000, "epochs":20, "lr":0.001, "hidden_size":64, 
                            "gp_inputs":5000}}

class GPbaseNN(NN):
    def __init__(self, dataset_name, input_shape, output_size, hidden_size, epochs, lr):
        super(GPbaseNN, self).__init__(dataset_name, input_shape, output_size, hidden_size,
                                        activation="leaky", architecture="fc")
        self.criterion = nn.CrossEntropyLoss()
        self.dataset_name = dataset_name
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs
        self.lr = lr
        self.savedir = str(dataset_name)+"_GPRedBNN_hid="+str(hidden_size)+\
                            "_ep="+str(self.epochs)+"_lr="+str(self.lr)
        self.name = str(dataset_name)+"_GPNbaseNN_hid="+str(hidden_size)+\
                            "_ep="+str(self.epochs)+"_lr="+str(self.lr)

    def train(self, train_loader, device):
        super(GPbaseNN, self).train(train_loader, self.epochs, self.lr, device)

    def save(self):
        os.makedirs(os.path.dirname(TESTS+self.savedir+"/"), exist_ok=True)
        print("\nSaving: ", TESTS+self.savedir+"/"+self.name+"_weights.pt")
        torch.save(self.state_dict(), TESTS+self.savedir+"/"+self.name+"_weights.pt")

    def load(self, device, rel_path=TESTS):
        print("\nLoading: ", rel_path+self.savedir+"/"+self.name+"_weights.pt")
        self.load_state_dict(torch.load(rel_path+self.savedir+"/"+self.name+"_weights.pt"))
        print("\n", list(self.state_dict().keys()), "\n")
        self.to(device)

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
        return nn.Softmax(dim=-1)(x)


class GPRedBNN:

    def __init__(self, dataset_name, base_net):
        self.dataset_name = dataset_name
        self.base_net = base_net

    def get_filename(self, n_inputs):
        return str(self.dataset_name)+"_GPRedBNN_inp="+str(n_inputs)

    def train(self, x_train, y_train, device):
        
        out_train = self.base_net.model(x_train)
        out_train=out_train.cpu().detach().numpy()
        start = time.time()
        gp = GaussianProcessClassifier(kernel=1.0 * RBF([1.0]), random_state=0,
                                       multi_class='one_vs_rest', n_jobs=10, copy_X_train=False)
        labels = y_train.argmax(-1).cpu().detach().numpy()
        gp.fit(out_train, labels)
        execution_time(start=start, end=time.time())

        self.gp = gp
        print(out_train, self.gp.predict(out_train))
        self.save(gp=gp, n_inputs=len(x_train))

        return gp

    def forward(self, inputs):
        torch_out = self.base_net.model(inputs)
        np_out = torch_out.cpu().detach().numpy()
        predictions = self.gp.predict_proba(np_out)
        torch_predictions = torch.from_numpy(predictions).float()
        return torch_predictions.to("cuda") if inputs.is_cuda else torch_predictions

    def evaluate(self, x_test, y_test, device, savedir=None):
        predictions = self.forward(x_test).argmax(-1)
        labels = y_test.argmax(-1)
        print("\npreds =", predictions[:10],"\tlabels =", labels[:10])

        correct_predictions = (predictions == labels).sum()
        accuracy = 100 * correct_predictions / len(x_test)
        print("\nTest accuracy = %.2f%%" % (accuracy))
        return accuracy

    def save(self, gp, n_inputs, savedir=None):
        path = TESTS+self.base_net.savedir+"/"
        name = self.get_filename(n_inputs=n_inputs)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_to_pickle(gp, path, self.name+"_weights.pkl")
        self.name = name

    def load(self, n_inputs, rel_path):
        name = self.get_filename(n_inputs=n_inputs)
        self.gp = load_from_pickle(rel_path+self.base_net.savedir+"/"+name+"_weights.pkl")
        print("\ngp params:", self.gp.get_params())
        self.name = name


def attack_increasing_eps(nn, bnn, dataset, device, method, n_inputs=100, n_samples=100):

    _, _, x_test, y_test, _, _ = load_dataset(dataset, n_inputs=n_inputs, shuffle=True)
    x_test, y_test = (torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device))

    df = pandas.DataFrame(columns=["attack", "epsilon", "test_acc", "adv_acc", 
                                   "softmax_rob", "model_type"])

    row_count = 0
    for epsilon in [0.1, 0.15, 0.2, 0.25, 0.3]:

        df_dict = {"epsilon":epsilon, "attack":method}
        hyperparams = {"epsilon":epsilon}

        ### attacking the base network
        x_attack = attack(net=nn, x_test=x_test, y_test=y_test, dataset_name=dataset, 
                          device=device, method=method, filename=nn.name+"_eps="+str(epsilon), 
                          savedir=nn.savedir, hyperparams=hyperparams)
        # x_attack = load_attack(model=nn, method=attack, rel_path=TESTS, n_samples=n_samples,
        #                        savedir=nn.name, filename=nn.name+"_eps="+str(epsilon))
        
        ### defending with both networks
        for net, model_type in [(nn,"nn"), (bnn, "bnn")]:
            n_samp = n_samples if model_type == "bnn" else None
            test_acc, adv_acc, softmax_rob = attack_evaluation(model=net, x_test=x_test, 
                        n_samples=n_samp, x_attack=x_attack, y_test=y_test, device=device)
            
            df_dict.update({"test_acc":test_acc, "adv_acc":adv_acc,
                            "softmax_rob":softmax_rob, "model_type":model_type})

            df.loc[row_count] = pandas.Series(df_dict)
            row_count += 1

    print("\nSaving:", df)
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    df.to_csv(TESTS+str(dataset)+"_increasing_eps_"+str(method)+".csv", index = False, header=True)
    return df

def plot_increasing_eps(df, dataset, method, n_samples):
    print(df)

    sns.set_style("darkgrid")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    plt.suptitle(f"{method} attack on {dataset}")
    sns.lineplot(data=df, x="epsilon", y="adv_acc", hue="model_type", style="model_type", ax=ax[0])
    sns.lineplot(data=df, x="epsilon", y="softmax_rob", hue="model_type", style="model_type", ax=ax[1])
    
    filename = str(dataset)+"_increasing_eps_"+str(method)+"_samp="+str(n_samples)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)

def main(args):

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    base_inputs, epochs, lr, hidden, gp_inputs = saved_GPbaseNNs[args.dataset].values()
    
    # === base NN ===
    train_loader, test_loader, inp_shape, out_size = \
                            data_loaders(dataset_name=args.dataset, batch_size=64, 
                                         n_inputs=base_inputs, shuffle=True)

    nn = GPbaseNN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size, 
                  hidden_size=hidden, epochs=epochs, lr=lr)
    # nn.train(train_loader=train_loader, device=args.device)
    nn.load(device=args.device, rel_path=TESTS)
    nn.evaluate(test_loader=test_loader, device=args.device)

    # === LaplaceRedBNN ===

    x_train, y_train, x_test, y_test, _, _ = \
                        load_dataset(args.dataset, n_inputs=gp_inputs, shuffle=True)

    x_train, y_train = (torch.from_numpy(x_train).to(args.device), 
                        torch.from_numpy(y_train).to(args.device))
    x_test, y_test = (torch.from_numpy(x_test).to(args.device), 
                      torch.from_numpy(y_test).to(args.device))

    gp = GPRedBNN(dataset_name=args.dataset, base_net=nn)

    # gp.train(x_train=x_train, y_train=y_train, device=args.device)
    gp.load(n_inputs=gp_inputs, rel_path=TESTS)
    gp.evaluate(x_test=x_test, y_test=y_test, device=args.device)

    # === single attack === 

    # x_attack = attack(net=nn, x_test=x_test, y_test=y_test, dataset_name=args.dataset, 
    #                   device=args.device, method=args.attack, filename=nn.name)
    # # x_attack = load_attack(model=nn, method=args.attack, rel_path=TESTS, filename=nn.name)
    # attack_evaluation(model=nn, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)
    # attack_evaluation(model=gp, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

    # === multiple attacks ===
    gp_samples = 100
    df = attack_increasing_eps(nn=nn, bnn=gp, dataset=args.dataset, device=args.device, 
                                method=args.attack, n_samples=gp_samples)
    # df = pandas.read_csv(TESTS+str(args.dataset)+"_increasing_eps_"+str(args.attack)+".csv")
    plot_increasing_eps(df, dataset=args.dataset, method=args.attack, n_samples=gp_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--inputs", default=1000, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, 
                        help="mnist, fashion_mnist, cifar, half_moons")   
    parser.add_argument("--hidden_size", default=16, type=int, help="power of 2")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
    parser.add_argument("--attack", default="fgsm", type=str, help="fgsm, pgd")
    main(args=parser.parse_args())