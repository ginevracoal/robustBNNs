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
import pandas
import itertools
from lossGradients import loss_gradients, load_loss_gradients
import matplotlib
from adversarialAttacks import attack, attack_evaluation, load_attack


DATA=DATA+"half_moons_grid_search/"
TEST_POINTS=100

#################################
# exp loss gradients components #
#################################


class MoonsBNN(BNN):
    def __init__(self, hidden_size, activation, architecture, inference, 
                 epochs, lr, n_samples, warmup, n_inputs, input_shape, output_size):
        super(MoonsBNN, self).__init__("half_moons", hidden_size, activation, architecture, 
                inference, epochs, lr, n_samples, warmup, input_shape, output_size)
        self.name = self.get_name(epochs, lr, n_samples, warmup, n_inputs)


def _train(hidden_size, activation, architecture, inference, 
           epochs, lr, n_samples, warmup, n_inputs, posterior_samples):

    train_loader, _, inp_shape, out_size = \
            data_loaders(dataset_name="half_moons", batch_size=64, n_inputs=n_inputs, shuffle=True)

    bnn = MoonsBNN(hidden_size, activation, architecture, inference, 
                   epochs, lr, n_samples, warmup, n_inputs, inp_shape, out_size)
    bnn.train(train_loader=train_loader, device="cpu")

def parallel_train(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples):
    from joblib import Parallel, delayed

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    Parallel(n_jobs=10)(
        delayed(_train)(*init) for init in combinations)

def _compute_grads(hidden_size, activation, architecture, inference, 
           epochs, lr, n_samples, warmup, n_inputs, posterior_samples):

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name="half_moons", batch_size=64, n_inputs=TEST_POINTS, shuffle=True)

    bnn = MoonsBNN(hidden_size, activation, architecture, inference, 
                   epochs, lr, n_samples, warmup, n_inputs, inp_shape, out_size)
    bnn.load(device="cpu", rel_path=TESTS)
    loss_gradients(net=bnn, n_samples=posterior_samples, savedir=bnn.name+"/", 
                    data_loader=test_loader, device="cpu", filename=bnn.name)

def parallel_compute_grads(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples):
    from joblib import Parallel, delayed

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    Parallel(n_jobs=10)(
        delayed(_compute_grads)(*init) for init in combinations)


def build_components_dataset(hidden_size, activation, architecture, inference, epochs, lr, 
                            n_samples,  warmup, n_inputs, posterior_samples, device="cuda", 
                            test_points=TEST_POINTS, rel_path=TESTS):

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name="half_moons", batch_size=64, n_inputs=test_points, shuffle=True)

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                      epochs, lr, n_samples, warmup, n_inputs))
    
    cols = ["hidden_size", "activation", "architecture", "inference", "epochs", "lr", 
            "n_samples", "warmup", "n_inputs", "posterior_samples", "test_acc",
            "loss_gradients_x","loss_gradients_y"]
    df = pandas.DataFrame(columns=cols)

    row_count = 0
    for init in combinations:

        bnn = MoonsBNN(*init, inp_shape, out_size)
        bnn.load(device=device, rel_path=rel_path)
        
        for p_samp in posterior_samples:
            bnn_dict = {cols[k]:val for k, val in enumerate(init)}

            test_acc = bnn.evaluate(test_loader=test_loader, device=device, n_samples=p_samp)
            loss_grads = load_loss_gradients(n_samples=p_samp, filename=bnn.name, 
                                             savedir=bnn.name+"/", relpath=rel_path) 
            loss_gradients_components = loss_grads[:test_points]
            for value in loss_gradients_components:
                bnn_dict.update({"posterior_samples":p_samp, "test_acc":test_acc,
                                 "loss_gradients_x":value[0], "loss_gradients_y":value[1]})
                df.loc[row_count] = pandas.Series(bnn_dict)
                row_count += 1

    print("\nSaving:", df.head())
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    df.to_csv(TESTS+"halfMoons_lossGrads_gridSearch_"+str(test_points)+".csv", 
              index = False, header=True)
    return df

def scatterplot_gridSearch_gradComponents(dataset, test_points, device="cuda"):

    print(dataset)
    categorical = dataset["posterior_samples"]
    ncols = len(np.unique(categorical))

    sns.set_style("darkgrid")
    cmap = sns.cubehelix_palette(rot=-.9, as_cmap=True)
    matplotlib.rc('font', **{'weight': 'bold', 'size': 10})
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(16, 5), dpi=150, 
                           facecolor='w', edgecolor='k')

    for idx, val in enumerate(np.unique(categorical)):
        # for y in ["loss_gradients_x","loss_gradients_y"]:
        df = dataset[categorical==val]
        g = sns.scatterplot(data=df, x="loss_gradients_x", y="loss_gradients_y", 
                            size="test_acc", hue="test_acc", alpha=0.85, 
                            #sizes=(5, 100), #style="hidden_size", 
                            ax=ax[idx], legend=False, palette=cmap)
        ax[idx].set_title(str(val)+" samples")
        ax[idx].set_xlabel("")
        ax[idx].set_ylabel("")
    # fig.text(0.5, 0.01, '(%) ', ha='center', fontsize=12)
    # fig.text(0.03, 0.5, 'Expected loss gradients components', va='center',fontsize=12,
    #                                                            rotation='vertical')
    # ax.legend(loc='upper right', title="posterior samples")

    fig.text(0.5, 0.01, "Samples involved in the expectations ($w \sim p(w|D)$)", ha='center')

    filename = "halfMoons_lossGradsComponents_"+str(test_points)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)


def scatterplot_gridSearch_samp_vs_hidden(dataset, test_points, device="cuda"):

    print(dataset)
    dataset = dataset[dataset["test_acc"]>60]

    # dataset["avg_grad"]=dataset[['loss_gradients_x', 'loss_gradients_y']].mean(axis=1)
    categorical_rows = dataset["hidden_size"]
    categorical_cols = dataset["posterior_samples"]
    nrows = len(np.unique(categorical_rows))
    ncols = len(np.unique(categorical_cols))

    sns.set_style("darkgrid")
    cmap = "gist_heat_r"
    # cmap = sns.cubehelix_palette(rot=-.7, as_cmap=True)
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6), dpi=150, 
                           facecolor='w', edgecolor='k')

    min_acc, max_acc = dataset["test_acc"].min(), dataset["test_acc"].max()

    for r, row_val in enumerate(np.unique(categorical_rows)):
        for c, col_val in enumerate(np.unique(categorical_cols)):

            df = dataset[(categorical_rows==row_val)&(categorical_cols==col_val)]

            g = sns.scatterplot(data=df, x="loss_gradients_x", y="loss_gradients_y", 
                                size="test_acc", hue="test_acc", alpha=0.8, 
                                vmin=min_acc, vmax=max_acc, 
                                ax=ax[r,c], legend=False, sizes=(20, 80), palette=cmap)
            ax[r,c].set_xlabel("")
            ax[r,c].set_ylabel("")
            # ax[0,c].xaxis.set_label_position("top")
            ax[-1,c].set_xlabel(str(col_val),labelpad=3,fontdict=dict(weight='bold'))
            # ax[r,-1].yaxis.set_label_position("right")
            ax[r,0].set_ylabel(str(row_val),labelpad=10,fontdict=dict(weight='bold')) #rotation=270,


    ## colorbar    
    cbar_ax = fig.add_axes([0.93, 0.08, 0.01, 0.8])
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=cmap), cax=cbar_ax)
    cbar.ax.set_ylabel('Test accuracy (%)', rotation=270, fontdict=dict(weight='bold'))
    cbar.set_ticks([0,1])
    cbar.set_ticklabels([60,100])
    
    ## titles and labels
    fig.text(0.03, 0.5, "Hidden size", va='center',fontsize=12, rotation='vertical',
        fontdict=dict(weight='bold'))
    fig.text(0.5, 0.01, r"Samples involved in the expectations ($w \sim p(w|D)$)", 
        fontsize=12, ha='center',fontdict=dict(weight='bold'))
    fig.suptitle(r"Expected loss gradients components $\langle \nabla_{x} L(x,w)\rangle_{w}$ on Half Moons dataset",
               fontsize=12, ha='center', fontdict=dict(weight='bold'))

    filename = "halfMoons_samp_vs_hidden_"+str(test_points)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)

def build_variance_dataset(hidden_size, activation, architecture, inference, epochs, lr, 
                            n_samples,  warmup, n_inputs, posterior_samples, device="cuda", 
                            test_points=TEST_POINTS, rel_path=TESTS):

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name="half_moons", batch_size=64, n_inputs=test_points, shuffle=True)

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                      epochs, lr, n_samples, warmup, n_inputs))
    
    cols = ["hidden_size", "activation", "architecture", "inference", "epochs", "lr", 
            "n_samples", "warmup", "n_inputs", "posterior_samples", "test_acc",
            "loss_gradients_var_x","loss_gradients_var_y"]
    df = pandas.DataFrame(columns=cols)

    row_count = 0
    for init in combinations:

        bnn = MoonsBNN(*init, inp_shape, out_size)
        bnn.load(device=device, rel_path=rel_path)
        
        for p_samp in posterior_samples:
            bnn_dict = {cols[k]:val for k, val in enumerate(init)}
            test_acc = bnn.evaluate(test_loader=test_loader, device=device, n_samples=p_samp)
            loss_grads = load_loss_gradients(n_samples=p_samp, filename=bnn.name, 
                                             savedir=bnn.name+"/", relpath=rel_path) 
            loss_grads_var = loss_grads[:test_points].var(0)
            bnn_dict.update({"posterior_samples":p_samp, "test_acc":test_acc,
                             "loss_gradients_var_x":loss_grads_var[0],
                             "loss_gradients_var_y":loss_grads_var[1]})
            df.loc[row_count] = pandas.Series(bnn_dict)
            row_count += 1

    print("\nSaving:", df.head())

    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    df.to_csv(TESTS+"halfMoons_lossGrads_compVariance_"+str(test_points)+".csv", 
              index = False, header=True)
    return df

def scatterplot_gridSearch_variance(dataset, test_points, device="cuda"):

    print(dataset)
    categorical = dataset["hidden_size"]
    ncols = len(np.unique(categorical))

    sns.set_style("darkgrid")
    cmap = sns.cubehelix_palette(rot=-.9, as_cmap=True)
    matplotlib.rc('font', **{'weight': 'bold', 'size': 10})
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(16, 5), dpi=150, 
                           facecolor='w', edgecolor='k')

    for idx, val in enumerate(np.unique(categorical)):
        for y in ["loss_gradients_var_x","loss_gradients_var_y"]:
            df = dataset[categorical==val]
            g = sns.scatterplot(data=df, x="test_acc", y=y, 
                                size="posterior_samples", hue="test_acc", alpha=0.85, 
                                sizes=(20, 100), #style="posterior_samples", 
                                ax=ax[idx], legend=False, palette=cmap)
            g.set(xlim=(60, None))
            # ax[idx].set_title(str(val)+" samples")
            ax[idx].set_xlabel("")
            ax[idx].set_ylabel("")
    # fig.text(0.5, 0.01, '(%) ', ha='center', fontsize=12)
    # fig.text(0.03, 0.5, 'Expected loss gradients components', va='center',fontsize=12,
    #                                                            rotation='vertical')
    # ax.legend(loc='upper right', title="posterior samples")

    # fig.text(0.5, 0.01, "Samples involved in the expectations ($w \sim p(w|D)$)", ha='center')

    filename = "halfMoons_lossGrads_compVariance_"+str(test_points)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)

##########################
# robustness vs accuracy #
##########################

def build_rob_acc_dataset(method, hidden_size, activation, architecture, inference, epochs, lr, 
                          n_samples, warmup, n_inputs, posterior_samples, device="cuda", 
                          test_points=TEST_POINTS, rel_path=TESTS):
    
    _, _, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name="half_moons", n_inputs=test_points, channels="first") 

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                        epochs, lr, n_samples, warmup, n_inputs))
    
    cols = ["hidden_size", "activation", "architecture", "inference", "epochs", "lr", 
            "n_samples", "warmup", "n_inputs", "posterior_samples", 
            "test_acc",  "adversarial_acc", "softmax_rob"]
    df = pandas.DataFrame(columns=cols)

    row_count = 0
    for init in combinations:

        bnn = MoonsBNN(*init, inp_shape, out_size)
        bnn.load(device=device, rel_path=rel_path)
        
        for p_samp in posterior_samples:
            bnn_dict = {cols[k]:val for k, val in enumerate(init)}

            # x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, dataset_name="half_moons", 
            #                   device=device, method=method, filename=bnn.name, 
            #                   n_samples=p_samp)
            x_attack = load_attack(model=bnn, method=method, filename=bnn.name, 
                                  n_samples=p_samp, rel_path=TESTS)

            test_acc, adversarial_acc, softmax_rob = \
                   attack_evaluation(model=bnn, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                                     device=device, n_samples=p_samp)

            bnn_dict.update({"posterior_samples":p_samp, "test_acc":test_acc, 
                             "adversarial_acc":adversarial_acc, "softmax_rob":softmax_rob})
            df.loc[row_count] = pandas.Series(bnn_dict)
            row_count += 1

        print("\nSaving:", df.head())
        os.makedirs(os.path.dirname(TESTS), exist_ok=True)
        df.to_csv(TESTS+"halfMoons_accVsRob_"+str(test_points)+"_"+str(method)+".csv", 
                  index = False, header=True)
    return df


def plot_rob_acc(dataset, test_points, method, device="cuda"):

    print(dataset)
    dataset = dataset[dataset["test_acc"]>60]

    sns.set_style("darkgrid")
    cmap = "gist_heat_r"
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=150, facecolor='w', edgecolor='k')

    g = sns.scatterplot(data=dataset, x="adversarial_acc", y="softmax_rob", 
                        # size="hidden_size", hue="n_inputs", alpha=0.8, 
                        ax=ax, legend=False, sizes=(20, 80), palette=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    ## titles and labels
    fig.text(0.5, 0.01, r"Test accuracy", fontsize=12, ha='center',fontdict=dict(weight='bold'))
    fig.text(0.03, 0.5, "Softmax robustness", va='center',fontsize=12, rotation='vertical',
        fontdict=dict(weight='bold'))

    filename = "halfMoons_acc_vs_rob_"+str(test_points)+"_"+str(method)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)


def main(args):

    # === train ===

    # dataset = "half_moons"

    # init = (args.hidden_size, args.activation, args.architecture, 
    #         args.inference, args.epochs, args.lr, args.samples, args.warmup)
    
    # train_loader, test_loader, inp_shape, out_size = \
    #                         data_loaders(dataset_name=dataset, batch_size=64, 
    #                                      n_inputs=args.inputs, shuffle=True)

    # bnn = MoonsBNN(dataset, *init, inp_shape, out_size, args.inputs)
   
    # bnn.train(train_loader=train_loader, device=args.device)
    # # bnn.load(device=args.device, rel_path=DATA)

    # bnn.evaluate(test_loader=test_loader, device=args.device)

    # === grid search ===

    hidden_size = [32]#, 128]# 32, 128, 256] 
    activation = ["leaky"]
    architecture = ["fc2"]
    inference = ["svi"]
    epochs = [5, 10, 20]
    lr = [0.01, 0.001, 0.0001]
    n_samples = [None]
    warmup = [None]
    n_inputs = [5000, 10000, 15000]
    posterior_samples = [1, 10, 50]
    init = (hidden_size, activation, architecture, inference, 
            epochs, lr, n_samples, warmup, n_inputs, posterior_samples)

    # init = [[arg] for arg in [args.hidden_size, args.activation, args.architecture, 
    #         args.inference, args.epochs, args.lr, args.samples, args.warmup, args.inputs, 3]]

    # parallel_train(*init)

    # === plots ===
    test_points = TEST_POINTS 

    parallel_compute_grads(*init)

    dataset = build_components_dataset(*init, device="cuda", test_points=test_points, rel_path=TESTS)
    dataset = pandas.read_csv(TESTS+"halfMoons_lossGrads_gridSearch_"+str(test_points)+".csv")
    # scatterplot_gridSearch_gradComponents(dataset=dataset, device="cuda", test_points=test_points)
    scatterplot_gridSearch_samp_vs_hidden(dataset=dataset, device="cuda", test_points=test_points)
    
    # # dataset = build_variance_dataset(*init, device="cuda", test_points=test_points, rel_path=DATA)
    # dataset = pandas.read_csv(TESTS+"halfMoons_lossGrads_compVariance_"+str(test_points)+".csv")
    # scatterplot_gridSearch_variance(dataset, test_points, device="cuda")

    # attack = "fgsm"
    # dataset = build_rob_acc_dataset(attack, *init, device="cuda", 
    #                                 test_points=test_points, rel_path=TESTS) 
    # plot_rob_acc(dataset, test_points, attack, device="cuda")


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