"""
NEEDS HEAVY REFACTORING
"""

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
ACC_THS=70

#################################
# exp loss gradients components #
#################################

class MoonsBNN(BNN):
    def __init__(self, hidden_size, activation, architecture, inference, 
                 epochs, lr, n_samples, warmup, n_inputs, input_shape, output_size):
        super(MoonsBNN, self).__init__("half_moons", hidden_size, activation, architecture, 
                inference, epochs, lr, n_samples, warmup, input_shape, output_size)
        self.name = self.get_name(epochs, lr, n_samples, warmup, n_inputs)

def plot_half_moons(n_points=200):

    x_train, y_train, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name="half_moons", n_inputs=n_points, channels="first") 
    
    labels = onehot_to_labels(y_train)
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=150, facecolor='w', edgecolor='k')
    df = pandas.DataFrame.from_dict({"x":x_train.squeeze()[:,0],
                                     "y":x_train.squeeze()[:,1],
                                     "label":labels[:]})
    g = sns.scatterplot(data=df, x="x", y="y", hue="label", alpha=0.9, ax=ax)
    filename = "halfMoons_"+str(n_points)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)

def _train(hidden_size, activation, architecture, inference, 
           epochs, lr, n_samples, warmup, n_inputs, posterior_samples, device):

    batch_size = 64 if inference=="svi" else 1024
    train_loader, _, inp_shape, out_size = \
            data_loaders(dataset_name="half_moons", batch_size=batch_size, 
                         n_inputs=n_inputs, shuffle=False)

    bnn = MoonsBNN(hidden_size, activation, architecture, inference, 
                   epochs, lr, n_samples, warmup, n_inputs, inp_shape, out_size)
    bnn.train(train_loader=train_loader, device=device)

def serial_train(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples):

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    for init in combinations:
        _train(*init, "cuda")

def parallel_train(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples):
    from joblib import Parallel, delayed

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    Parallel(n_jobs=10)(
        delayed(_train)(*init, "cpu") for init in combinations)

def _compute_grads(hidden_size, activation, architecture, inference, 
           epochs, lr, n_samples, warmup, n_inputs, posterior_samples, rel_path, test_points,
           device):

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name="half_moons", batch_size=32, n_inputs=test_points, shuffle=True)

    bnn = MoonsBNN(hidden_size, activation, architecture, inference, 
                   epochs, lr, n_samples, warmup, n_inputs, inp_shape, out_size)
    bnn.load(device="cpu", rel_path=rel_path)
    loss_gradients(net=bnn, n_samples=posterior_samples, savedir=bnn.name+"/", 
                    data_loader=test_loader, device=device, filename=bnn.name)

def parallel_compute_grads(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples, 
                          rel_path, test_points):
    from joblib import Parallel, delayed

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    Parallel(n_jobs=10)(
        delayed(_compute_grads)(*init, rel_path, test_points, "cpu") for init in combinations)

def serial_compute_grads(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples, 
                          rel_path, test_points):

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                         epochs, lr, n_samples, warmup, n_inputs, posterior_samples))
    
    for init in combinations:
        _compute_grads(*init, rel_path, test_points, "cuda")

def build_components_dataset(hidden_size, activation, architecture, inference, epochs, lr, 
                            n_samples,  warmup, n_inputs, posterior_samples, test_points,
                            device="cuda", rel_path=TESTS):

    _, _, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name="half_moons", n_inputs=test_points, channels="first") 

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                                          epochs, lr, n_samples, warmup, n_inputs))
    
    cols = ["hidden_size", "activation", "architecture", "inference", "epochs", "lr", 
            "n_samples", "warmup", "n_inputs", "posterior_samples", "test_acc",
            "x","y","loss_gradients_x","loss_gradients_y"]
    df = pandas.DataFrame(columns=cols)

    row_count = 0
    for init in combinations:

        bnn = MoonsBNN(*init, inp_shape, out_size)
        bnn.load(device=device, rel_path=rel_path)
        
        for p_samp in posterior_samples:
            bnn_dict = {cols[k]:val for k, val in enumerate(init)}

            test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=64)
            test_acc = bnn.evaluate(test_loader=test_loader, device=device, n_samples=p_samp)
            loss_grads = load_loss_gradients(n_samples=p_samp, filename=bnn.name, 
                                             savedir=bnn.name+"/", relpath=rel_path)

            loss_gradients_components = loss_grads[:test_points]
            for idx, grad in enumerate(loss_gradients_components):
                x, y = x_test[idx].squeeze()
                bnn_dict.update({"posterior_samples":p_samp, "test_acc":test_acc,
                                 "x":x,"y":y,
                                 "loss_gradients_x":grad[0], "loss_gradients_y":grad[1]})
                df.loc[row_count] = pandas.Series(bnn_dict)
                row_count += 1

    print("\nSaving:", df.head())
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    df.to_csv(TESTS+"halfMoons_lossGrads_gridSearch_"+str(test_points)+".csv", 
              index = False, header=True)
    return df

def scatterplot_gridSearch_samp_vs_hidden(dataset, posterior_samples, hidden_size, 
                                           test_points, device="cuda"):

    dataset = dataset[dataset["test_acc"]>ACC_THS]
    print("\n---scatterplot_gridSearch_samp_vs_hidden---\n", dataset)

    categorical_rows = dataset["hidden_size"]
    categorical_cols = dataset["posterior_samples"]
    nrows = len(hidden_size)
    ncols = len(posterior_samples)

    sns.set_style("darkgrid")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6), dpi=150, 
                           facecolor='w', edgecolor='k')

    min_acc, max_acc = dataset["test_acc"].min(), dataset["test_acc"].max()
    print(min_acc, max_acc)

    for c, col_val in enumerate(posterior_samples):
        for r, row_val in enumerate(hidden_size):

            legend = "full" if (r==3)&(c==2) else None
            df = dataset[(categorical_rows==row_val)&(categorical_cols==col_val)]
            g = sns.scatterplot(data=df, x="loss_gradients_x", y="loss_gradients_y", 
                                size="n_inputs", hue="n_inputs", alpha=0.8, legend=legend,
                                ax=ax[r,c], sizes=(20, 80), palette=cmap)
            ax[r,c].set_xlabel("")
            ax[r,c].set_ylabel("")
            xlim=1.1*np.max(np.abs(df["loss_gradients_x"]))
            ylim=1.1*np.max(np.abs(df["loss_gradients_y"]))
            ax[r,c].set_xlim(-xlim,+xlim)
            ax[r,c].set_ylim(-ylim,+ylim)
            ax[-1,c].set_xlabel(str(col_val),labelpad=3,fontdict=dict(weight='bold'))
            ax[r,0].set_ylabel(str(row_val),labelpad=10,fontdict=dict(weight='bold')) 

    g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

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
                            n_samples,  warmup, n_inputs, posterior_samples, test_points,
                            device="cuda", rel_path=TESTS):

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name="half_moons", batch_size=64, n_inputs=test_points, shuffle=True)

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                      epochs, lr, n_samples, warmup, n_inputs))
    
    cols = ["hidden_size", "activation", "architecture", "inference", "epochs", "lr", 
            "n_samples", "warmup", "n_inputs", "posterior_samples", "test_acc", "var"]
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

            for value in loss_grads_var:
                bnn_dict.update({"posterior_samples":p_samp, "test_acc":test_acc, "var":value})
                df.loc[row_count] = pandas.Series(bnn_dict)
                row_count += 1

    print("\nSaving:", df.head())

    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    df.to_csv(TESTS+"halfMoons_lossGrads_compVariance_"+str(test_points)+".csv", 
              index = False, header=True)
    return df

def scatterplot_gridSearch_variance(dataset, test_points, device="cuda"):

    dataset = dataset[dataset["test_acc"]>ACC_THS]
    print("\n---scatterplot_gridSearch_variance---\n", dataset)

    sns.set_style("darkgrid")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])
    matplotlib.rc('font', **{'weight': 'bold', 'size': 10})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5), dpi=150, 
                           facecolor='w', edgecolor='k')

    var_ths = np.max(df["var"])*0.01
    df = dataset[dataset["var"]>var_ths]
    g = sns.stripplot(data=df, x="hidden_size", y="var", hue="posterior_samples",
                        # size="posterior_samples", hue="posterior_samples", alpha=0.8, 
                        # sizes=(20, 100), #style="posterior_samples", 
                        ax=ax, palette="gist_heat")

    g.set(ylim=(0, None))
    ax.set_xlabel(f"{val}")
    ax.set_ylabel("")
    ax.xaxis.set_label_position("top")

    fig.text(0.5, 0.01, 'Posterior samples', ha='center', fontsize=12)
    filename = "halfMoons_lossGrads_compVariance_"+str(test_points)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)


##########################
# robustness vs accuracy #
##########################


def _compute_attacks(method, hidden_size, activation, architecture, inference, 
           epochs, lr, n_samples, warmup, n_inputs, posterior_samples, rel_path, test_points):

    _, _, x_test, y_test, inp_shape, out_size = \
            load_dataset(dataset_name="half_moons", n_inputs=test_points, channels="first") 

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    bnn = MoonsBNN(hidden_size, activation, architecture, inference, 
                   epochs, lr, n_samples, warmup, n_inputs, inp_shape, out_size)
    bnn.load(device="cpu", rel_path=rel_path)
        
    x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, dataset_name="half_moons", 
                      device="cpu", method=method, filename=bnn.name, 
                      n_samples=posterior_samples)

def parallel_grid_attack(method, hidden_size, activation, architecture, inference, epochs, lr, 
                         n_samples, warmup, n_inputs, posterior_samples, rel_path, test_points):
    from joblib import Parallel, delayed

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                        epochs, lr, n_samples, warmup, n_inputs, posterior_samples))

    Parallel(n_jobs=10)(
        delayed(_compute_attacks)(method, *init, rel_path, test_points) 
                for init in combinations)

def grid_attack(method, hidden_size, activation, architecture, inference, epochs, lr, 
                  n_samples, warmup, n_inputs, posterior_samples, test_points, device="cuda", 
                  rel_path=TESTS):
   
    _, _, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name="half_moons", n_inputs=test_points, channels="first") 

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                        epochs, lr, n_samples, warmup, n_inputs))
    for init in combinations:

        bnn = MoonsBNN(*init, inp_shape, out_size)
        bnn.load(device=device, rel_path=rel_path)
        
        for p_samp in posterior_samples:
            x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, dataset_name="half_moons", 
                              device=device, method=method, filename=bnn.name, 
                              n_samples=p_samp)

def build_attack_dataset(method, hidden_size, activation, architecture, inference, epochs, lr, 
                          n_samples, warmup, n_inputs, posterior_samples, 
                         test_points, device="cuda", rel_path=TESTS):
    
    _, _, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name="half_moons", n_inputs=test_points, channels="first") 

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                        epochs, lr, n_samples, warmup, n_inputs))

    cols = ["hidden_size", "activation", "architecture", "inference", "epochs", "lr", 
            "n_samples", "warmup", "n_inputs", "posterior_samples", 
            "x_orig","y_orig","x_adv","y_adv","label",
            "test_acc",  "adversarial_acc", "softmax_rob"]
    df = pandas.DataFrame(columns=cols)

    row_count = 0
    for init in combinations:

        bnn = MoonsBNN(*init, inp_shape, out_size)
        bnn.load(device=device, rel_path=DATA)
        
        for p_samp in posterior_samples:
            bnn_dict = {cols[k]:val for k, val in enumerate(init)}

            x_test = x_test[:test_points]
            x_attack = load_attack(model=bnn, method=method, filename=bnn.name, 
                                  n_samples=p_samp, rel_path=rel_path)[:test_points]

            test_acc, adversarial_acc, softmax_rob = \
                   attack_evaluation(model=bnn, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                                     device=device, n_samples=p_samp)
            labels = onehot_to_labels(y_test)

            for idx in range(len(x_test)):
                x_orig, y_orig = x_test[idx].flatten()
                x_adv, y_adv = x_attack[idx].flatten()

                bnn_dict.update({"posterior_samples":p_samp, "test_acc":test_acc, 
                                 "x_orig":x_orig.item(),"y_orig":y_orig.item(),
                                 "label":labels[idx].item(),
                                 "x_adv":x_adv.item(),"y_adv":y_adv.item(),
                                 "adversarial_acc":adversarial_acc, "softmax_rob":softmax_rob})
            
                df.loc[row_count] = pandas.Series(bnn_dict)
                row_count += 1

    print("\nSaving:", df)
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    df.to_csv(TESTS+"halfMoons_attack_"+str(test_points)+"_"+str(method)+".csv", 
              index = False, header=True)
    return df

def plot_rob_acc(dataset, test_points, method, device="cuda"):

    dataset = dataset[dataset["test_acc"]>ACC_THS]
    print("\n---plot_rob_acc---\n", dataset)

    categorical_rows=dataset["hidden_size"]
    nrows = len(np.unique(categorical_rows))
    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 6), dpi=150, facecolor='w', edgecolor='k')

    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightseagreen","darkred","black"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])

    for i, categ_row in enumerate(np.unique(categorical_rows)):
        df = dataset[categorical_rows==categ_row]
        g = sns.scatterplot(data=df, x="test_acc", y="softmax_rob", 
                            size="n_inputs", hue="n_inputs", alpha=0.9, 
                            ax=ax[i], legend="full", sizes=(20, 100), palette=cmap)
        ax[i].set_ylim(-0.1,1.1)
        ax[i].set_xlabel("")
        ax[i].set_ylabel(f"{categ_row}", rotation=270, labelpad=10,
                          fontdict=dict(weight='bold'))
        ax[i].yaxis.set_label_position("right")

        if i != 2:
            ax[i].set(xticklabels=[])
            ax[i].legend_.remove()

    fig.text(0.5, 0.02, r"Test accuracy", fontsize=11, ha='center',fontdict=dict(weight='bold'))
    fig.text(0.06, 0.5, "Softmax robustness", va='center',fontsize=11, rotation='vertical',
        fontdict=dict(weight='bold'))
    fig.text(0.92, 0.5, "Hidden size", va='center',fontsize=10, rotation=270,
        fontdict=dict(weight='bold'))

    filename = "halfMoons_acc_vs_rob_scatter_"+str(test_points)+"_"+str(method)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)


def stripplot_rob_acc(dataset, test_points, method, device="cuda"):

    dataset = dataset[dataset["test_acc"]>ACC_THS]
    print("\n---stripplot_rob_acc---\n", dataset)

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'weight':'bold','size': 10})
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), dpi=150, facecolor='w', edgecolor='k')

    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightseagreen","darkred","black"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])

    for idx, y in enumerate(["posterior_samples","hidden_size"]):
        df = dataset[["test_acc", y, "n_inputs", "softmax_rob"]].drop_duplicates()

        g = sns.stripplot(data=df, y=y, x="test_acc", 
                            hue="n_inputs", alpha=0.8,
                            ax=ax[idx,0], palette="gist_heat", orient="h")
        ax[idx,0].set_ylabel("")
        ax[idx,0].set_xlabel("Test accuracy", fontdict=dict(weight='bold'))

        g = sns.stripplot(data=df, y=y, x="softmax_rob",  
                          hue="n_inputs", alpha=0.8, 
                            ax=ax[idx,1], palette="gist_heat", orient="h")
        ax[idx,1].set_ylabel("")
        ax[idx,1].set_xlabel("Softmax robustness", fontdict=dict(weight='bold'))

    ax[0,1].legend_.remove()
    ax[1,1].legend_.remove()
    ax[1,0].legend_.remove()

    fig.text(0.03, 0.3, "Hidden size", va='center',fontsize=10, rotation='vertical', fontdict=dict(weight='bold'))
    fig.text(0.03, 0.8, "Posterior samples", va='center',fontsize=10, rotation='vertical', fontdict=dict(weight='bold'))

    filename = "halfMoons_acc_vs_rob_strip_"+str(test_points)+"_"+str(method)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)

def plot_attacks(dataset, test_points, method, device="cuda"):
    dataset = dataset[dataset["test_acc"]>ACC_THS].sample(500)
    print("\n---plot_attacks---\n", dataset)

    dataset = pandas.DataFrame(dataset, columns=dataset.columns)
    dataset['color']=dataset["x_orig"]+dataset["y_orig"]

    categorical_cols = dataset["hidden_size"]
    ncols = len(np.unique(categorical_cols))

    sns.set_style("darkgrid")
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightseagreen","darkgreen","black"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])
    cmap = [cmap1, cmap2]
    marker = ["d","o"]
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(10, 6), dpi=150, 
                           facecolor='w', edgecolor='k')

    vmin, vmax = (dataset["color"].min(), dataset["color"].max())

    for label in [0,1]:
        for c, col_val in enumerate(np.unique(categorical_cols)):
            for r, (x,y) in enumerate([("x_orig","y_orig"),("x_adv","y_adv")]):

                df = dataset[(dataset["label"]==label)&(categorical_cols==col_val)]
                g = sns.scatterplot(data=df, x=x, y=y, alpha=0.7, marker=marker[label],
                                    hue="color",  palette=cmap[label], size="softmax_rob", 
                                    sizes=(20,100), 
                                    ax=ax[r,c], legend=False)
                ax[r,c].set_xlabel("")
                ax[r,c].set_ylabel("")

            ax[-1,c].set_xlabel(str(col_val),labelpad=3,fontdict=dict(weight='bold'))

    ax[0,0].set_ylabel("Original points",labelpad=3,fontdict=dict(weight='bold'))
    ax[1,0].set_ylabel("Adversarial points",labelpad=10,fontdict=dict(weight='bold')) 

    ## titles and labels
    fig.text(0.5, 0.01, r"Hidden size", 
        fontsize=12, ha='center',fontdict=dict(weight='bold'))
    fig.suptitle(f"{method} adversarial attack on Half Moons dataset",
               fontsize=12, ha='center', fontdict=dict(weight='bold'))

    filename = "halfMoons_attack_"+str(test_points)+"_"+str(method)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)

# === final scatterplot ===

def build_final_dataset(test_points, device="cuda"):

    posterior_samples = 250

    _, _, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name="half_moons", n_inputs=test_points, channels="first") 

    cols = ["hidden_size", "activation", "architecture", "inference", "epochs", "lr", 
            "n_samples", "warmup", "n_inputs", "posterior_samples", "test_acc",
            "x","y","loss_gradients_x","loss_gradients_y"]
    
    df = pandas.DataFrame(columns=cols)

    row_count = 0
    for inference_idx in [0,1]:

        if inference_idx==0:
            inference = ["svi"]
            epochs = [5, 10, 20]
            lr = [0.01, 0.001, 0.0001]
            n_inputs = [5000, 10000, 15000]
            n_samples, warmup = ([None],[None])
            rel_path = DATA
            hidden_size = [32, 128, 256] 

        elif inference_idx==1:
            inference = ["hmc"]
            n_samples = [250]
            warmup = [100, 200, 500]
            n_inputs = [5000, 10000, 15000]
            epochs, lr = ([None],[None])
            rel_path = TESTS
            hidden_size = [32, 128, 256, 512] 

        activation = ["leaky"]
        architecture = ["fc2"]

        combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                                              epochs, lr, n_samples, warmup, n_inputs))

        for init in combinations:

            bnn = MoonsBNN(*init, inp_shape, out_size)
            bnn.load(device=device, rel_path=rel_path)
            
            bnn_dict = {cols[k]:val for k, val in enumerate(init)}

            test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=64)
            test_acc = bnn.evaluate(test_loader=test_loader, device=device, 
                       n_samples=posterior_samples)
            loss_grads = load_loss_gradients(n_samples=posterior_samples, filename=bnn.name, 
                                             savedir=bnn.name+"/", relpath=rel_path)

            loss_gradients_components = loss_grads[:test_points]
            for idx, grad in enumerate(loss_gradients_components):
                x, y = x_test[idx].squeeze()
                bnn_dict.update({"posterior_samples":posterior_samples, "test_acc":test_acc,
                                 "x":x,"y":y,
                                 "loss_gradients_x":grad[0], "loss_gradients_y":grad[1]})
                df.loc[row_count] = pandas.Series(bnn_dict)
                row_count += 1

    print("\nSaving:", df.head())
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    df.to_csv(TESTS+"halfMoons_lossGrads_final_"+str(test_points)+".csv", 
              index = False, header=True)
    return df

def final_scatterplot_svi_hmc(dataset, hidden_size, test_points, device="cuda"):

    dataset = dataset[dataset["test_acc"]>ACC_THS]
    print("\n---scatterplot_gridSearch_samp_vs_hidden---\n", dataset)

    categorical_rows = dataset["hidden_size"]
    categorical_cols = dataset["inference"]
    nrows = len(np.unique(categorical_rows))
    ncols = 2

    sns.set_style("darkgrid")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    
    # print(dataset[dataset["inference"]=="hmc"]["test_acc"].drop_duplicates())

    for c, col_val in enumerate(np.unique(categorical_cols)):

        vmin = dataset[categorical_cols==col_val]["test_acc"].min()
        vmax = dataset[categorical_cols==col_val]["test_acc"].max()
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        
        for r, row_val in enumerate(np.unique(categorical_rows)):
            df = dataset[(categorical_rows==row_val)&(categorical_cols==col_val)]

            g = sns.scatterplot(data=df, x="loss_gradients_x", y="loss_gradients_y", alpha=0.8, 
                            hue="test_acc", hue_norm=norm, size="n_inputs", legend=False, 
                            ax=ax[r,c], sizes=(30, 80), palette=cmap)
            ax[r,c].set_xlabel("")
            ax[r,c].set_ylabel("")
            xlim=1.1*np.max(np.abs(df["loss_gradients_x"]))
            ylim=1.1*np.max(np.abs(df["loss_gradients_y"]))
            ax[r,0].set_ylabel(str(row_val),labelpad=10,fontdict=dict(weight='bold')) 
    
    ## colorbar    
    cbar_ax = fig.add_axes([0.93, 0.11, 0.01, 0.77])
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.ax.set_ylabel('Test accuracy (%)',labelpad=10, rotation=270, fontdict=dict(weight='bold'))
    cbar.set_ticks([0,1])
    cbar.set_ticklabels([ACC_THS,100])

    ax[-1,0].set_xlabel("HMC",labelpad=4,fontdict=dict(weight='bold'))
    ax[-1,1].set_xlabel("VI",labelpad=4,fontdict=dict(weight='bold'))
    
    ## titles and labels
    fig.text(0.02, 0.5, "Hidden size", va='center',fontsize=11, rotation='vertical',
        fontdict=dict(weight='bold'))
    fig.suptitle(r"Expected loss gradients components $\langle \nabla_{x} L(x,w)\rangle_{w}$ on Half Moons dataset",
               fontsize=12, ha='center', fontdict=dict(weight='bold'))

    filename = "halfMoons_final_"+str(test_points)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)

def final_scatterplot_hmc(dataset, hidden_size, test_points, orient="v", device="cuda"):
    dataset = dataset[dataset["inference"]=="hmc"]
    dataset = dataset[dataset["test_acc"]>ACC_THS]
    dataset = dataset[dataset["hidden_size"].isin(hidden_size)]
    print("\n---scatterplot_gridSearch_samp_vs_hidden---\n", dataset)

    categorical_rows = dataset["hidden_size"]
    nrows = len(np.unique(categorical_rows))

    sns.set_style("darkgrid")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])
    matplotlib.rc('font', **{'size': 10, 'weight' : 'bold'})

    if orient == "v":
        num_rows, num_cols = (nrows, 1) 
        figsize = (4, 7)

    else:
        num_rows, num_cols = (1, nrows)
        figsize = (10, 2.3)
    
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, dpi=300, 
                           facecolor='w', edgecolor='k')
    vmin, vmax = (dataset["test_acc"].min(), dataset["test_acc"].max())
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        
    for r, row_val in enumerate(np.unique(categorical_rows)):
        df = dataset[categorical_rows==row_val]

        legend = "full" if r==3 else None
        g = sns.scatterplot(data=df, x="loss_gradients_x", y="loss_gradients_y", alpha=0.8, 
                            hue="n_inputs", size="n_inputs", legend=legend, 
                            ax=ax[r], sizes=(30, 80), palette=cmap)
        ax[r].set_xlabel("")
        ax[r].set_ylabel("")
        xlim=1.1*np.max(np.abs(df["loss_gradients_x"]))
        ylim=1.1*np.max(np.abs(df["loss_gradients_y"]))
        ax[r].set_xlim(-xlim,+xlim)
        ax[r].set_ylim(-ylim,+ylim)

        if orient == "v":
            ax[r].set_ylabel(str(row_val),labelpad=10,fontdict=dict(weight='bold'),rotation=270) 
            ax[r].yaxis.set_label_position("right")
        else:
            ax[r].set_title(str(row_val), fontdict=dict(weight='bold',size=10)) 
            ax[r].xaxis.set_label_position("bottom")
            ax[r].set_xlabel(r"$\langle \frac{\partial L}{\partial x_1}(x,w)\rangle_{w}$", 
                             labelpad=3, fontsize=11)

    ax[0].set_ylabel(r"$\langle \frac{\partial L}{\partial x_2}(x,w)\rangle_{w}$",
                     labelpad=3, fontsize=11)

    if orient == "h":
        legend = g.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), ncol=1, title="")
        legend.texts[0].set_text("training\ninputs")

    plt.tight_layout()
    filename = "halfMoons_final_hmc_"+str(test_points)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)


def main(args):

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # plot_half_moons()

    # posterior_samples = 5
    # _train(args.hidden_size, args.activation, args.architecture, args.inference, 
    #        args.epochs, args.lr, args.samples, args.warmup, args.inputs, 
    #        posterior_samples, args.device)

    ## === svi ===
    # inference = ["svi"]
    # epochs = [5, 10, 20]
    # lr = [0.01, 0.001, 0.0001]
    # n_inputs = [5000, 10000, 15000]
    # posterior_samples = [50,100,250]
    # n_samples, warmup = ([None],[None])

    # === hmc ===
    inference = ["hmc"]
    n_samples = [250]

    # warmup = [100, 200, 500]
    # n_inputs = [5000, 10000, 15000]
    # test_points = 99

    warmup = [50]#, 100, 500]
    n_inputs = [5000]#5000, 10000, 15000]
    test_points = 100

    posterior_samples = [10,20,50]#1, 50,250]
    epochs, lr = ([None],[None])

    # === grid search ===
    hidden_size = [32, 128, 256, 512] 
    activation = ["leaky"]
    architecture = ["fc2"]
    attack = "fgsm"
    init = (hidden_size, activation, architecture, inference, 
            epochs, lr, n_samples, warmup, n_inputs, posterior_samples)

    # init = [[arg] for arg in [args.hidden_size, args.activation, args.architecture, 
    #         args.inference, args.epochs, args.lr, args.samples, args.warmup, args.inputs, 3]]

    # serial_train(*init)
    # parallel_train(*init)

    # serial_compute_grads(*init, rel_path=TESTS, test_points=test_points)
    # parallel_compute_grads(*init, rel_path=TESTS, test_points=test_points)
    # parallel_grid_attack(attack, *init, rel_path=TESTS, test_points=test_points) 
    ## grid_attack(attack, *init, test_points=test_points, device="cuda", rel_path=DATA) 

    # === plots ===
    # dataset = build_components_dataset(*init, device=args.device, test_points=test_points, rel_path=TESTS)
    # dataset = pandas.read_csv(DATA+"halfMoons_lossGrads_gridSearch_"+str(test_points)+".csv")
    # scatterplot_gridSearch_samp_vs_hidden(dataset=dataset, device=args.device, 
    #      test_points=test_points, posterior_samples=posterior_samples, hidden_size=hidden_size)

    # === concat datasets ===
    # warmup = [100, 200, 500]
    # n_inputs = [5000, 10000, 15000]
    # test_points = 99
    # dataset1 = pandas.read_csv(TESTS+"halfMoons_lossGrads_gridSearch_"+str(test_points)+".csv")

    # warmup = [10, 50, 100]
    # n_inputs = [50,100,1000]
    # test_points = 98
    # dataset2 = pandas.read_csv(TESTS+"halfMoons_lossGrads_gridSearch_"+str(test_points)+".csv")
    
    # df = pandas.concat([dataset1, dataset2], ignore_index=True)
    # scatterplot_gridSearch_samp_vs_hidden(dataset=df, device=args.device, 
    #      test_points=200, posterior_samples=posterior_samples, hidden_size=hidden_size)

    # dataset = build_variance_dataset(*init, device=args.device, test_points=test_points, rel_path=DATA)
    # dataset = pandas.read_csv(TESTS+"halfMoons_lossGrads_compVariance_"+str(test_points)+".csv")
    # scatterplot_gridSearch_variance(dataset, test_points, device="cuda")

    # # dataset = build_attack_dataset(attack, *init, device=args.device, test_points=test_points, rel_path=TESTS) 
    # dataset = pandas.read_csv(TESTS+"halfMoons_attack_"+str(test_points)+"_"+str(attack)+".csv")
    # plot_attacks(dataset, test_points, attack, device=args.device)
    # plot_rob_acc(dataset, test_points, attack, device=args.device)
    # stripplot_rob_acc(dataset, test_points, attack, device=args.device)

    # # === final SVI + HMC plot === 
    test_points = 100
    # dataset = build_final_dataset(device=args.device, test_points=test_points)
    dataset = pandas.read_csv(DATA+"halfMoons_lossGrads_final_"+str(test_points)+".csv")
    # final_scatterplot_svi_hmc(dataset, device=args.device, test_points=test_points, 
    #                                  hidden_size=[32,128,256])
    final_scatterplot_hmc(dataset, device=args.device, test_points=test_points, 
                                     hidden_size=[32,128,256,512], orient="h")

if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="Toy example on half moons")
    parser.add_argument("--n_inputs", default=1000, type=int)
    parser.add_argument("--hidden_size", default=128, type=int, help="power of 2 >= 16")
    parser.add_argument("--activation", default="leaky", type=str, 
                        help="relu, leaky, sigm, tanh")
    parser.add_argument("--architecture", default="fc2", type=str, help="fc, fc2")
    parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--samples", default=10, type=int)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
    main(args=parser.parse_args())