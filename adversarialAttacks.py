import sys
from directories import *
import argparse
from tqdm import tqdm
import pyro
import random
import copy
import torch
from nn import NN, saved_NNs
from bnn import BNN, saved_BNNs
# from reducedBNN import redBNN
from utils import load_dataset, save_to_pickle, load_from_pickle
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as nnf
import pandas
import os


#######################
# robustness measures #
#######################


def softmax_difference(original_predictions, adversarial_predictions):
    """
    Compute the expected l-inf norm of the difference between predictions and adversarial predictions.
    This is point-wise robustness measure.
    """

    original_predictions = nnf.softmax(original_predictions, dim=-1)
    adversarial_predictions = nnf.softmax(adversarial_predictions, dim=-1)

    # print(original_predictions.sum(-1))

    if len(original_predictions) != len(adversarial_predictions):
        raise ValueError("\nInput arrays should have the same length.")

    print("\n", original_predictions[0], "\t", adversarial_predictions[0])

    softmax_diff = original_predictions-adversarial_predictions
    softmax_diff_norms = softmax_diff.abs().max(dim=-1)[0]

    if softmax_diff_norms.min() < 0. or softmax_diff_norms.max() > 1.:
        raise ValueError("Softmax difference should be in [0,1]")

    return softmax_diff_norms

def softmax_robustness(original_outputs, adversarial_outputs):
    """ This robustness measure is global and it is stricly dependent on the epsilon chosen for the 
    perturbations."""

    softmax_differences = softmax_difference(original_outputs, adversarial_outputs)
    robustness = (torch.ones_like(softmax_differences)-softmax_differences)#.mean().item()
    print(f"avg softmax robustness = {robustness.mean().item():.2f}")
    return robustness


#######################
# adversarial attacks #
#######################

def fgsm_attack(model, image, label, hyperparams=None, n_samples=None, avg_posterior=False):

    epsilon = hyperparams["epsilon"] if hyperparams else 0.3

    image.requires_grad = True

    if n_samples or avg_posterior:
        output = model.forward(inputs=image, n_samples=n_samples, avg_posterior=avg_posterior)
    else:
        output = model.forward(inputs=image)

    loss = torch.nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()
    image_grad = image.grad.data

    perturbed_image = image + epsilon * image_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def pgd_attack(model, image, label, hyperparams=None, n_samples=None, avg_posterior=False):

    if hyperparams: 
        epsilon, alpha, iters = (hyperparams["epsilon"], 2/image.max(), 40)
    else:
        epsilon, alpha, iters = (0.5, 2/225, 40)

    original_image = copy.deepcopy(image)
    
    for i in range(iters):
        image.requires_grad = True  

        if n_samples or avg_posterior:
            output = model.forward(image, n_samples, avg_posterior)
        else:
            output = model.forward(image)

        loss = torch.nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()

        perturbed_image = image + alpha * image.grad.data.sign()
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        image = torch.clamp(original_image + eta, min=0, max=1).detach()

    perturbed_image = image
    return perturbed_image


def attack(net, x_test, y_test, dataset_name, device, method, filename, savedir=None,
           hyperparams=None, n_samples=None, avg_posterior=False):

    print(f"\nProducing {method} attacks on {dataset_name}:")

    adversarial_attack = []

    for idx in tqdm(range(len(x_test))):
        image = x_test[idx].unsqueeze(0).to(device)
        label = y_test[idx].argmax(-1).unsqueeze(0).to(device)

        if method == "fgsm":
            perturbed_image = fgsm_attack(model=net, image=image, label=label, 
                                          hyperparams=hyperparams, n_samples=n_samples,
                                          avg_posterior=avg_posterior)
        elif method == "pgd":
            perturbed_image = pgd_attack(model=net, image=image, label=label, 
                                          hyperparams=hyperparams, n_samples=n_samples,
                                          avg_posterior=avg_posterior)

        adversarial_attack.append(perturbed_image)

    # concatenate list of tensors 
    adversarial_attack = torch.cat(adversarial_attack)

    path = TESTS+filename+"/" if savedir is None else TESTS+savedir+"/"
    name = filename+"_"+str(method)
    name = name+"_attackSamp="+str(n_samples)+"_attack.pkl" if n_samples else name+"_attack.pkl"
    save_to_pickle(data=adversarial_attack, path=path, filename=name)
    return adversarial_attack

def load_attack(model, method, filename, savedir=None, n_samples=None, rel_path=TESTS):
    path = TESTS+filename+"/" if savedir is None else TESTS+savedir+"/"
    name = filename+"_"+str(method)
    name = name+"_attackSamp="+str(n_samples)+"_attack.pkl" if n_samples else name+"_attack.pkl"
    return load_from_pickle(path=path+name)

def attack_evaluation(model, x_test, x_attack, y_test, device, n_samples=None):

    if device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    print(f"\nEvaluating against the attacks", end="")
    if n_samples:
        print(f" with {n_samples} defence samples")

    random.seed(0)
    pyro.set_rng_seed(0)
    
    x_test = x_test.to(device)
    x_attack = x_attack.to(device)
    y_test = y_test.to(device)
    # model.to(device)
    if hasattr(model, 'net'):
        model.net.to(device) # fixed layers in BNN

    test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=128, shuffle=False)
    attack_loader = DataLoader(dataset=list(zip(x_attack, y_test)), batch_size=128, shuffle=False)

    with torch.no_grad():

        original_outputs = []
        original_correct = 0.0
        for images, labels in test_loader:
            out = model.forward(images, n_samples) if n_samples else model.forward(images)
            original_correct += ((out.argmax(-1) == labels.argmax(-1)).sum().item())
            original_outputs.append(out)

        adversarial_outputs = []
        adversarial_correct = 0.0
        for attacks, labels in attack_loader:
            out = model.forward(attacks, n_samples) if n_samples else model.forward(attacks)
            adversarial_correct += ((out.argmax(-1) == labels.argmax(-1)).sum().item())
            adversarial_outputs.append(out)

        original_accuracy = 100 * original_correct / len(x_test)
        adversarial_accuracy = 100 * adversarial_correct / len(x_test)
        print(f"\ntest accuracy = {original_accuracy}\tadversarial accuracy = {adversarial_accuracy}",
              end="\t")

        original_outputs = torch.cat(original_outputs)
        adversarial_outputs = torch.cat(adversarial_outputs)
        softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

    return original_accuracy, adversarial_accuracy, softmax_rob


def attack_increasing_eps(nn, bnn, dataset, device, method, n_inputs=100, n_samples=100, savedir=None):

    savedir = nn.savedir if hasattr(nn, 'savedir') else "attack"

    _, _, x_test, y_test, _, _ = load_dataset(dataset, n_inputs=n_inputs, shuffle=True)
    x_test, y_test = (torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device))

    df = pandas.DataFrame(columns=["attack", "epsilon", "test_acc", "adv_acc", 
                                   "softmax_rob", "model_type"])

    row_count = 0
    for epsilon in [0.1, 0.15, 0.2, 0.25, 0.3]:

        df_dict = {"epsilon":epsilon, "attack":method}
        hyperparams = {"epsilon":epsilon}

        ### attacking the base network
        x_attack = attack(net=nn, x_test=x_test, y_test=y_test, dataset_name=dataset, n_samples = 1,
                          device=device, method=method, filename=nn.name+"_eps="+str(epsilon), 
                          savedir=savedir, hyperparams=hyperparams)
        
        ### defending with both networks
        for net, model_type in [(nn,"nn"), (bnn, "bnn")]:
            n_samp = n_samples if model_type == "bnn" else None
            test_acc, adv_acc, softmax_rob = attack_evaluation(model=net, x_test=x_test, 
                        n_samples=n_samp, x_attack=x_attack, y_test=y_test, device=device)
            
            for pointwise_rob in softmax_rob:
                df_dict.update({"test_acc":test_acc, "adv_acc":adv_acc,
                                "softmax_rob":pointwise_rob.item(), "model_type":model_type})

                df.loc[row_count] = pandas.Series(df_dict)
                row_count += 1

    print("\nSaving:", df)
    os.makedirs(os.path.dirname(TESTS+savedir+"/"), exist_ok=True)
    df.to_csv(TESTS+savedir+"/"+str(dataset)+"_increasing_eps_"+str(method)+"_samp="+str(n_samples)+".csv", 
              index = False, header=True)
    return df

def plot_increasing_eps(df, dataset, method, n_samples):
    print(df)
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt

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

def attack_increasing_eps_avg_posterior(nn, bnn, dataset, device, method, n_inputs=100, savedir=None):

    savedir = nn.savedir if hasattr(nn, 'savedir') else "attack"

    _, _, x_test, y_test, _, _ = load_dataset(dataset, n_inputs=n_inputs, shuffle=True)
    x_test, y_test = (torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device))

    df = pandas.DataFrame(columns=["attack", "epsilon", "test_acc", "adv_acc", 
                                   "softmax_rob", "defence_samples"])

    row_count = 0
    for epsilon in [0.1, 0.15, 0.2, 0.25, 0.3]:

        df_dict = {"epsilon":epsilon, "attack":method}
        hyperparams = {"epsilon":epsilon}
        
        ### attacking the avg network
        x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, dataset_name=dataset, 
                          device=device, method=method, filename=nn.name+"_eps="+str(epsilon), 
                          savedir=savedir, hyperparams=hyperparams, avg_posterior=True)

        ### defending with different n_samples
        for n_samples in [1, 100, 500]:
            test_acc, adv_acc, softmax_rob = attack_evaluation(model=bnn, x_test=x_test, 
                        n_samples=n_samples, x_attack=x_attack, y_test=y_test, device=device)
            
            for pointwise_rob in softmax_rob:
                df_dict.update({"test_acc":test_acc, "adv_acc":adv_acc,
                                "softmax_rob":pointwise_rob.item(), "defence_samples":n_samples})

                df.loc[row_count] = pandas.Series(df_dict)
                row_count += 1

    print("\nSaving:", df)
    os.makedirs(os.path.dirname(TESTS+savedir+"/"), exist_ok=True)
    df.to_csv(TESTS+savedir+"/"+str(dataset)+"_increasing_eps_"+str(method)+"_avg_posterior.csv", 
              index = False, header=True)
    return df

def plot_increasing_eps_avg_posterior(df, dataset, method):
    print(df)
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt

    sns.set_style("darkgrid")
    palette = ["orange","darkred","black"]
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    plt.suptitle(f"{method} attack on {dataset}")
    sns.lineplot(data=df, x="epsilon", y="adv_acc", hue="defence_samples", 
                 palette=palette,style="defence_samples", ax=ax[0], legend="full")
    g = sns.lineplot(data=df, x="epsilon", y="softmax_rob", hue="defence_samples", 
                 palette=palette, style="defence_samples", ax=ax[1], legend=False)
    # g.legend(loc='upper right')
    
    filename = str(dataset)+"_increasing_eps_"+str(method)+"_avg_posterior.png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)


########
# main #
########

def main(args):

    _, _, x_test, y_test, inp_shape, out_size = \
                                load_dataset(dataset_name=args.dataset, n_inputs=args.inputs)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

    dataset, hid, activ, arch, ep, lr = saved_NNs["model_0"].values()
    nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size,
            hidden_size=hid, activation=activ, architecture=arch)
    nn.load(epochs=ep, lr=lr, device=args.device, rel_path=TESTS)

    # # x_attack = attack(net=nn, x_test=x_test, y_test=y_test, dataset_name=dataset, 
    # #                   device=args.device, method=args.attack, filename=nn.filename)
    # x_attack = load_attack(model=nn, method=args.attack, rel_path=DATA, filename=nn.filename)

    # attack_evaluation(model=nn, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

    # === BNN ===
    model = saved_BNNs["model_0"]
    dataset, init = list(model.values())[0], list(model.values())[1:]
    bnn = BNN(dataset, *init, inp_shape, out_size)
    bnn.load(device=args.device, rel_path=TESTS)

    # for attack_samples in [1,10,50]:
    #     x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, dataset_name=args.dataset, 
    #                       device=args.device, method=args.attack, filename=bnn.name, 
    #                       n_samples=attack_samples)

    #     for defence_samples in [attack_samples, 100]:
    #         attack_evaluation(model=bnn, x_test=x_test, x_attack=x_attack, y_test=y_test, 
    #                           device=args.device, n_samples=defence_samples)


    # === redBNN ===

    # rBNN = redBNN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size, 
    #              inference=args.inference, base_net=nn)
    # hyperparams = rBNN.get_hyperparams(args)
    # rBNN.load(n_inputs=args.inputs, hyperparams=hyperparams, device=args.device, rel_path=TESTS)
    # attack_evaluation(model=rBNN, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

    # === multiple attacks ===

    bnn_samples = 100
    # df = attack_increasing_eps(nn=nn, bnn=bnn, dataset=dataset, device=args.device, method=args.attack, n_samples=bnn_samples)
    # df = pandas.read_csv(TESTS+nn.savedir+"/"+str(dataset)+"_increasing_eps_"+str(args.attack)+"_samp="+str(bnn_samples)+".csv")
    # plot_increasing_eps(df, dataset=dataset, method=args.attack, n_samples=bnn_samples)

    # df = attack_increasing_eps_avg_posterior(nn=nn, bnn=bnn, dataset=dataset, device=args.device, method=args.attack)
    df = pandas.read_csv(TESTS+"attack/"+str(dataset)+"_increasing_eps_"+str(args.attack)+"_avg_posterior.csv")
    plot_increasing_eps_avg_posterior(df, dataset=dataset, method=args.attack)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="Adversarial attacks")

    parser.add_argument("--inputs", default=100, type=int)
    parser.add_argument("--dataset", default="half_moons", type=str, 
                        help="mnist, fashion_mnist, cifar, half_moons")
    parser.add_argument("--attack", default="fgsm", type=str, help="fgsm, pgd")
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")   
    main(args=parser.parse_args())
