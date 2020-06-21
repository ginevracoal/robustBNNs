"""
FGSM and PGD classic & bayesian adversarial attacks + robustness measures
"""

import sys
from savedir import *
from utils import *
import argparse
from tqdm import tqdm
import pyro
import random
import copy
import torch
from model_nn import NN, saved_NNs
from model_bnn import BNN, saved_BNNs
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as nnf
import pandas
import os

DEBUG=False

#######################
# robustness measures #
#######################


def softmax_difference(original_predictions, adversarial_predictions):
    """
    Compute the expected l-inf norm of the difference between predictions and adversarial 
    predictions. This is also a point-wise robustness measure.
    """

    original_predictions = nnf.softmax(original_predictions, dim=-1)
    adversarial_predictions = nnf.softmax(adversarial_predictions, dim=-1)

    if len(original_predictions) != len(adversarial_predictions):
        raise ValueError("\nInput arrays should have the same length.")

    if DEBUG:
        print("\n\n", original_predictions[0], "\t", adversarial_predictions[0], end="\n\n")

    softmax_diff = original_predictions-adversarial_predictions
    softmax_diff_norms = softmax_diff.abs().max(dim=-1)[0]

    if softmax_diff_norms.min() < 0. or softmax_diff_norms.max() > 1.:
        raise ValueError("Softmax difference should be in [0,1]")

    return softmax_diff_norms

def softmax_robustness(original_outputs, adversarial_outputs):
    """ 
    This robustness measure is global and it is stricly dependent on the epsilon chosen for the 
    perturbations.
    """

    softmax_differences = softmax_difference(original_outputs, adversarial_outputs)
    robustness = (torch.ones_like(softmax_differences)-softmax_differences)
    print(f"avg softmax robustness = {robustness.mean().item():.2f}")
    return robustness


#######################
# adversarial attacks #
#######################

def fgsm_attack(net, image, label, hyperparams=None, n_samples=None, avg_posterior=False):

    epsilon = hyperparams["epsilon"] if hyperparams is not None else 0.3

    image.requires_grad = True
    output = net.forward(inputs=image, n_samples=n_samples, avg_posterior=avg_posterior)

    loss = torch.nn.CrossEntropyLoss()(output, label)
    net.zero_grad()
    loss.backward()
    image_grad = image.grad.data

    perturbed_image = image + epsilon * image_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def pgd_attack(net, image, label, hyperparams=None, n_samples=None, avg_posterior=False):

    if hyperparams is not None: 
        epsilon, alpha, iters = (hyperparams["epsilon"], 2/image.max(), 40)
    else:
        epsilon, alpha, iters = (0.5, 2/225, 40)

    original_image = copy.deepcopy(image)
    
    for i in range(iters):
        image.requires_grad = True  
        output = net.forward(inputs=image, n_samples=n_samples, avg_posterior=avg_posterior)

        loss = torch.nn.CrossEntropyLoss()(output, label)
        net.zero_grad()
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
            perturbed_image = fgsm_attack(net=net, image=image, label=label, 
                                          hyperparams=hyperparams, n_samples=n_samples,
                                          avg_posterior=avg_posterior)
        elif method == "pgd":
            perturbed_image = pgd_attack(net=net, image=image, label=label, 
                                          hyperparams=hyperparams, n_samples=n_samples,
                                          avg_posterior=avg_posterior)

        adversarial_attack.append(perturbed_image)

    adversarial_attack = torch.cat(adversarial_attack)

    path = TESTS+filename+"/" if savedir is None else TESTS+savedir+"/"
    name = filename+"_"+str(method)
    name = name+"_attackSamp="+str(n_samples)+"_attack.pkl" if n_samples else name+"_attack.pkl"
    save_to_pickle(data=adversarial_attack, path=path, filename=name)
    return adversarial_attack

def load_attack(method, filename, savedir=None, n_samples=None, rel_path=TESTS):
    path = TESTS+filename+"/" if savedir is None else TESTS+savedir+"/"
    name = filename+"_"+str(method)
    name = name+"_attackSamp="+str(n_samples)+"_attack.pkl" if n_samples else name+"_attack.pkl"
    return load_from_pickle(path=path+name)

def attack_evaluation(net, x_test, x_attack, y_test, device, n_samples=None):

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

    if hasattr(net, 'net'):
        net.basenet.to(device) # fixed layers in BNN

    test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=128, shuffle=False)
    attack_loader = DataLoader(dataset=list(zip(x_attack, y_test)), batch_size=128, shuffle=False)

    with torch.no_grad():

        original_outputs = []
        original_correct = 0.0
        for images, labels in test_loader:
            out = net.forward(images, n_samples)
            original_correct += ((out.argmax(-1) == labels.argmax(-1)).sum().item())
            original_outputs.append(out)

        adversarial_outputs = []
        adversarial_correct = 0.0
        for attacks, labels in attack_loader:
            out = net.forward(attacks, n_samples)
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


########
# main #
########

def main(args):

    bayesian_attack_samples=[1,10,50]

    rel_path=DATA if args.savedir=="DATA" else TESTS
    train_inputs = 100 if DEBUG else None

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if args.deterministic:

        ### NN model
        dataset, hid, activ, arch, ep, lr = saved_NNs["model_"+str(args.model_idx)].values()

        x_train, y_train, x_test, y_test, inp_shape, out_size = \
            load_dataset(dataset_name=dataset, n_inputs=train_inputs)
        train_loader = DataLoader(dataset=list(zip(x_train, y_train)), shuffle=True)
        test_loader = DataLoader(dataset=list(zip(x_test, y_test)))

        nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size, 
                hidden_size=hid, activation=activ, architecture=arch, epochs=ep, lr=lr)

        if args.train:
            nn.train(train_loader=train_loader, device=args.device)
        else:
            nn.load(device=args.device, rel_path=rel_path)
        
        if args.test:
            nn.evaluate(test_loader=test_loader, device=args.device)

        ### attack NN
        if args.attack:
            x_test, y_test = (torch.from_numpy(x_test[:args.attack_inputs]), 
                              torch.from_numpy(y_test[:args.attack_inputs]))
            x_attack = attack(net=nn, x_test=x_test, y_test=y_test, dataset_name=dataset, 
                              device=args.device, method=args.attack_method, filename=nn.name)
        else:
            x_attack = load_attack(net=nn, method=args.attack_method, rel_path=DATA, filename=nn.name)

        attack_evaluation(net=nn, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                            device=args.device)

    else:

        ### BNN model
        dataset, model = saved_BNNs["model_"+str(args.model_idx)]
        batch_size = 5000 if model["inference"] == "hmc" else 128

        x_train, y_train, x_test, y_test, inp_shape, out_size = \
            load_dataset(dataset_name=dataset, n_inputs=train_inputs)
        train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size, 
                                  shuffle=True)
        test_loader = DataLoader(dataset=list(zip(x_test, y_test)))

        bnn = BNN(dataset, *list(model.values()), inp_shape, out_size)

        if args.train:
            bnn.train(train_loader=train_loader, device=args.device)
        else:
            bnn.load(device=args.device, rel_path=rel_path)

        if args.test:
            bnn.evaluate(test_loader=test_loader, device=args.device, n_samples=10)

        ### attack BNN
        x_test, y_test = (torch.from_numpy(x_test[:args.n_inputs]), 
                          torch.from_numpy(y_test[:args.n_inputs]))

        for attack_samples in bayesian_attack_samples:
            x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, dataset_name=dataset, 
                              device=args.device, method=args.attack_method, filename=bnn.name, 
                              n_samples=attack_samples)

            for defence_samples in [attack_samples]:
                attack_evaluation(net=bnn, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                                  device=args.device, n_samples=defence_samples)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", default=100, type=int, help="inputs to be attacked")
    parser.add_argument("--deterministic", default=False, type=eval, help="choose NN or BNN model")
    parser.add_argument("--model_idx", default=0, type=int, help="choose idx from saved_NNs")
    parser.add_argument("--train", default=True, type=eval)
    parser.add_argument("--test", default=True, type=eval)
    parser.add_argument("--attack", default=True, type=eval)
    parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
    parser.add_argument("--savedir", default='DATA', type=str, help="DATA, TESTS")  
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")   
    main(args=parser.parse_args())
